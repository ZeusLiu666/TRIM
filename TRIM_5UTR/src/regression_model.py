from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import copy


# ===== Metrics Calculation =====
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Mean Squared Error ignoring NaN values in the target.
    Returns 0.0 if all targets are NaN to avoid zero gradients.
    """
    mask = torch.isnan(target)
    if (~mask).sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return ((pred[~mask] - target[~mask]) ** 2).mean()


def r2_score_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes R2 Score ignoring NaN values in the target.
    """
    mask = torch.isnan(target)
    y = target[~mask]
    p = pred[~mask]

    if y.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    sse = torch.sum((p - y) ** 2)
    mean_y = torch.mean(y)
    sst = torch.sum((y - mean_y) ** 2) + 1e-8

    return 1 - (sse / sst)


class ParallelExpertsMixer(nn.Module):
    """
    Parallel Experts + Soft Gating Fusion.

    Input:  branches (N, F, B)  # F=Features, B=Branches (e.g., 3 or 4)
    Output: (N, fusion_hidden, B)

    Experts:
      1. ConvExpert: Depthwise 3x1 Conv + SE on the 'Branch' dimension.
      2. AttnExpert: MHSA on the 'Branch' dimension (treating branches as tokens).
      3. MLPExpert: Per-branch MLP (element-wise interaction within channels).

    Gating:
      - Global context (N, 2F) derived via mean/max pooling -> MLP -> Softmax weights [α1..αK].
    """

    def __init__(
        self,
        feat_dim: int,
        fusion_hidden: int,
        num_experts: int = 3,
        attn_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusion_hidden = fusion_hidden
        self.num_experts = num_experts

        # --- Expert 1: Convolutional Expert (Depthwise Conv + SE) ---
        self.conv_pw = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.conv_dw = nn.Conv1d(
            feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim, bias=False
        )
        self.conv_gn = nn.GroupNorm(num_groups=1, num_channels=feat_dim)
        self.conv_se = SEBlock(feat_dim, r=8)
        self.conv_act = nn.GELU()

        # --- Expert 2: Attention Expert (Treat branches as tokens) ---
        self.ln_attn = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=attn_heads, dropout=dropout, batch_first=True
        )
        self.ln_attn_ffn = nn.LayerNorm(feat_dim)
        self.attn_ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * mlp_ratio, feat_dim),
        )

        # --- Expert 3: MLP Expert (Per-branch feedforward) ---
        self.mlp_ln = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * mlp_ratio, feat_dim),
        )

        # Gating Network
        gate_hidden = max(feat_dim // 4, 32)
        self.gate = nn.Sequential(
            nn.Linear(2 * feat_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )

        # Output Projection: F -> fusion_hidden
        self.out_proj = nn.Conv1d(feat_dim, fusion_hidden, kernel_size=1, bias=False)

    def forward(self, branches: torch.Tensor) -> torch.Tensor:
        # branches: (N, F, B)
        N, F, B = branches.shape

        # ---------------- Expert 1: Conv ----------------
        x1 = self.conv_pw(branches)
        x1 = self.conv_dw(x1)
        x1 = self.conv_se(x1)
        x1 = self.conv_gn(x1)
        x1 = self.conv_act(x1)  # (N,F,B)

        # ---------------- Expert 2: Attn ----------------
        x2 = branches.transpose(1, 2)  # (N,B,F)
        h2 = self.ln_attn(x2)
        attn_out, _ = self.attn(h2, h2, h2, need_weights=False)
        x2 = x2 + attn_out
        h2 = self.ln_attn_ffn(x2)
        x2 = x2 + self.attn_ffn(h2)
        x2 = x2.transpose(1, 2)  # back to (N,F,B)

        # ---------------- Expert 3: MLP -----------------
        x3 = branches.transpose(1, 2)  # (N,B,F)
        h3 = self.mlp_ln(x3)
        x3 = x3 + self.mlp(h3)
        x3 = x3.transpose(1, 2)  # (N,F,B)

        # ---------------- Gating ------------------------
        # Global context: Mean + Max pooling along branches -> (N, 2F)
        mean_pool = branches.mean(dim=-1)
        max_pool = branches.amax(dim=-1)
        ctx = torch.cat([mean_pool, max_pool], dim=-1)
        logits = self.gate(ctx)  # (N, K)
        alpha = torch.softmax(logits, dim=-1)  # (N, K)

        # Stack experts: (N, K, F, B)
        experts = torch.stack([x1, x2, x3], dim=1)

        # Weighted sum: (N,K,1,1) * (N,K,F,B) -> sum_k -> (N,F,B)
        alpha_ = alpha.unsqueeze(-1).unsqueeze(-1)
        fused = (alpha_ * experts).sum(dim=1)

        # Final Projection
        out = self.out_proj(fused)  # (N, fusion_hidden, B)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (N,C,L)->(N,C,1)
            nn.Flatten(),
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: (N,C,L)
        s = self.fc(x).unsqueeze(-1)  # (N,C,1)
        return x * s


class ConvBlock(nn.Module):
    """Conv1D block with optional residual connection and pre/post-norm logic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        eps: float,
        dropout: float,
        residual: bool = False,
        is_initial: bool = False,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.residual = residual
        self.is_initial = is_initial
        self.activation = activation

        if self.is_initial:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            )
            if self.residual:
                self.residual_layernorm = nn.LayerNorm(
                    normalized_shape=out_channels, eps=eps
                )
                self.residual_conv = nn.Sequential(
                    self.activation(),
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding="same",
                    ),
                    nn.Dropout(dropout),
                )
                self.scale = nn.Parameter(torch.zeros(out_channels))
        else:
            self.layernorm = nn.LayerNorm(normalized_shape=in_channels, eps=eps)
            self.conv = nn.Sequential(
                self.activation(),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.Dropout(dropout),
            )
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

            if self.residual:
                self.residual_layernorm = nn.LayerNorm(
                    normalized_shape=out_channels, eps=eps
                )
                self.residual_conv = nn.Sequential(
                    self.activation(),
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding="same",
                    ),
                    nn.Dropout(dropout),
                )
                self.scale = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, L_in)
        Returns:
            Tensor of shape (N, C, L_out)
        """
        if self.is_initial:
            # Direct convolution for the initial layer
            x = self.conv(x)

            if self.residual:
                # Transpose for LayerNorm: (N, C, L) -> (N, L, C)
                xt = x.transpose(1, 2)
                xt = self.residual_layernorm(xt)
                xt = xt.transpose(1, 2)
                # Add residual with scaling
                x += self.residual_conv(xt) * self.scale[None, :, None]

            return x

        # Pre-activation / Pre-norm style for subsequent layers
        x = x.transpose(1, 2)
        x = self.layernorm(x)
        x = x.transpose(1, 2)

        x = self.conv(x)

        if self.residual:
            xt = x.transpose(1, 2)
            xt = self.residual_layernorm(xt)
            xt = xt.transpose(1, 2)
            x += self.residual_conv(xt) * self.scale[None, :, None]

        return self.pool(x)


class FeatureExtraction(nn.Module):
    """
    Backbone for sequence feature extraction.
    Input:  (N, Cin, L)
    Output: (N, Filters, L')
    """

    def __init__(
        self,
        in_channels: int = 5,
        filters: int = 256,
        kernel_size: int = 11,
        stride: int = 1,
        padding: int = 5,
        ln_epsilon: float = 1e-5,
        dropout: float = 0.1,
        residual: bool = True,
        num_conv_layers: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        self.activation = nn.ReLU
        self.initial_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            eps=ln_epsilon,
            dropout=dropout,
            residual=residual,
            is_initial=True,
            activation=self.activation,
        )
        self.middle_convs = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    eps=ln_epsilon,
                    dropout=dropout,
                    residual=residual,
                    is_initial=False,
                    activation=self.activation,
                )
                for _ in range(num_conv_layers)
            ]
        )

    def forward(self, x):
        h = self.initial_conv(x)
        for blk in self.middle_convs:
            h = blk(h)
        return h


class ResidualMLPBlock(nn.Module):
    """ResNet-style MLP block with LayerNorm and GELU."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class EnvEncoder(nn.Module):
    """
    Encoder for environmental variables using residual MLPs.
    """

    def __init__(
        self,
        env_input_size: int,
        width: int = 256,
        depth: int = 3,
        env_out_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(env_input_size, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(dim=width, hidden=width * 2, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Sequential(nn.Linear(width, env_out_dim), nn.GELU())

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(env)
        for blk in self.blocks:
            x = blk(x)
        x = self.proj_out(x)
        return x


class RegBlock(nn.Module):
    """
    Pre-LN Transformer Block for Regression Head.
    Input/Output: (N, 4, C)
    """

    def __init__(
        self, dim: int, n_heads: int, mlp_ratio: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop_attn = nn.Dropout(dropout)

        hidden = dim * mlp_ratio
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # MHSA with Residual
        h = self.ln_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(attn_out)
        # FFN with Residual
        h = self.ln_ffn(x)
        x = x + self.ffn(h)
        return x


class RegHead(nn.Module):
    """
    Regression Head using Multi-layer MHSA.
    Input: (N, C, 4) -> Transposes to (N, 4, C) internally
    Output: (N,)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.2,
        depth: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        # Learnable embeddings for the 4 branches
        self.branch_embed = nn.Parameter(torch.zeros(4, dim))
        nn.init.trunc_normal_(self.branch_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                RegBlock(dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )

        # Attention Pooling (Learnable Query)
        self.pool_q = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.trunc_normal_(self.pool_q, std=0.02)
        self.pool_proj = nn.Linear(dim, dim, bias=False)

        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):  # x: (N, C, 4)
        # 1) Transpose -> (N, 4, C)
        x = x.transpose(1, 2)
        # 2) Add Branch Embeddings
        x = x + self.branch_embed.unsqueeze(0)

        # 3) Transformer Layers
        for blk in self.blocks:
            x = blk(x)

        # 4) Attention Pooling: Aggregate 4 tokens into 1 vector
        q = self.pool_q.expand(x.size(0), -1, -1)  # (N, 1, C)
        k = self.pool_proj(x)  # (N, 4, C)

        attn_score = torch.softmax(
            (q @ k.transpose(1, 2)) / (self.dim**0.5), dim=-1
        )  # (N, 1, 4)

        z = attn_score @ x  # (N, 1, C)
        z = z.squeeze(1)  # (N, C)

        # 5) Regression
        y = self.out(z).squeeze(-1)
        return y


class Regression(pl.LightningModule):
    """
    Main Transfer Learning Model for Translation Efficiency Prediction.

    Architecture:
    1. Separate CNN Backbones for 5'UTR, CDS, and 3'UTR.
    2. Environment Variable Encoder.
    3. Parallel Experts Mixer for fusing the 4 branches.
    4. Transformer-based Regression Head.
    """

    def __init__(
        self,
        # === Feature Extraction Hyperparameters ===
        FE_in_channels: int,
        FE_filters: int,
        FE_kernel_size: int,
        FE_conv_stride: int,
        FE_conv_padding: int,
        FE_ln_epsilon: float,
        FE_dropout_conv: float,
        FE_residual: bool,
        FE_num_conv_layers: int,
        FE_activation: str,
        # === Environment Encoder Params ===
        ENV_input_size: int,
        ENV_width: int,
        ENV_depth: int,
        ENV_out_dim: int,
        ENV_dropout: float,
        # === Regression Params ===
        REG_dropout: float,
        REG_use_huber: bool,
        REG_huber_beta: float,
        REG_use_r2_in_loss: bool,  # Include (1 - R2) in loss
        REG_r2_lambda: float,  # Weight for R2 loss
        # === Training Params ===
        REG_lr: float,
        REG_filters: int,
        REG_warmup_epochs: int,
        REG_eta_min: float,
        # === RegHead Params ===
        REG_fusion_hidden: int,
        REG_RegHead_heads: int,
        REG_RegHead_depth: int,
        REG_mlp_ratio: int,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Define Backbones (Reusing the TRIM structure)
        self.backbone_5utr = FeatureExtraction(
            in_channels=FE_in_channels,
            filters=FE_filters,
            kernel_size=FE_kernel_size,
            stride=FE_conv_stride,
            padding=FE_conv_padding,
            ln_epsilon=FE_ln_epsilon,
            dropout=FE_dropout_conv,
            residual=FE_residual,
            num_conv_layers=FE_num_conv_layers,
            activation=FE_activation,
        )
        self.backbone_cds = FeatureExtraction(
            in_channels=FE_in_channels,
            filters=FE_filters,
            kernel_size=FE_kernel_size,
            stride=FE_conv_stride,
            padding=FE_conv_padding,
            ln_epsilon=FE_ln_epsilon,
            dropout=FE_dropout_conv,
            residual=FE_residual,
            num_conv_layers=FE_num_conv_layers,
            activation=FE_activation,
        )
        self.backbone_3utr = FeatureExtraction(
            in_channels=FE_in_channels,
            filters=FE_filters,
            kernel_size=FE_kernel_size,
            stride=FE_conv_stride,
            padding=FE_conv_padding,
            ln_epsilon=FE_ln_epsilon,
            dropout=FE_dropout_conv,
            residual=FE_residual,
            num_conv_layers=FE_num_conv_layers,
            activation=FE_activation,
        )

        # 2) Environment Encoder
        self.env_enc = EnvEncoder(
            env_input_size=ENV_input_size,
            width=ENV_width,
            depth=ENV_depth,
            env_out_dim=ENV_out_dim,
            dropout=ENV_dropout,
        )

        # 3) Regression Head
        self.reg_head = RegHead(
            dim=REG_fusion_hidden,
            n_heads=REG_RegHead_heads,
            mlp_ratio=REG_mlp_ratio,
            depth=REG_RegHead_depth,
            dropout=REG_dropout,
        )

        # Projections to unify dimensions
        self.lr = REG_lr
        self.filters = REG_filters

        self.proj_env_to_F = nn.Sequential(
            nn.Linear(ENV_out_dim, self.filters), nn.GELU()
        )

        self.seq_proj_to_F = nn.Sequential(
            nn.Linear(FE_filters, self.filters), nn.GELU()
        )

        # 4) Parallel Experts Mixer (Branch Mixing)
        self.branch_mixer = ParallelExpertsMixer(
            feat_dim=self.filters,
            fusion_hidden=REG_fusion_hidden,
            num_experts=3,
            attn_heads=4,
            mlp_ratio=2,
            dropout=0.3,
        )

        # Scheduler Params
        self.warmup_epochs = REG_warmup_epochs
        self.eta_min = REG_eta_min

        # Loss Config
        self.use_huber = REG_use_huber
        self.huber_beta = REG_huber_beta
        self.use_r2_in_loss = REG_use_r2_in_loss
        self.r2_lambda = float(REG_r2_lambda)

        # Buffers for target statistics (persistent state)
        self.register_buffer("y_mu", torch.tensor(0.0), persistent=True)
        self.register_buffer("y_std", torch.tensor(1.0), persistent=True)

        self._val_cache = None

    def forward(self, x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        # Split input into regions (Channels 0-3: 5'UTR, 4-7: CDS, 8-11: 3'UTR)
        x_5utr = x[:, 0:4, :]
        x_cds = x[:, 4:8, :]
        x_3utr = x[:, 8:12, :]

        # Extract auxiliary features (p_paired, codon scores)
        u5_pp = x[:, 12:13, :]
        cds_ex = x[:, 13:14, :]
        u3_pp = x[:, 14:15, :]

        # Concatenate aux features (In-channels becomes 5)
        x_5utr = torch.cat([x_5utr, u5_pp], dim=1)
        x_cds = torch.cat([x_cds, cds_ex], dim=1)
        x_3utr = torch.cat([x_3utr, u3_pp], dim=1)

        # Backbone Feature Extraction
        f_5utr = self.backbone_5utr(x_5utr)
        f_cds = self.backbone_cds(x_cds)
        f_3utr = self.backbone_3utr(x_3utr)

        # Global Average Pooling -> (N, F)
        v_5utr = f_5utr.mean(dim=-1)
        v_cds = f_cds.mean(dim=-1)
        v_3utr = f_3utr.mean(dim=-1)

        # Projection to common dimension
        v_5utr = self.seq_proj_to_F(v_5utr)
        v_cds = self.seq_proj_to_F(v_cds)
        v_3utr = self.seq_proj_to_F(v_3utr)

        # Encode Environment
        v_env = self.env_enc(env)
        v_env = self.proj_env_to_F(v_env)

        # Stack Branches: [5UTR, CDS, 3UTR, ENV] -> (N, F, 4)
        branches = torch.stack([v_5utr, v_cds, v_3utr, v_env], dim=-1)

        # Fuse Branches via Mixer
        fused = self.branch_mixer(branches)  # (N, fusion_hidden, 4)

        # Regression Head
        te_pred = self.reg_head(fused)

        return te_pred

    def on_fit_start(self):
        """
        Calculate global mean/std of targets using the training set to avoid data leakage.
        This is used for potential z-score scaling logic if needed.
        """
        assert (
            self.trainer is not None and self.trainer.datamodule is not None
        ), "DataModule required for on_fit_start"

        y_list = []
        for b in self.trainer.datamodule.train_dataloader():
            yt = b["TE_value"].float()
            mask = torch.isfinite(yt)
            if mask.any():
                y_list.append(yt[mask])

        y_all = torch.cat(y_list, dim=0).to(self.device)
        mu = y_all.mean()
        std = y_all.std(unbiased=False).clamp_min(1e-6)

        self.y_mu.copy_(mu.to(self.y_mu.device, dtype=self.y_mu.dtype))
        self.y_std.copy_(std.to(self.y_std.device, dtype=self.y_std.dtype))

    def training_step(self, batch, batch_idx):
        x = batch["x"].to(self.device).float()
        env = batch["env"].to(self.device).float()
        yt = batch["TE_value"].to(self.device).float()

        yhat = self(x, env)

        # Mask NaN/Inf values
        mask = torch.isfinite(yt).view(-1).to(self.device)

        if mask.any():
            if self.use_huber:
                loss = F.smooth_l1_loss(yhat[mask], yt[mask], beta=self.huber_beta)
            else:
                loss = F.mse_loss(yhat[mask], yt[mask])
        else:
            loss = torch.tensor(0.0, device=self.device)

        # Calculate Metrics
        if mask.any():
            mse = masked_mse_loss(yhat[mask], yt[mask])
            r2 = r2_score_torch(yhat[mask], yt[mask])

            # Pearson Correlation
            vx = yhat[mask] - yhat[mask].mean()
            vy = yt[mask] - yt[mask].mean()
            pearson = (vx * vy).sum() / (
                vx.square().sum() * vy.square().sum()
            ).sqrt().clamp_min(1e-8)
        else:
            mse = r2 = pearson = torch.tensor(0.0, device=self.device)

        # Composite Loss: Loss + Lambda * (1 - R2)
        if self.use_r2_in_loss and mask.any():
            loss = (1.0 - self.r2_lambda) * loss + self.r2_lambda * (1.0 - r2)

        self.log_dict(
            {
                "train/loss_reg": loss,
                "train/mse_reg": mse,
                "train/r2_reg": r2,
                "train/pearson_reg": pearson,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"].to(self.device).float()
        env = batch["env"].to(self.device).float()
        yt = batch["TE_value"].to(self.device).float()

        yhat = self(x, env)

        mask = torch.isfinite(yt).view(-1).to(self.device)

        if mask.any():
            if self.use_huber:
                loss = F.smooth_l1_loss(yhat[mask], yt[mask], beta=self.huber_beta)
            else:
                loss = F.mse_loss(yhat[mask], yt[mask])
        else:
            loss = torch.tensor(0.0, device=self.device)

        if mask.any():
            mse = masked_mse_loss(yhat[mask], yt[mask])
            r2 = r2_score_torch(yhat[mask], yt[mask])
            vx = yhat[mask] - yhat[mask].mean()
            vy = yt[mask] - yt[mask].mean()
            pearson = (vx * vy).sum() / (
                vx.square().sum() * vy.square().sum()
            ).sqrt().clamp_min(1e-8)
        else:
            mse = r2 = pearson = torch.tensor(0.0, device=self.device)

        if self.use_r2_in_loss and mask.any():
            loss = (1.0 - self.r2_lambda) * loss + self.r2_lambda * (1.0 - r2)

        self.log_dict(
            {
                "val/loss_reg": loss,
                "val/mse_reg": mse,
                "val/r2_reg": r2,
                "val/pearson_reg": pearson,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return {"y": yt.detach(), "yhat": yhat.detach(), "mask": mask.detach()}

    def validation_epoch_end(self, outputs):
        """
        Aggregate validation results at the end of epoch to calculate global metrics using NumPy.
        NumPy is preferred here for numerical stability over the entire dataset.
        """
        with torch.no_grad():
            y_list = [o["y"].view(-1) for o in outputs]
            yhat_list = [o["yhat"].view(-1) for o in outputs]
            m_list = [o["mask"].view(-1) for o in outputs]

            y = torch.cat(y_list, dim=0)
            yhat = torch.cat(yhat_list, dim=0)
            mask = torch.cat(m_list, dim=0).bool()

            y = y[mask].detach().float().cpu().numpy()
            yhat = yhat[mask].detach().float().cpu().numpy()

            N = int(y.size)
            if N >= 2:
                r = float(np.corrcoef(yhat, y)[0, 1])
                sse = float(np.sum((yhat - y) ** 2))
                sst = float(np.sum((y - y.mean()) ** 2)) + 1e-8
                r2 = float(1.0 - sse / sst)
                mse = float(sse / N)
            elif N == 1:
                r = 0.0
                r2 = 0.0
                mse = float((yhat - y) ** 2)
            else:
                r = r2 = mse = 0.0

            # Cache results for callbacks
            self._val_cache = {
                "y": y,
                "yhat": yhat,
                "pearson_reg": r,
                "r2_reg": r2,
                "mse_reg": mse,
                "N": N,
            }

    def configure_optimizers(self):
        # Collect parameters
        params = []
        params += list(self.env_enc.parameters())
        params += list(self.seq_proj_to_F.parameters())
        params += list(self.proj_env_to_F.parameters())
        params += list(self.branch_mixer.parameters())
        params += list(self.reg_head.parameters())

        # Backbone parameters (included here, could use different LR if needed)
        params += list(self.backbone_5utr.parameters())
        params += list(self.backbone_cds.parameters())
        params += list(self.backbone_3utr.parameters())

        optim = AdamW(
            [
                {"params": params, "lr": self.lr, "weight_decay": 1e-4},
            ]
        )

        # Learning Rate Scheduler: Linear Warmup -> Cosine Annealing
        if self.trainer is not None and self.trainer.max_epochs is not None:
            total_epochs = int(self.trainer.max_epochs)
        else:
            raise KeyError("Max epochs must be defined for scheduler configuration.")

        cosine_epochs = max(total_epochs - self.warmup_epochs, 1)

        warmup = LinearLR(
            optim,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine = CosineAnnealingLR(optim, T_max=cosine_epochs, eta_min=self.eta_min)

        scheduler = SequentialLR(
            optim,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "warmup_cosine",
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step. Returns unmasked predictions.
        Masking for non-expressing proteins should be handled downstream if needed.
        """
        yhat = self(batch["x"], batch["env"])
        return {"te": yhat}
