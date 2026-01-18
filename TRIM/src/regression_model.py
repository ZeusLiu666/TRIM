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


# ===== MSE and R2 Calculation =====
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss with masking for NaN targets.
    """
    mask = torch.isnan(target)
    if (~mask).sum() == 0:
        # Return 0 if all targets are NaN to avoid 0 gradient (constant loss)
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return ((pred[~mask] - target[~mask]) ** 2).mean()


def r2_score_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute R2 score.
    """
    mask = torch.isnan(target)
    y = target[~mask]
    p = pred[~mask]
    if y.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    sse = torch.sum((p - y) ** 2)
    mean_y = torch.mean(y)
    sst = torch.sum((y - mean_y) ** 2) + 1e-8  # Add epsilon to avoid division by zero
    return 1 - (sse / sst)


"""
DualHeadTransfer defines the model and training logic.
Functions provided:
* __init__: Receives pre-trained backbone, calls ENV encoder, implements FiLM modulator, defines classification/regression heads.
* forward: Gets encoded ENV sequence and TRIM backbone output (UTR+CDS), passes through FiLM modulator to heads.
* training_step/validation_step: Calculates loss and logs metrics using dataset sequences and model predictions.
* configure_optimizers: Sets up learning rate.
* predict_step: Prediction step.
"""


class ParallelExpertsMixer(nn.Module):
    """
    Parallel Experts + Soft Gating Fusion:
    Input: branches (N, F, B)     # F=feature dim (filters), B=num_branches (3 or 4)
    Output: (N, fusion_hidden, B)

    Experts:
      - ConvExpert: Depthwise 3x1 Conv + SE on "branch dimension".
      - AttnExpert: MHSA on "branch dimension" (treating branches as tokens).
      - MLPExpert: Per-branch MLP modeling intra-channel interactions (no cross-branch attention).
    Gating:
      - Mean/max pooling from (N, F, B) -> global context (N, 2F), then MLP -> [α1..αK] (softmax).
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

        # --- Expert 1: Conv Expert (Depthwise conv + SE on branch dim B) ---
        self.conv_pw = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.conv_dw = nn.Conv1d(
            feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim, bias=False
        )
        self.conv_gn = nn.GroupNorm(num_groups=1, num_channels=feat_dim)
        self.conv_se = SEBlock(feat_dim, r=8)
        self.conv_act = nn.GELU()

        # --- Expert 2: Attention Expert (Treat branches as tokens for MHSA) ---
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

        # --- Expert 3: MLP Expert (Per-branch, intra-channel feed-forward) ---
        self.mlp_ln = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * mlp_ratio, feat_dim),
        )

        # Gating: Generate num_experts weights from concatenated mean/max pooling (N, 2F)
        gate_hidden = max(feat_dim // 4, 32)
        self.gate = nn.Sequential(
            nn.Linear(2 * feat_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )

        # Output projection: F -> fusion_hidden
        self.out_proj = nn.Conv1d(feat_dim, fusion_hidden, kernel_size=1, bias=False)

    def forward(self, branches: torch.Tensor) -> torch.Tensor:
        # branches: (N, F, B)
        N, F, B = branches.shape
        # ---------------- Expert 1: Conv ----------------
        x1 = self.conv_pw(branches)  # (N,F,B)
        x1 = self.conv_dw(x1)  # (N,F,B)
        x1 = self.conv_se(x1)  # (N,F,B)
        x1 = self.conv_gn(x1)
        x1 = self.conv_act(x1)  # (N,F,B)

        # ---------------- Expert 2: Attn ----------------
        x2 = branches.transpose(1, 2)  # (N,B,F) as tokens
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
        # Global context: mean/max along branch dim -> (N, F)
        mean_pool = branches.mean(dim=-1)
        max_pool = branches.amax(dim=-1)
        ctx = torch.cat([mean_pool, max_pool], dim=-1)  # (N, 2F)
        logits = self.gate(ctx)  # (N, K)
        alpha = torch.softmax(logits, dim=-1)  # (N, K), sum=1

        # Stack experts output (N, K, F, B), fuse by alpha
        experts = torch.stack([x1, x2, x3], dim=1)  # (N, 3, F, B)
        # (N,K,1,1) * (N,K,F,B) -> sum_k -> (N,F,B)
        alpha_ = alpha.unsqueeze(-1).unsqueeze(-1)
        fused = (alpha_ * experts).sum(dim=1)  # (N,F,B)

        # Map output to fusion_hidden
        out = self.out_proj(fused)  # (N,fusion_hidden,B)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: Channel-wise attention scaling."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (N,C,L)->(N,C,1)
            nn.Flatten(),  # (N,C)
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: (N,C,L)
        s = self.fc(x).unsqueeze(-1)  # (N,C,1)
        return x * s


# ============== Conv Block (Replicating TRIM Backbone Style) ==============
class ConvBlock(nn.Module):
    """Conv1D block with or without residual layers"""

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
                # (N, L, C) --> (N, L, C), normalize across the last dimension
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
            # (N, L, C) --> (N, L, C), normalize across the last dimension
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
                # (N, L, C) --> (N, L, C), normalize across the last dimension
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
        """The forward pass

        Args:
            x (Tensor of shape: (N, C, L1)): input tensor

        Returns:
            Tensor of shape: (N, C, L2)
        """
        if self.is_initial:
            # (N, C, L1) --> (N, C, L2)
            x = self.conv(x)

            if self.residual:
                # (N, C, L) --> (N, L, C)
                xt = x.transpose(1, 2)
                # (N, L, C) --> (N, L, C)
                xt = self.residual_layernorm(xt)
                # (N, L, C) --> (N, C, L)
                xt = xt.transpose(1, 2)
                # scale in the channel dimension
                x += self.residual_conv(xt) * self.scale[None, :, None]

            return x

        # (N, C, L) --> (N, L, C)
        x = x.transpose(1, 2)
        # (N, L, C) --> (N, L, C)
        x = self.layernorm(x)
        # (N, L, C) --> (N, C, L)
        x = x.transpose(1, 2)
        # (N, C, L1) --> (N, C, L2)
        x = self.conv(x)

        if self.residual:
            # (N, C, L) --> (N, L, C)
            xt = x.transpose(1, 2)
            # (N, L, C) --> (N, L, C)
            xt = self.residual_layernorm(xt)
            # (N, L, C) --> (N, C, L)
            xt = xt.transpose(1, 2)
            # scale in the channel dimension
            x += self.residual_conv(xt) * self.scale[None, :, None]

        return self.pool(x)


# ============== FeatureExtraction Backbone (Replicating TRIM initial+middle) ==============
class FeatureExtraction(nn.Module):
    """
    Feature Extraction only (No RNN / Dense / Final layers):
      input:  (N, Cin=5, L)  <- Extra mask channel appended
      output: (N, F, L')     <- F=filters
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

    def forward(self, x):  # (N, Cin, L) -> (N, F, L')
        h = self.initial_conv(x)
        for blk in self.middle_convs:
            h = blk(h)
        return h


# 1. Residual Block to prevent overfitting
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # LayerNorm is more stable for small batches
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()  # GELU is better for deep MLPs
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h  # Residual connection


# 2. Environment Variable Encoder Block
class EnvEncoder(nn.Module):
    def __init__(
        self,
        env_input_size: int,  # Input dimension after ENV encoding
        width: int = 256,  # Width of intermediate layers
        depth: int = 3,  # Number of residual blocks
        env_out_dim: int = 128,  # Output dimension of encoded ENV (concatenated with backbone features)
        dropout: float = 0.3,  # Dropout probability
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

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(env)
        for blk in self.blocks:
            x = blk(x)
        x = self.proj_out(x)  # Output dimension: (N, env_out_dim)
        return x


class RegBlock(nn.Module):
    """
    Single Pre-LN Transformer Block:
      x -> LN -> MHSA -> Res -> LN -> FFN -> Res
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

    def forward(self, x):  # (N, 4, C)
        # MHSA Residual
        h = self.ln_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(attn_out)
        # FFN Residual
        h = self.ln_ffn(x)
        x = x + self.ffn(h)
        return x


class RegHead(nn.Module):
    """
    Multi-layer MHSA Regression Head (Branch dimension B=4 as token dimension)
      Input: (N, C, 4)  -> Transpose to (N, 4, C)
      Output: (N,)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,  # Number of attention heads
        mlp_ratio: int = 2,
        dropout: float = 0.2,
        depth: int = 4,  # Number of attention layers
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        # Learnable branch type embedding
        self.branch_embed = nn.Parameter(torch.zeros(4, dim))
        nn.init.trunc_normal_(self.branch_embed, std=0.02)

        # Stacked MHSA+FFN blocks
        self.blocks = nn.ModuleList(
            [
                RegBlock(dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )

        # Attention Pooling (Learnable Query Vector)
        self.pool_q = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.trunc_normal_(self.pool_q, std=0.02)
        self.pool_proj = nn.Linear(dim, dim, bias=False)

        # Output Layer
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):  # x: (N, C, 4)
        # 1) Swap channel and token dims -> (N, 4, C)
        x = x.transpose(1, 2)
        # 2) Add branch embeddings
        x = x + self.branch_embed.unsqueeze(0)

        # 3) Pass through L stacked Transformer blocks
        for blk in self.blocks:
            x = blk(x)  # (N, 4, C)

        # 4) Attention Pooling (Learnable Query Vector vs 4 tokens)
        q = self.pool_q.expand(x.size(0), -1, -1)  # (N, 1, C)
        k = self.pool_proj(x)  # (N, 4, C)
        attn_score = torch.softmax(
            (q @ k.transpose(1, 2)) / (self.dim**0.5), dim=-1
        )  # (N,1,4)
        z = attn_score @ x  # (N,1,C)
        z = z.squeeze(1)  # (N,C)

        # 5) Regress to scalar
        y = self.out(z).squeeze(-1)  # (N,)
        return y


# Transfer Learning Backbone
class Regression(pl.LightningModule):

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
        # === Internal Network Parameters ===
        ENV_input_size: int,
        ENV_width: int,
        ENV_depth: int,
        ENV_out_dim: int,
        ENV_dropout: float,
        # === Regression Parameters ===
        REG_dropout: float,
        # === Huber Loss Configuration ===
        REG_use_huber: bool,
        REG_huber_beta: float,
        # === Include R2 in Loss ===
        REG_use_r2_in_loss: bool,  # Include (1 - R2) in loss
        REG_r2_lambda: float,  # Weight lambda
        # === Learning Rate ===
        REG_lr: float,
        REG_filters: int,
        # LR Scheduler
        REG_warmup_epochs: int,
        REG_eta_min: float,  # Cosine annealing min LR
        # RegHead Parameters
        REG_fusion_hidden: int,  # Regression head dimension
        REG_RegHead_heads: int,  # Number of attention heads
        REG_RegHead_depth: int,  # Attention depth
        REG_mlp_ratio: int,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Three "TRIM isomorphic backbones" for each branch
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

        # Env Encoder
        self.env_enc = EnvEncoder(
            env_input_size=ENV_input_size,
            width=ENV_width,
            depth=ENV_depth,
            env_out_dim=ENV_out_dim,
            dropout=ENV_dropout,
        )

        # Regression Head
        self.reg_head = RegHead(
            dim=REG_fusion_hidden,
            n_heads=REG_RegHead_heads,
            mlp_ratio=REG_mlp_ratio,
            depth=REG_RegHead_depth,
            dropout=REG_dropout,
        )

        # Training Strategy
        self.lr = REG_lr
        self.filters = REG_filters

        # Projections
        self.proj_env_to_F = nn.Sequential(
            nn.Linear(ENV_out_dim, self.filters), nn.GELU()
        )  # Project encoded ENV vector to filters dim

        self.seq_proj_to_F = nn.Sequential(
            nn.Linear(FE_filters, self.filters), nn.GELU()
        )  # Project feature extraction output to filters dim

        # Branch Mixer: 3-branch convolution fusion
        self.branch_mixer = ParallelExpertsMixer(
            feat_dim=self.filters,
            fusion_hidden=REG_fusion_hidden,
            num_experts=3,
            attn_heads=4,
            mlp_ratio=2,
            dropout=0.3,
        )

        # LR Scheduler
        self.warmup_epochs = REG_warmup_epochs
        self.eta_min = REG_eta_min

        # Loss Config
        self.use_huber = REG_use_huber
        self.huber_beta = REG_huber_beta
        self.use_r2_in_loss = REG_use_r2_in_loss
        self.r2_lambda = float(REG_r2_lambda)

        # Register buffers (saved in state_dict but not trained)
        self.register_buffer("y_mu", torch.tensor(0.0), persistent=True)
        self.register_buffer("y_std", torch.tensor(1.0), persistent=True)

        # Validation cache
        self._val_cache = None

    def forward(self, x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:

        # Extract 5-channel inputs from data
        x_5utr = x[:, 0:4, :]
        x_cds = x[:, 4:8, :]
        x_3utr = x[:, 8:12, :]

        # Secondary structure / Auxiliary channels
        u5_pp = x[:, 12:13, :]  # 5'UTR p_paired
        cds_ex = x[:, 13:14, :]  # CDS Triplet/Bias Aux
        u3_pp = x[:, 14:15, :]  # 3'UTR p_paired

        x_5utr = torch.cat([x_5utr, u5_pp], dim=1)
        x_cds = torch.cat([x_cds, cds_ex], dim=1)
        x_3utr = torch.cat([x_3utr, u3_pp], dim=1)

        # Pass through backbones
        f_5utr = self.backbone_5utr(x_5utr)
        f_cds = self.backbone_cds(x_cds)
        f_3utr = self.backbone_3utr(x_3utr)

        v_5utr = f_5utr.mean(dim=-1)
        v_cds = f_cds.mean(dim=-1)
        v_3utr = f_3utr.mean(dim=-1)

        v_5utr = self.seq_proj_to_F(v_5utr)
        v_cds = self.seq_proj_to_F(v_cds)
        v_3utr = self.seq_proj_to_F(v_3utr)

        v_env = self.env_enc(env)
        v_env = self.proj_env_to_F(v_env)

        # Stack branches: [5UTR, CDS, 3UTR, ENV] -> (N, F, 4)
        branches = torch.stack([v_5utr, v_cds, v_3utr, v_env], dim=-1)

        # Mix branches using Experts
        fused = self.branch_mixer(branches)  # (N, fusion_hidden, 4)

        te_pred = self.reg_head(fused)  # Prediction

        return te_pred

    def on_fit_start(self):
        """
        Fit target mu/std using only training set to avoid leakage.
        """
        assert (
            self.trainer is not None and self.trainer.datamodule is not None
        ), "Datamodule required"
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
        # Keys match TRIMDataset __getitem__
        x, env = (
            batch["x"].to(self.device).float(),
            batch["env"].to(self.device).float(),
        )

        yt = batch["TE_value"].to(self.device).float()

        yhat = self(x, env)

        # Filter NaN / Inf
        mask = torch.isfinite(yt).view(-1).to(self.device)

        # Compute Loss
        if mask.any():
            if self.use_huber:
                loss = F.smooth_l1_loss(yhat[mask], yt[mask], beta=self.huber_beta)
            else:
                loss = F.mse_loss(yhat[mask], yt[mask])
        else:
            loss = torch.tensor(0.0, device=self.device)

        # Compute Metrics
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

        # Composite Loss
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
        x, env = (
            batch["x"].to(self.device).float(),
            batch["env"].to(self.device).float(),
        )

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
                # Global metrics
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

            # Cache for callbacks
            self._val_cache = {
                "y": y,
                "yhat": yhat,
                "pearson_reg": r,
                "r2_reg": r2,
                "mse_reg": mse,
                "N": N,
            }

    def configure_optimizers(self):
        params = []
        params += list(self.env_enc.parameters())
        params += list(self.seq_proj_to_F.parameters())
        params += list(self.proj_env_to_F.parameters())
        params += list(self.branch_mixer.parameters())
        params += list(self.reg_head.parameters())

        # Backbone params
        params += list(self.backbone_5utr.parameters())
        params += list(self.backbone_cds.parameters())
        params += list(self.backbone_3utr.parameters())

        optim = AdamW(
            [
                {"params": params, "lr": self.lr, "weight_decay": 1e-4},
            ]
        )

        # ------- 2) Warmup -> Cosine Scheduler -------
        if self.trainer is not None and self.trainer.max_epochs is not None:
            total_epochs = int(self.trainer.max_epochs)
        else:
            raise KeyError("Missing total epochs")

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
        # Note: No masking for non-expressing proteins here.
        yhat = self(batch["x"], batch["env"])
        return {"te": yhat}
