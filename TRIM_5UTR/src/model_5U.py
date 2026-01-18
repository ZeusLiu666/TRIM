import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Reuse FeatureExtraction and metrics from the main model
from src.regression_model import (
    FeatureExtraction,
    masked_mse_loss,
    r2_score_torch,
)


class Expert_5U(pl.LightningModule):
    """
    Small Expert Model focusing on the 200nt context around the start codon.

    Input:  x200: (N, 4, 200) - One-hot encoded sequence
    Output: TE_value prediction (N,)

    Architecture:
    Reuses the FeatureExtraction backbone from the main TRIM model,
    adapted for 4 input channels (A, T, C, G).
    """

    def __init__(
        self,
        # Backbone Hyperparameters
        FE_filters: int = 64,
        FE_kernel_size: int = 5,
        FE_conv_stride: int = 1,
        FE_conv_padding: int = 0,
        FE_ln_epsilon: float = 0.007,
        FE_dropout_conv: float = 0.3,
        FE_residual: bool = False,
        FE_num_conv_layers: int = 4,
        FE_activation: str = "relu",
        # Training Hyperparameters
        lr: float = 1e-3,
        use_huber: bool = True,
        huber_beta: float = 1.0,
        use_r2_in_loss: bool = True,
        r2_lambda: float = 0.1,
        target_col: str = "TE_value",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.target_col = target_col

        # 1) Feature Extraction Backbone: Identical structure, but in_channels=4
        self.backbone_local = FeatureExtraction(
            in_channels=4,
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

        # 2) Regression Head: Maps (N, FE_filters) to scalar output
        self.reg_head = nn.Sequential(
            nn.Linear(FE_filters, FE_filters),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(FE_filters, 1),
        )

        self.lr = lr
        self.use_huber = use_huber
        self.huber_beta = huber_beta
        self.use_r2_in_loss = use_r2_in_loss
        self.r2_lambda = float(r2_lambda)

    def forward(self, x200: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x200: (N, 4, 200)
        Returns:
            yhat: (N,)
        """
        h = self.backbone_local(x200)  # -> (N, FE_filters, L')
        v = h.mean(dim=-1)  # Global Average Pooling -> (N, FE_filters)
        yhat = self.reg_head(v).squeeze(-1)  # -> (N,)
        return yhat

    def _step(self, batch, stage: str):
        x200 = batch["x200"].float()
        yt = batch[self.target_col].float()

        yhat = self(x200)

        # Filter NaN / Inf
        mask = torch.isfinite(yt)

        if mask.any():
            if self.use_huber:
                loss_reg = F.smooth_l1_loss(yhat[mask], yt[mask], beta=self.huber_beta)
            else:
                loss_reg = F.mse_loss(yhat[mask], yt[mask])
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        # Metrics calculation
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

        # Combined Loss: (1 - λ) * L_reg + λ * (1 - R²)
        if self.use_r2_in_loss and mask.any():
            loss = (1.0 - self.r2_lambda) * loss_reg + self.r2_lambda * (1.0 - r2)
        else:
            loss = loss_reg

        self.log_dict(
            {
                f"{stage}/loss_reg": loss,
                f"{stage}/mse_reg": mse,
                f"{stage}/r2_reg": r2,
                f"{stage}/pearson_reg": pearson,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
