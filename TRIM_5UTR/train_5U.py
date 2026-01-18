import warnings
warnings.filterwarnings(
    "ignore", message=r".*pkg_resources is deprecated.*", category=UserWarning
)
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from src.dataset_5U import Dataset_5U
from src.model_5U import Expert_5U

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

def plot_loss_curves_from_csv(log_dir: str, out_dir: str | None = None):
    """
    Extract train/val loss from metrics.csv of Lightning in each epoch.
    Plot Learning Curve。
    """
    metrics_csv = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"[Plot-5U] metrics.csv not found: {metrics_csv}")
        return
    df = pd.read_csv(metrics_csv)

    # —— Ensuring sving directoty ——
    save_dir = out_dir or os.path.join(log_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    def _last(df_, col):
        d = df_[df_[col].notna()][["epoch", col]]
        return d.groupby("epoch", as_index=False).last() if not d.empty else d

    # Total loss figure
    d_train = _last(df, "train/loss_reg")
    d_val = _last(df, "val/loss_reg")
    else:
        plt.figure(figsize=(7, 4), dpi=150)
        if not d_train.empty:
            plt.plot(
                d_train["epoch"],
                d_train["train/loss_reg"],
                "o-",
                label="train/loss_reg",
                ms=3,
            )
        if not d_val.empty:
            plt.plot(
                d_val["epoch"],
                d_val["val/loss_reg"],
                "o-",
                label="val/loss_reg",
                ms=3,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss_REG")
        plt.title("Learning Curves (5U Expert Regression)")
        plt.legend()
        plt.tight_layout()
        out1 = os.path.join(save_dir, "loss_regression.png")
        plt.savefig(out1)
        plt.close()
        print(f"[Plot-5U] Saved {out1}")
        

class ValScatterPlot5U(pl.Callback):
    """
    Plot True vs Pred Scatter every epoch
    """

    def __init__(self, every_n_epochs: int = 5, out_dir: str = "figures_5U"):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs <= 0 or (epoch % self.every_n_epochs != 0):
            return

        val_loaders = trainer.val_dataloaders
        val_loader = val_loaders[0]

        device = pl_module.device

        y_true_list = []
        y_pred_list = []

        for batch in val_loader:
            x200 = batch["x200"].to(device).float()
            y = batch["TE_value"].to(device).float()
            mask = torch.isfinite(y)

            if not mask.any():
                continue

            yhat = pl_module(x200)

            y_true_list.append(y[mask].detach().cpu().numpy().ravel())
            y_pred_list.append(yhat[mask].detach().cpu().numpy().ravel())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        N = y_true.size

        # MSE: 1/N ∑(y_i - ŷ_i)^2
        mse = float(np.mean((y_true - y_pred) ** 2))

        # R²: 1 - SSE / SST
        sse = float(np.sum((y_pred - y_true) ** 2))
        sst = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-8
        r2 = float(1.0 - sse / sst)

        # Pearson r:
        # r = cov(y, ŷ) / (σ_y * σ_ŷ)
        if N >= 2:
            vy = y_true - y_true.mean()
            vp = y_pred - y_pred.mean()
            num = float(np.sum(vy * vp))
            den = float(np.sqrt(np.sum(vy ** 2) * np.sum(vp ** 2)) + 1e-8)
            pearson = num / den
        else:
            pearson = 0.0

        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.scatter(y_true, y_pred, s=8, alpha=0.6)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

        if N >= 2:
            a, b = np.polyfit(y_true, y_pred, 1)
            xs = np.linspace(lo, hi, 100)
            ax.plot(xs, a * xs + b, linewidth=1)

        ax.set_xlabel("True TE")
        ax.set_ylabel("Predicted TE")
        ax.set_title(f"5U Expert Validation (epoch {epoch})")

        txt = f"R² = {r2:.3f}\nPearson r = {pearson:.3f}\nMSE = {mse:.3f}\nN = {N}"
        ax.text(0.02, 0.98, txt, ha="left", va="top", transform=ax.transAxes)

        out_path = os.path.join(self.out_dir, f"val_5U_epoch_{epoch:03d}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[Plot-5U] Saved {out_path}")


def main():
    pl.seed_everything(42, workers=True)

    train_csv = "../TRIM/data/filtered_dataset/filtered_dataset.train.csv"
    val_csv = "../TRIM/data/filtered_dataset/filtered_dataset.val.csv"

    train_ds = Dataset_5U(train_csv)
    val_ds = Dataset_5U(val_csv)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=16)

    model = Expert_5U(
        FE_filters=64,
        FE_kernel_size=5,
        FE_conv_stride=1,
        FE_conv_padding=0,
        FE_ln_epsilon=0.007,
        FE_dropout_conv=0.3,
        FE_residual=False,
        FE_num_conv_layers=4,
        FE_activation="relu",
        lr=1e-3,
        use_huber=True,
        huber_beta=1.0,
        use_r2_in_loss=True,
        r2_lambda=0.1,
        target_col="TE_value",
    )

    out_dir = "outputs_5U"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    logger = CSVLogger(save_dir=out_dir, name="logs")

    # Checkpoint + LR Monitor
    ckpt_cb = ModelCheckpoint(
        monitor="val/r2_reg",
        mode="max",
        save_top_k=1,
        filename="local200-{epoch:03d}-{val/r2_reg:.3f}",
    )

    lr_cb = LearningRateMonitor(logging_interval="epoch")
    val_scatter_cb = ValScatterPlot5U(
        every_n_epochs=5,
        out_dir=os.path.join(out_dir, "figures_5U"),
    )

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices="auto",
        callbacks=[ckpt_cb, lr_cb, val_scatter_cb],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Plot loss curve
    plot_loss_curves_from_csv(
        log_dir=logger.log_dir,
        out_dir=os.path.join(out_dir, "figures_5U"),
    )


if __name__ == "__main__":
    main()
