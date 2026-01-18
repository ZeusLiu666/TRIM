import warnings

# Mute setuptools/pkg_resources deprecation warnings
warnings.filterwarnings(
    "ignore", message=r".*pkg_resources is deprecated.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"lightning_utilities\.core\.imports"
)
# Mute PyTorch Lazy module warnings
warnings.filterwarnings("ignore", message=r".*Lazy modules are a new feature.*")
# Mute Model Summary initialization warnings
warnings.filterwarnings("ignore", message=r".*UninitializedParameter.*")

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace as _NS
from src.data import TRIMDataModule
from src.regression_model import Regression
from src.utils.helpers import load_config
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import seaborn as sns
from pytorch_lightning.callbacks import LearningRateMonitor


# ========== Script Built-in Configuration ==========
CFG = _NS(
    # Paths
    train_csv="data/filtered_dataset/filtered_dataset.train.csv",
    val_csv="data/filtered_dataset/filtered_dataset.val.csv",
    test_csv="data/filtered_dataset/filtered_dataset.test.csv",
    all_csv="data/filtered_dataset/filtered_dataset.txt",
    env_preproc="data/preprocessor/env_preproc.joblib",
    target_scaler_path="data/preprocessor/target_TE_value_zscaler.pkl",
    utr_struct_jsonl="data/preprocessor/Gene_utr_struct.jsonl",
    out_dir="outputs",
    # Load previous best results for training
    load_best_ckpt=False,
    best_ckpt="outputs/logs/Head-Only/checkpoints/best-epoch=012-val/r2_reg=0.179.ckpt",
    # Data
    batch_size=768,
    num_workers=16,
    max_utr5=1381,
    max_cds_utr3=11937,
    # Training
    max_epochs=100,
    seed=42,
    precision=32,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=10,
    plot_every=3,  # Run validation plotting every 3 epochs
    # Early Stopping
    early_stop=False,
    early_patience=20,
    # ===== Model Backbone Hyperparameters =====
    # FeatureExtraction
    FE_in_channels=5,
    FE_filters=64,
    FE_kernel_size=5,
    FE_conv_stride=1,
    FE_conv_padding=0,
    FE_ln_epsilon=0.007,
    FE_dropout=0.3,
    FE_residual=False,
    FE_num_conv_layers=10,
    FE_activation="relu",
    # EnvEncoder
    ENV_width=256,
    ENV_depth=1,
    ENV_out_dim=64,
    ENV_dropout=0.3,
    # Regression
    REG_dropout=0.3,
    # Whether to use Huber loss
    REG_use_huber=True,
    REG_huber_beta=1.5,
    # Whether to include R2 in loss
    REG_use_r2_in_loss=True,
    REG_r2_lambda=0.1,  # Weight of R2 in loss
    # Learning Rate
    REG_lr=1e-3,
    REG_filters=64,
    # Learning Rate Scheduler
    REG_warmup_epochs=5,
    REG_eta_min=1e-6,  # Min LR for Cosine Annealing
    # RegHead Parameters
    REG_fusion_hidden=512,
    REG_RegHead_heads=4,
    REG_RegHead_depth=4,
    REG_mlp_ratio=2,
)

# Monitor metric
MONITOR_KEY = "val/r2_reg"
MODE = "max"


# Set random seed for reproducibility
def set_seed(seed: int = 42):
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    # Enable CuDNN benchmark for performance
    torch.backends.cudnn.benchmark = True


def plot_loss_curves_from_csv(log_dir: str, out_dir: str | None = None):
    metrics_csv = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"[Plot-REG] metrics.csv does not exist: {metrics_csv}")
        return
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns:
        print("[Plot-REG] metrics.csv missing 'epoch' column, cannot aggregate.")
        return

    # Safely determine save directory
    save_dir = out_dir or os.path.join(log_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    # Take the last record for each epoch
    def _last(df_, col):
        d = df_[df_[col].notna()][["epoch", col]]
        return d.groupby("epoch", as_index=False).last() if not d.empty else d

    # 1) Loss Curve
    d_train = _last(df, "train/loss_reg")
    d_val = _last(df, "val/loss_reg")
    if d_train.empty and d_val.empty:
        print("[Plot] Cannot find train/loss_reg or val/loss_reg.")
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
                d_val["epoch"], d_val["val/loss_reg"], "o-", label="val/loss_reg", ms=3
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss_REG")
        plt.title("Learning Curves (Regression)")
        plt.legend()
        plt.tight_layout()
        out1 = os.path.join(save_dir, "loss_regression.png")
        plt.savefig(out1)
        plt.close()
        print(f"[Plot] Saved {out1}")


# Custom Callback to evaluate and plot every N epochs
class EvalEveryNEpochsPlot(pl.Callback):
    """
    Every N epochs, predict on validation set using current weights and plot scatter plot.
    """

    def __init__(self, n=3, out_dir="figures_reg"):
        super().__init__()
        self.n = n
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if self.n <= 0 or epoch % self.n != 0:
            return
        cache = getattr(pl_module, "_val_cache", None)
        if not cache or cache.get("N", 0) == 0:
            print("[Plot] Fast path skipped: no cached val outputs.")
            return

        os.makedirs(self.out_dir, exist_ok=True)

        # 1) Retrieve scaler (from datamodule)
        import joblib

        scaler = joblib.load(trainer.datamodule.target_scaler_path)
        mu, std = float(scaler["mean"]), float(scaler["std"])

        # 2) Inverse transform
        y_true = cache["y"] * std + mu
        y_pred = cache["yhat"] * std + mu

        r2 = cache["r2_reg"]
        r = cache["pearson_reg"]
        mse = cache["mse_reg"]
        N = cache["N"]

        # 4) Plot
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.scatter(y_true, y_pred, s=8, alpha=0.6)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)  # Diagonal line

        if N >= 2:
            a, b = np.polyfit(y_true, y_pred, 1)
            xs = np.linspace(lo, hi, 100)
            ax.plot(xs, a * xs + b, linewidth=1)

        ax.set_xlabel("True TE")
        ax.set_ylabel("Predicted TE")
        ax.set_title(f"Validation @ best ckpt (epoch {epoch})")

        txt = f"R² = {r2:.3f}\nPearson r = {r:.3f}\nMSE = {mse:.3f}\nN = {N}"
        ax.text(0.02, 0.98, txt, ha="left", va="top", transform=ax.transAxes)

        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, f"val_reg_epoch_{epoch}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"\n[Plot-REG] Saved {out_path}")

        import gc

        gc.collect()


def main():
    import multiprocessing as mp

    try:
        # Set start method to spawn for safety
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 1) Directories and Logger
    os.makedirs(CFG.out_dir, exist_ok=True)
    os.makedirs(os.path.join(CFG.out_dir, "logs"), exist_ok=True)
    logger = CSVLogger(save_dir=CFG.out_dir, name="logs")
    with open(os.path.join(CFG.out_dir, "config.json"), "w") as f:
        json.dump(vars(CFG), f, indent=2, ensure_ascii=False)

    # 2) Set seed
    set_seed(CFG.seed)

    # 3) Initialize DataModule
    dm = TRIMDataModule(
        train_csv=CFG.train_csv,
        val_csv=CFG.val_csv,
        test_csv=CFG.test_csv,
        all_csv=CFG.all_csv,
        env_preproc_path=CFG.env_preproc,
        target_scaler_path=CFG.target_scaler_path,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        max_utr5=CFG.max_utr5,
        max_cds_utr3=CFG.max_cds_utr3,
        targets=("TE_value",),
        utr_struct_jsonl=CFG.utr_struct_jsonl,
    )

    if CFG.load_best_ckpt is False:
        print(">>>>> Training from very beginning. <<<<<")

    dm.setup("fit")

    # Get encoded ENV input size
    ENV_input_size = dm.ds_train.ep.ct.transform(dm.ds_train.env_df.iloc[[0]]).shape[1]

    # 4) Model
    FeatureExtraction_args = dict(
        FE_in_channels=CFG.FE_in_channels,
        FE_filters=CFG.FE_filters,
        FE_kernel_size=CFG.FE_kernel_size,
        FE_conv_stride=CFG.FE_conv_stride,
        FE_conv_padding=CFG.FE_conv_padding,
        FE_ln_epsilon=CFG.FE_ln_epsilon,
        FE_dropout_conv=CFG.FE_dropout,
        FE_residual=CFG.FE_residual,
        FE_num_conv_layers=CFG.FE_num_conv_layers,
        FE_activation=CFG.FE_activation,
    )

    ENVEncoder_args = dict(
        ENV_input_size=ENV_input_size,
        ENV_width=CFG.ENV_width,
        ENV_depth=CFG.ENV_depth,
        ENV_out_dim=CFG.ENV_out_dim,
        ENV_dropout=CFG.ENV_dropout,
    )

    REGHead_args = dict(
        REG_dropout=CFG.REG_dropout,
        REG_use_huber=CFG.REG_use_huber,
        REG_huber_beta=CFG.REG_huber_beta,
        REG_use_r2_in_loss=CFG.REG_use_r2_in_loss,
        REG_r2_lambda=CFG.REG_r2_lambda,
        REG_lr=CFG.REG_lr,
        REG_filters=CFG.REG_filters,
        REG_warmup_epochs=CFG.REG_warmup_epochs,
        REG_eta_min=CFG.REG_eta_min,
        REG_fusion_hidden=CFG.REG_fusion_hidden,
        REG_RegHead_heads=CFG.REG_RegHead_heads,
        REG_RegHead_depth=CFG.REG_RegHead_depth,
        REG_mlp_ratio=CFG.REG_mlp_ratio,
    )

    model = Regression(
        **FeatureExtraction_args,
        **ENVEncoder_args,
        **REGHead_args,
    )

    # —— Load best weights as new starting point (without restoring optimizer state) ——
    if (CFG.load_best_ckpt) and (CFG.best_ckpt is not None):
        assert os.path.isfile(
            CFG.best_ckpt
        ), f"best_ckpt does not exist: {CFG.best_ckpt}"
        state = torch.load(CFG.best_ckpt, map_location="cpu")
        incompat = model.load_state_dict(state["state_dict"], strict=False)

        if getattr(incompat, "missing_keys", None):
            print("[Resume-B] missing keys:", incompat.missing_keys)
        if getattr(incompat, "unexpected_keys", None):
            print("[Resume-B] unexpected keys:", incompat.unexpected_keys)
        print(f">>>>> loaded weights from best ckpt: {CFG.best_ckpt} <<<<<")

    # 5) Callbacks
    ckpt = ModelCheckpoint(
        monitor=MONITOR_KEY,
        mode=MODE,
        save_top_k=1,
        filename=f"best-{{epoch:03d}}-{{{MONITOR_KEY}:.3f}}",
    )
    eval_plot = EvalEveryNEpochsPlot(
        n=CFG.plot_every,
        out_dir=os.path.join(CFG.out_dir, "figures_reg"),
    )

    lrmon = LearningRateMonitor(logging_interval="step")
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    callbacks = [ckpt, eval_plot, lrmon, lr_cb]
    if CFG.early_stop:
        callbacks.append(
            EarlyStopping(monitor=MONITOR_KEY, mode=MODE, patience=CFG.early_patience)
        )

    # 6) Trainer
    trainer = pl.Trainer(
        max_epochs=CFG.max_epochs,
        accelerator=CFG.accelerator,
        devices=CFG.devices,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=CFG.log_every_n_steps,
        enable_progress_bar=True,
        precision=CFG.precision,
        deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        limit_val_batches=0.25,
    )

    # 7) Train + Validate
    trainer.fit(model, datamodule=dm)

    # === Plot loss curves after training ===
    try:
        plot_loss_curves_from_csv(
            log_dir=logger.log_dir,
            out_dir=os.path.join(CFG.out_dir, "figures"),
        )
    except Exception as e:
        print(f"[Plot] Failed to plot loss curves: {e}")

    print("[Eval] Running validation with best checkpoint...")
    trainer.validate(model=None, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
