import warnings

warnings.filterwarnings(
    "ignore", message=r".*pkg_resources is deprecated.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"lightning_utilities\.core\.imports"
)
warnings.filterwarnings("ignore", message=r".*Lazy modules are a new feature.*")
warnings.filterwarnings("ignore", message=r".*UninitializedParameter.*")

import os
import shutil
import joblib
import json
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from types import SimpleNamespace as _NS
import multiprocessing as mp

from src.data import TRIMDataModule
from src.regression_model import Regression

# ================= Configuration =================
CV_CONFIG = _NS(
    # Data paths
    all_csv="data/filtered_dataset/filtered_dataset.txt",
    output_dir="outputs/outputs_kfold",
    k_folds=5,
    random_seed=42,
    # Training parameters
    batch_size=768,
    max_epochs=50,
    num_workers=16,
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
    # Regression Head
    REG_dropout=0.3,
    REG_use_huber=True,
    REG_huber_beta=1.5,
    REG_use_r2_in_loss=True,
    REG_r2_lambda=0.1,
    # Learning Rate
    REG_lr=1e-3,
    REG_filters=64,
    # LR Scheduler
    REG_warmup_epochs=5,
    REG_eta_min=1e-6,
    # RegHead
    REG_fusion_hidden=512,
    REG_RegHead_heads=4,
    REG_RegHead_depth=4,
    REG_mlp_ratio=2,
    # Data parameters
    max_utr5=1381,
    max_cds_utr3=11937,
)


def run_cross_validation():
    # --- 0. Multiprocessing Safety ---
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # --- Setup ---
    pl.seed_everything(CV_CONFIG.random_seed, workers=True)
    # Enable CuDNN benchmark
    torch.backends.cudnn.benchmark = True

    os.makedirs(CV_CONFIG.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(CV_CONFIG.output_dir, "cv_config.json"), "w") as f:
        cfg_dict = {k: v for k, v in vars(CV_CONFIG).items() if not k.startswith("__")}
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

    print(f"[CV] Loading full dataset from {CV_CONFIG.all_csv}")
    full_df = pd.read_csv(CV_CONFIG.all_csv, sep=",")

    # Initialize K-Fold
    kf = KFold(
        n_splits=CV_CONFIG.k_folds, shuffle=True, random_state=CV_CONFIG.random_seed
    )

    # Store metrics for each fold
    summary_metrics = []

    # --- 2. Iterate Through Folds ---
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_df)):
        fold_num = fold_idx + 1
        print(f"\n{'='*20} Processing Fold {fold_num}/{CV_CONFIG.k_folds} {'='*20}")

        # A. Path Preparation
        fold_dir = os.path.join(CV_CONFIG.output_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        train_csv_path = os.path.join(fold_dir, "train.csv")
        val_csv_path = os.path.join(fold_dir, "val.csv")
        env_preproc_path = os.path.join(fold_dir, "env_preproc.joblib")
        target_scaler_path = os.path.join(fold_dir, "target_scaler.pkl")

        # B. Split and Save Data (Ensure fold independence)
        train_df = full_df.iloc[train_indices]
        val_df = full_df.iloc[val_indices]
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        # C. Clean old preprocessors (Prevent data leakage: must refit on current fold)
        if os.path.exists(env_preproc_path):
            os.remove(env_preproc_path)
        if os.path.exists(target_scaler_path):
            os.remove(target_scaler_path)

        # D. Initialize DataModule
        dm = TRIMDataModule(
            train_csv=train_csv_path,
            val_csv=val_csv_path,
            test_csv=val_csv_path,  # Use val set as placeholder for test
            env_preproc_path=env_preproc_path,
            target_scaler_path=target_scaler_path,
            batch_size=CV_CONFIG.batch_size,
            num_workers=CV_CONFIG.num_workers,
            max_utr5=CV_CONFIG.max_utr5,
            max_cds_utr3=CV_CONFIG.max_cds_utr3,
        )
        dm.setup("fit")

        # Get ENV dimension
        _ = dm.ds_train
        env_input_dim = dm.ds_train.ep.ct.transform(dm.ds_train.env_df.iloc[[0]]).shape[
            1
        ]
        print(f"[Fold {fold_num}] ENV Input Dimension: {env_input_dim}")

        # E. Initialize Model (Using dictionary unpacking for consistency)
        # ---------------------------------------------------------
        FeatureExtraction_args = dict(
            FE_in_channels=CV_CONFIG.FE_in_channels,
            FE_filters=CV_CONFIG.FE_filters,
            FE_kernel_size=CV_CONFIG.FE_kernel_size,
            FE_conv_stride=CV_CONFIG.FE_conv_stride,
            FE_conv_padding=CV_CONFIG.FE_conv_padding,
            FE_ln_epsilon=CV_CONFIG.FE_ln_epsilon,
            FE_dropout_conv=CV_CONFIG.FE_dropout,
            FE_residual=CV_CONFIG.FE_residual,
            FE_num_conv_layers=CV_CONFIG.FE_num_conv_layers,
            FE_activation=CV_CONFIG.FE_activation,
        )

        ENVEncoder_args = dict(
            ENV_input_size=env_input_dim,  # Dynamic
            ENV_width=CV_CONFIG.ENV_width,
            ENV_depth=CV_CONFIG.ENV_depth,
            ENV_out_dim=CV_CONFIG.ENV_out_dim,
            ENV_dropout=CV_CONFIG.ENV_dropout,
        )

        REGHead_args = dict(
            REG_dropout=CV_CONFIG.REG_dropout,
            REG_use_huber=CV_CONFIG.REG_use_huber,
            REG_huber_beta=CV_CONFIG.REG_huber_beta,
            REG_use_r2_in_loss=CV_CONFIG.REG_use_r2_in_loss,
            REG_r2_lambda=CV_CONFIG.REG_r2_lambda,
            REG_lr=CV_CONFIG.REG_lr,
            REG_filters=CV_CONFIG.REG_filters,
            REG_warmup_epochs=CV_CONFIG.REG_warmup_epochs,
            REG_eta_min=CV_CONFIG.REG_eta_min,
            REG_fusion_hidden=CV_CONFIG.REG_fusion_hidden,
            REG_RegHead_heads=CV_CONFIG.REG_RegHead_heads,
            REG_RegHead_depth=CV_CONFIG.REG_RegHead_depth,
            REG_mlp_ratio=CV_CONFIG.REG_mlp_ratio,
        )

        model = Regression(
            **FeatureExtraction_args,
            **ENVEncoder_args,
            **REGHead_args,
        )
        # ---------------------------------------------------------

        # F. Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val/r2_reg",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val/r2_reg:.3f}",
        )
        early_stop_callback = EarlyStopping(
            monitor="val/r2_reg", mode="max", patience=15
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # G. Trainer
        logger = CSVLogger(save_dir=fold_dir, name="logs")

        trainer = pl.Trainer(
            max_epochs=CV_CONFIG.max_epochs,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            enable_progress_bar=True,
            log_every_n_steps=10,
            precision=32,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

        # H. Start Training
        trainer.fit(model, datamodule=dm)

        # I. Validate and Save Results
        print(f"[CV] Validating best model for Fold {fold_num}...")
        # Reload best weights
        val_results = trainer.validate(model, datamodule=dm, ckpt_path="best")[0]

        # Record metrics
        summary_metrics.append(
            {
                "fold": fold_num,
                "R2": val_results.get("val/r2_reg", 0.0),
                "Pearson": val_results.get("val/pearson_reg", 0.0),
                "MSE": val_results.get("val/mse_reg", 0.0),
            }
        )

        # --- J. Denormalize and Save Predictions ---
        if hasattr(model, "_val_cache") and model._val_cache:
            cache = model._val_cache
            y_z = cache["y"]  # True values (Z-score)
            yhat_z = cache["yhat"]  # Predicted values (Z-score)

            # Load Scaler for denormalization
            scaler_data = joblib.load(target_scaler_path)
            # Compatibility logic
            if isinstance(scaler_data, dict):
                mu = float(scaler_data["mean"])
                std = float(scaler_data["std"])
            else:
                mu = scaler_data.mean
                std = scaler_data.std

            y_real = y_z * std + mu
            yhat_real = yhat_z * std + mu

            # Save as CSV
            df_res = pd.DataFrame(
                {"y_true": y_real, "y_pred": yhat_real, "fold": fold_num}
            )
            res_save_path = os.path.join(fold_dir, "val_predictions.csv")
            df_res.to_csv(res_save_path, index=False)
            print(f"[CV] Saved predictions to {res_save_path}")
        else:
            print("[Warning] No validation cache found, skipping prediction save.")

        # Cleanup
        del model, trainer, dm
        torch.cuda.empty_cache()

    # --- 3. Summary Output ---
    print("\n" + "=" * 40)
    print("Cross Validation Complete")
    print("=" * 40)
    df_summary = pd.DataFrame(summary_metrics)
    print(df_summary)
    print("-" * 40)
    print(f"Mean R2: {df_summary['R2'].mean():.4f} ± {df_summary['R2'].std():.4f}")

    # Save summary
    df_summary.to_csv(os.path.join(CV_CONFIG.output_dir, "cv_summary.csv"), index=False)


if __name__ == "__main__":
    run_cross_validation()
