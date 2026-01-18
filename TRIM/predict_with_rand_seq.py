"""
Use a trained optimal checkpoint (ckpt) to infer on a CSV and write the predicted TE_value back to the file.

Input:  rand_seq_input.csv  (Columns: tx_id, utr5_sequence, cds_sequence, utr3_sequence, ENV, TE_value, Is_Translated)
Output: rand_seq_input.pred.csv  (Adds te_pred column, or overwrites/fills TE_value)
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from src.data import TRIMDataModule
from src.regression_model import Regression


def try_inverse_transform(pred: np.ndarray, target_scaler_path: str) -> np.ndarray:
    if (not target_scaler_path) or (not os.path.exists(target_scaler_path)):
        return pred
    try:
        obj = joblib.load(target_scaler_path)
        if hasattr(obj, "inverse_transform"):
            return np.asarray(
                obj.inverse_transform(pred.reshape(-1, 1)).reshape(-1), dtype=float
            )
        if isinstance(obj, dict) and ("mean" in obj and "std" in obj):
            return pred * float(obj["std"]) + float(obj["mean"])
    except Exception as e:
        print(f"[WARN] Inverse transform failed, returning z-scores. Reason: {e}")
    return pred


# ========= Preserves original functionality =========
def predict_csv_with_ckpt(
    CKPT_PATH: str,
    CSV_PATH: str,
    ENV_PREPROC_PATH: str,
    TARGET_SCALER_PATH: str,
    OUT_PATH: str | None = None,
    BATCH_SIZE: int = 512,
    NUM_WORKERS: int = 16,
    MAX_UTR5: int = 1381,
    MAX_CDS_UTR3: int = 11937,
) -> str:
    """
    Execute inference and write results based on current script logic.
    Returns the path to the output file.
    """
    pl.seed_everything(42)

    assert os.path.isfile(CKPT_PATH), f"ckpt does not exist: {CKPT_PATH}"
    assert os.path.isfile(ENV_PREPROC_PATH), "Missing training ENV preprocessor."
    assert os.path.isfile(TARGET_SCALER_PATH), "Missing training target z-scaler."

    if OUT_PATH is None:
        OUT_PATH = CSV_PATH.replace(".csv", "_pred_out.csv")

    # DataModule
    dm = TRIMDataModule(
        train_csv=CSV_PATH,
        val_csv=CSV_PATH,
        test_csv=CSV_PATH,
        all_csv=None,
        env_preproc_path=ENV_PREPROC_PATH,
        target_scaler_path=TARGET_SCALER_PATH,
        utr_struct_jsonl="data/preprocessor/OCH1_utr_struct.jsonl",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_utr5=MAX_UTR5,
        max_cds_utr3=MAX_CDS_UTR3,
        targets=("Is_Translated", "TE_binary", "TE_value"),
    )
    dm.setup(stage="test")

    # Simple character histogram info
    def char_hist(s):
        s = str(s)
        return {c: s.count(c) for c in sorted(set(s))}

    row0 = dm.ds_test.df.iloc[0]

    # Take two samples for x comparison (Keep original check output)
    item1 = dm.ds_test[0]
    item2 = dm.ds_test[1] if len(dm.ds_test) > 1 else dm.ds_test[0]

    x1, x2 = item1["x"].numpy(), item2["x"].numpy()

    def extract_cds(x, lens, max_utr5):
        L5, Lc, _ = lens.numpy().tolist()
        ch = slice(4, 8)  # CDS one-hot channels
        start = max_utr5
        end = start + Lc
        return x[ch, start:end]

    cds1 = extract_cds(x1, item1["lens"], dm.ds_test.max_utr5)
    cds2 = extract_cds(x2, item2["lens"], dm.ds_test.max_utr5)

    # Load model and predict
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model: Regression = Regression.load_from_checkpoint(
        CKPT_PATH, map_location=map_location, strict=True
    )
    model.eval()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    preds = trainer.predict(model=model, datamodule=dm)

    te_list = []
    for item in preds:
        y = item.get("te")
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().reshape(-1)
        te_list.append(np.asarray(y))
    te_pred = np.concatenate(te_list, axis=0)

    te_real = try_inverse_transform(te_pred, TARGET_SCALER_PATH)

    # Write output: Overwrite TE_value, keep te_pred (z-score or raw)
    df = pd.read_csv(CSV_PATH, sep=",")
    df["TE_value"] = te_real
    df["te_pred"] = te_pred
    df.to_csv(OUT_PATH, sep=",", index=False)

    print(f"[OK] Inference results written to: {OUT_PATH}")
    print(f"    Sample count: {len(df)}; Prediction shape: {te_real.shape}")
    return OUT_PATH


if __name__ == "__main__":
    CKPT_PATH = (
        "outputs/logs/version_0/checkpoints/best-epoch=022-val/r2_reg=0.795.ckpt"
    )
    CSV_PATH = "data/filtered_dataset/20_Gene_Predict.csv"
    ENV_PREPROC_PATH = "data/preprocessor/env_preproc.joblib"
    TARGET_SCALER_PATH = "data/preprocessor/target_TE_value_zscaler.pkl"
    BATCH_SIZE = 512
    NUM_WORKERS = 16
    OUT_PATH = CSV_PATH.replace(".csv", "_pred_out.csv")

    predict_csv_with_ckpt(
        CKPT_PATH=CKPT_PATH,
        CSV_PATH=CSV_PATH,
        ENV_PREPROC_PATH=ENV_PREPROC_PATH,
        TARGET_SCALER_PATH=TARGET_SCALER_PATH,
        OUT_PATH=OUT_PATH,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=NUM_WORKERS,
        MAX_UTR5=1381,
        MAX_CDS_UTR3=11937,
    )
