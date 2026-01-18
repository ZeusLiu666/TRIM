"""
Inference script for the 5'UTR Small Expert Model (Expert_5U).

Input:
    A CSV file containing at least the following columns:
    - utr5_sequence
    - cds_sequence

Output:
    A new CSV file based on the input, with an additional column:
    - pred_TE_5U (Predicted Translation Efficiency)
"""

import os
from typing import List

import numpy as np
import pandas as pd
import torch

from src.data import NUC2IDX  # Map A/T/C/G to channel indices
from src.model_5U import Expert_5U  # The 5U Expert Model class


# ====================== 1. Configuration ======================


def get_cfg():
    """
    Central configuration for file paths and model hyperparameters.
    """
    return {
        # Input CSV path
        "in_csv_path": "data/RPL8A_double_mutations_input.csv",
        # Path to the trained Expert_5U checkpoint
        # "ckpt_path": "outputs_5U/logs/version_1/checkpoints/local200-epoch=184-val/r2_reg=0.737.ckpt",
        "ckpt_path": "outputs_5U/logs/version_0/checkpoints/local200-epoch=160-val/r2_reg=0.785.ckpt",
        # Output CSV path (will contain the prediction column)
        "out_csv_path": "outputs_5U/predict/RPL8A_double_mutations_input.with_pred.csv",
        # Encoding parameters (Must match Dataset_5U configuration)
        "L": 200,
        "utr_tail_len": 150,
        "cds_head_len": 50,
        # Inference batch size
        "batch_size": 2048,
    }


# ====================== 2. Encoding Logic ======================


def encode_utr_cds_to_x200(
    utr_seq: str,
    cds_seq: str,
    L: int,
    utr_tail_len: int,
    cds_head_len: int,
) -> np.ndarray:
    """
    Encodes `utr5_sequence` + `cds_sequence` into a (4, L) one-hot numpy array.

    Logic:
      1) Extract the last `utr_tail_len` nucleotides of the 5'UTR.
      2) Extract the first `cds_head_len` nucleotides of the CDS.
      3) Concatenate: merged = utr_tail + cds_head.
      4) Right-align the merged sequence to length L and apply one-hot encoding.
    """
    # Defensive processing: Upper case and standardize U -> T
    utr = (utr_seq or "").upper().replace("U", "T")
    cds = (cds_seq or "").upper().replace("U", "T")

    # Extract UTR Tail
    if len(utr) >= utr_tail_len:
        utr_tail = utr[-utr_tail_len:]
    else:
        utr_tail = utr

    # Extract CDS Head
    cds_head = cds[:cds_head_len]

    # Concatenate
    merged = utr_tail + cds_head

    # Truncate from left if merged sequence exceeds L
    merged = merged[-L:]

    # Initialize zero array: shape = (4, L)
    x = np.zeros((4, L), dtype=np.float32)

    # Calculate start position for right alignment
    start_pos = L - len(merged)

    # One-hot encoding based on NUC2IDX
    for i, base in enumerate(merged):
        pos = start_pos + i
        idx = NUC2IDX.get(base, None)  # NUC2IDX = {"A":0, "T":1, "C":2, "G":3}
        if idx is not None and 0 <= pos < L:
            x[idx, pos] = 1.0

    return x


# ====================== 3. Model Loading ======================


def load_expert5u(ckpt_path: str, device: torch.device) -> Expert_5U:
    """
    Load the Expert_5U model from a PyTorch Lightning checkpoint.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = Expert_5U.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


# ====================== 4. Main Inference Pipeline ======================


def run_inference_on_csv():
    cfg = get_cfg()
    os.makedirs(os.path.dirname(cfg["out_csv_path"]), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load input CSV
    df = pd.read_csv(cfg["in_csv_path"])
    print(f"[INFO] Total rows: {len(df)}")

    # Validate columns
    for col in ["utr5_sequence", "cds_sequence"]:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: {col}")

    # 2) Load Model
    model = load_expert5u(cfg["ckpt_path"], device)

    L = cfg["L"]
    utr_tail_len = cfg["utr_tail_len"]
    cds_head_len = cfg["cds_head_len"]
    batch_size = cfg["batch_size"]

    preds: List[float] = [0.0] * len(df)

    # Manual batching loop
    num_rows = len(df)
    num_batches = (num_rows + batch_size - 1) // batch_size

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_rows)
        sub_df = df.iloc[start:end]

        # 3) Batch Encoding -> (B, 4, L)
        xs = []
        for _, row in sub_df.iterrows():
            utr = str(row["utr5_sequence"])
            cds = str(row["cds_sequence"])
            x = encode_utr_cds_to_x200(
                utr,
                cds,
                L=L,
                utr_tail_len=utr_tail_len,
                cds_head_len=cds_head_len,
            )
            xs.append(x)

        x_batch = np.stack(xs, axis=0)  # (B, 4, L)
        x_batch_t = torch.from_numpy(x_batch).to(device=device, dtype=torch.float32)

        # 4) Inference (No Gradient)
        with torch.no_grad():
            y_pred = model(x_batch_t)  # Output shape: (B,) or (B, 1)
            y_pred = y_pred.detach().cpu().numpy()

        # Flatten output
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]

        # Store predictions
        for i, val in enumerate(y_pred):
            preds[start + i] = float(val)

        print(f"[INFO] Batch {b+1}/{num_batches} completed. Rows {start} to {end-1}")

    # 5) Save results
    df["pred_TE_5U"] = preds
    df.to_csv(cfg["out_csv_path"], index=False)
    print(f"[INFO] Inference complete. Saved to: {cfg['out_csv_path']}")


# ====================== 5. Entry Point ======================

if __name__ == "__main__":
    run_inference_on_csv()
