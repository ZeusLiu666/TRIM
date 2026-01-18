import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import NUC2IDX


class Dataset_5U(Dataset):
    """
    Dataset for the 200nt 'Small Expert' model targeting the Translation Initiation Site (TIS) context.

    Logic:
    For each sample, it extracts:
        - The last 150nt of the 5' UTR.
        - The first 50nt of the CDS.
    These are concatenated into a sequence (<= 200nt) and encoded as a
    right-aligned one-hot tensor of shape (4, 200).
    """

    def __init__(
        self,
        csv_path: str,
        utr5_col: str = "utr5_sequence",
        cds_col: str = "cds_sequence",
        target_col: str = "TE_value",
    ):
        super().__init__()
        self.csv_path = csv_path
        self.utr5_col = utr5_col
        self.cds_col = cds_col
        self.target_col = target_col

        df = pd.read_csv(csv_path, sep=",")

        # Standardize sequence format: Upper case, replace U with T
        for c in [utr5_col, cds_col]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.upper().str.replace("U", "T")
            else:
                df[c] = ""

        if target_col not in df.columns:
            raise KeyError(f"{csv_path} is missing target column: {target_col}")

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _encode_4xL(self, seq: str, L: int = 200) -> np.ndarray:
        """
        Encodes sequence into a (4, L) one-hot tensor.
        Right-aligned: The sequence fills from the right, padded with zeros on the left.
        """
        x = np.zeros((4, L), dtype=np.float32)

        n = len(seq)
        use = min(n, L)  # Actual length to encode
        start_seq = n - use  # Start index in the source sequence
        start_pos = L - use  # Start index in the target tensor (padding on left)

        for i in range(use):
            ch = seq[start_seq + i]
            j = NUC2IDX.get(ch, None)
            if j is not None:
                x[j, start_pos + i] = 1.0
        return x

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]

        u5 = str(r.get(self.utr5_col, "") or "")
        cds = str(r.get(self.cds_col, "") or "")

        # 1) Extract the last 150nt of the 5' UTR (Context upstream of start codon)
        u5_tail = u5[-150:] if len(u5) >= 150 else u5

        # 2) Extract the first 50nt of the CDS (Context downstream of start codon)
        cds_head = cds[:50]

        # Merge (Max length <= 200nt)
        merged = u5_tail + cds_head

        x = self._encode_4xL(merged, L=200)  # Shape: (4, 200)
        y = float(r[self.target_col])

        item = {
            "x200": torch.from_numpy(x),
            self.target_col: torch.tensor(y, dtype=torch.float32),
        }
        return item
