from typing import List, Dict, Optional, Tuple
import json, re, os, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.utils.helpers import TargetZScaler
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ===== Defining the structure of environment variables =====
# Note: Keys correspond to column names in the input data (Chinese) and must be preserved.
DEFAULT_SCHEMA = {
    # Strain
    "菌株": "cat",
    # Cultivation
    "预培养.培养基": "cat",
    "预培养.碳源": "cat",
    "预培养.碳源浓度_%": "num",
    "预培养.氮源": "cat",
    "预培养.氮源浓度_%": "num",
    "预培养.时间_h": "num",
    "预培养.温度_C": "num",
    "预培养.RNA终点": "num",
    "预培养.蛋白质终点": "num",
    # Drugged culture
    "加药培养.培养基": "cat",
    "加药培养.碳源": "cat",
    "加药培养.碳源浓度_%": "num",
    "加药培养.氮源": "cat",
    "加药培养.氮源浓度_%": "num",
    "加药培养.PH": "num",
    "加药培养.温度_C": "num",
    "加药培养.时间_h": "num",
    "加药培养.终点": "cat",
    # Drug A/B/C
    "药物.A": "cat",
    "药物.A_μM": "num",
    "药物.A_μg_mL": "num",
    "药物.B": "cat",
    "药物.B_μM": "num",
    "药物.B_μg_mL": "num",
    "药物.C": "cat",
    "药物.C_μM": "num",
    "药物.C_μg_mL": "num",
    # Genetic Modification
    "遗传改造.预留1": "cat",
    "遗传改造.预留2": "cat",
    "遗传改造.预留3": "cat",
    "遗传改造.预留4": "cat",
    "遗传改造.预留5": "num",
    # Treatment
    "处理": "cat",
    "处理时间_h": "num",
}

# ===== Codon Bias Table =====
CODON2SCORE = {
    "TTT": 0.59,
    "TTC": 0.41,
    "TTA": 0.28,
    "TTG": 0.29,
    "CTT": 0.13,
    "CTC": 0.06,
    "CTA": 0.14,
    "CTG": 0.11,
    "ATT": 0.46,
    "ATC": 0.26,
    "ATA": 0.27,
    "ATG": 1.00,
    "GTT": 0.39,
    "GTC": 0.21,
    "GTA": 0.21,
    "GTG": 0.19,
    "TCT": 0.26,
    "TCC": 0.16,
    "TCA": 0.21,
    "TCG": 0.10,
    "AGT": 0.16,
    "AGC": 0.11,
    "CCT": 0.31,
    "CCC": 0.15,
    "CCA": 0.42,
    "CCG": 0.12,
    "ACT": 0.35,
    "ACC": 0.22,
    "ACA": 0.30,
    "ACG": 0.14,
    "GCT": 0.38,
    "GCC": 0.22,
    "GCA": 0.29,
    "GCG": 0.11,
    "TAT": 0.56,
    "TAC": 0.44,
    "TAA": 0.47,
    "TAG": 0.23,
    "TGA": 0.30,
    "CAT": 0.64,
    "CAC": 0.36,
    "CAA": 0.69,
    "CAG": 0.31,
    "AAT": 0.59,
    "AAC": 0.41,
    "AAA": 0.58,
    "AAG": 0.42,
    "GAT": 0.65,
    "GAC": 0.35,
    "GAA": 0.70,
    "GAG": 0.30,
    "TGT": 0.63,
    "TGC": 0.37,
    "TGG": 1.00,
    "CGT": 0.14,
    "CGC": 0.06,
    "CGA": 0.07,
    "CGG": 0.04,
    "AGA": 0.48,
    "AGG": 0.21,
    "GGT": 0.47,
    "GGC": 0.19,
    "GGA": 0.22,
    "GGG": 0.12,
}


# ===== ENV Preprocessing =====
def _to_hours(x):
    """Convert formats like 30m/0.5h/2h to hours (float)."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower().replace(" ", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)([hm])$", s)
    if m:
        num, unit = m.groups()
        num = float(num)
        return num if unit == "h" else num / 60.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _to_percent_number(x):
    """
    Keep Numbers;
    '2%' -> 2.0;
    None->0.
    Caution: The converted number refers to the percentage number.
    e.g.: 2 means 2%
    """
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _conc_to_two_scales(x):
    """
    Concentration keeps two units:
      - By μM
      - By μg/mL
    return (val_uM, val_ug_ml)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return 0.0, 0.0
    if isinstance(x, (int, float)):
        return 0.0, 0.0
    s = str(x).strip().replace(" ", "")

    m = re.match(r"^([0-9]*\.?[0-9]+)(μg/mL|ug/mL|ng/mL|mg/mL)$", s, flags=re.I)
    if m:
        num, unit = m.groups()
        num = float(num)
        unit = unit.lower()
        if unit in ("μg/ml", "ug/ml"):
            return 0.0, num
        if unit == "ng/ml":
            return 0.0, num / 1000.0
        if unit == "mg/ml":
            return 0.0, num * 1000.0

    m2 = re.match(r"^([0-9]*\.?[0-9]+)([μumn]?M)$", s, flags=re.I)
    if m2:
        num, unit = m2.groups()
        num = float(num)
        unit = unit.lower()
        if unit == "nm":  # nM -> μM
            return num / 1000.0, 0.0
        if unit in ("μm", "um"):  # μM
            return num, 0.0
        if unit == "mm":  # mM -> μM
            return num * 1000.0, 0.0
        if unit == "m":  # M  -> μM
            return num * 1e6, 0.0
    return 0.0, 0.0


def _normalize_env_record(env: dict) -> dict:
    """
    Flatten nested ENV JSON into a single-level dictionary:
      - Resolve placeholders like "same as pre-culture xxx"
      - Parse %, time, temperature, pH, drug concentrations
      - Generate stable column names for subsequent type inference
    """
    env = env or {}
    pre = dict(env.get("预培养", {}) or {})
    drug = dict(env.get("加药培养", {}) or {})
    meds = dict(env.get("药物", {}) or {})
    edit = dict(env.get("遗传改造", {}) or {})

    # Replace "Same as Pre-culture" with actual values
    mapping = {
        "培养基": "培养基",
        "碳源": "碳源",
        "碳源浓度": "碳源浓度",
        "氮源": "氮源",
        "氮源浓度": "氮源浓度",
        "温度": "温度",
        "时间": "时间",
    }

    def _same_as_pre(val, key):
        if isinstance(val, str) and val.startswith("同预培养"):
            return pre.get(mapping.get(key, key))
        return val

    for k in list(drug.keys()):
        drug[k] = _same_as_pre(drug[k], k)

    # Standardize values
    out = {
        "菌株": env.get("菌株", "NA"),
        # Pre-culture
        "预培养.培养基": pre.get("培养基", "NA"),
        "预培养.碳源": pre.get("碳源", "NA"),
        "预培养.碳源浓度_%": _to_percent_number(pre.get("碳源浓度")),
        "预培养.氮源": pre.get("氮源", "NA"),
        "预培养.氮源浓度_%": _to_percent_number(pre.get("氮源浓度")),
        "预培养.时间_h": _to_hours(pre.get("时间")),
        "预培养.温度_C": float(pre.get("温度") or 0),
        "预培养.RNA终点": float(pre.get("RNA终点") or 0),
        "预培养.蛋白质终点": float(pre.get("蛋白质终点") or 0),
        # Drugged culture
        "加药培养.培养基": drug.get("培养基", "NA"),
        "加药培养.碳源": drug.get("碳源", "NA"),
        "加药培养.碳源浓度_%": _to_percent_number(drug.get("碳源浓度")),
        "加药培养.氮源": drug.get("氮源", "NA"),
        "加药培养.氮源浓度_%": _to_percent_number(drug.get("氮源浓度")),
        "加药培养.PH": float(drug.get("PH") or 0),
        "加药培养.温度_C": float(drug.get("温度") or 0),
        "加药培养.时间_h": _to_hours(drug.get("时间")),
        "加药培养.终点": drug.get("终点", "NA"),
        # Treatment
        "处理": env.get("处理", "NA"),
        "处理时间_h": _to_hours(env.get("处理时间")),
    }

    # Drug A/B/C: Split into two numerical columns (*_uM and *_ug_ml); keep name for classification
    for tag in ["A", "B", "C"]:
        name = meds.get(f"药物{tag}")
        conc = meds.get(f"浓度{tag}")
        uM, ug_ml = _conc_to_two_scales(conc)
        out[f"药物.{tag}"] = name if name is not None else "NA"
        out[f"药物.{tag}_μM"] = uM
        out[f"药物.{tag}_μg_mL"] = ug_ml

    # Genetic Modification
    for i in range(1, 6):
        key = f"预留{i}"
        val = edit.get(key)
        # Reserved5 is numeric 0 in example; others might be string placeholders
        if i == 5:
            try:
                out[f"遗传改造.{key}"] = float(val or 0.0)
            except Exception:
                out[f"遗传改造.{key}"] = 0.0
        else:
            out[f"遗传改造.{key}"] = val if val is not None else "NA"

    return out


# Numeric post-processing compatible with old pipeline (retained for backward compatibility)
def preprocess_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Compatibility for old column names
    if "预培养时间" in data.columns:
        data["预培养时间"] = data["预培养时间"].apply(_to_hours)
    if "加药培养时间" in data.columns:
        data["加药培养时间"] = data["加药培养时间"].apply(_to_hours)
    if "浓度" in data.columns:
        # Old single-column concentration (if it exists)
        def _old(x):
            uM, ug_ml = _conc_to_two_scales(x)
            return ug_ml if ug_ml > 0 else uM

        data["浓度"] = data["浓度"].apply(_old)

    # New names: Suffix pattern recognition (even if already numeric)
    for col in list(data.columns):
        if (
            col.endswith(".时间_h")
            or col.endswith(".温度_C")
            or col.endswith(".PH")
            or col.endswith("_μM")
            or col.endswith("_μg_mL")
            or col.endswith("处理时间_h")
            # Added: Two numeric endpoints
            or col.endswith("RNA终点")
            or col.endswith("蛋白质终点")
        ):
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
        if col.endswith("浓度_%"):
            data[col] = data[col].apply(_to_percent_number)

    return data


class EnvPreprocessor:
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.ct = ColumnTransformer(
            [
                ("num", StandardScaler(), self.numeric_cols),
                (
                    "cat",
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    self.categorical_cols,
                ),
            ]
        )

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        for c in self.numeric_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
        for c in self.categorical_cols:
            if c in df.columns:
                df[c] = df[c].fillna("NA")
        df = preprocess_numeric_data(df)
        self.ct.fit(
            df
        )  # Fit once so the model knows how to encode categorical variables
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for c in self.numeric_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
            else:
                df[c] = 0.0
        for c in self.categorical_cols:
            if c in df.columns:
                df[c] = df[c].fillna("NA")
            else:
                df[c] = "NA"
        df = preprocess_numeric_data(df)
        return self.ct.transform(df)

    def save(self, path: str):
        joblib.dump(
            {
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
                "ct": self.ct,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "EnvPreprocessor":
        obj = joblib.load(path)
        ep = EnvPreprocessor(obj["numeric_cols"], obj["categorical_cols"])
        ep.ct = obj["ct"]
        return ep


# ====== 2. Sequence One-hot ======
NUC2IDX = {"A": 0, "T": 1, "C": 2, "G": 3}


# ====== 3. Build TSV ======
def load_seq_table(
    txt_path: str,
    key_col: str = "tx_id",  # Key column for alignment
    utr5_col: str = "utr5_sequence",
    cds_col: str = "cds_sequence",
    utr3_col: str = "utr3_sequence",
    sep: str = "\t",
) -> Dict[str, Dict[str, str]]:
    """Read three sequence segments from TXT/TSV/CSV and assemble into a mapping table."""
    df_seq = pd.read_csv(txt_path, sep=sep)
    for c in [key_col, utr5_col, cds_col, utr3_col]:
        if c not in df_seq.columns:
            raise ValueError(f"Sequence table missing column: {c}")
    mapping: Dict[str, Dict[str, str]] = {}
    for _, r in df_seq.iterrows():
        k = str(r[key_col])
        mapping[k] = {
            "utr5_sequence": str(r.get(utr5_col, "") or ""),
            "cds_sequence": str(r.get(cds_col, "") or ""),
            "utr3_sequence": str(r.get(utr3_col, "") or ""),
        }
    return mapping


def build_dataset(
    df_long: pd.DataFrame,
    seq_table_path: str,  # Path to sequence file (TXT/TSV)
    out_csv: str,
    env_numeric_cols: List[str],
    env_categorical_cols: List[str],
    env_preproc_out: str,
    tx_id_col: Optional[str] = "tx_id",
    gene_col: str = "SystematicName",
    sample_col: str = "sample",
    TE_value_col: str = "TE_value",
    df_key_col: str = "tx_id",  # Align with sequence table using tx_id
    seq_sep: str = "\t",
    utr5_col: str = "utr5_sequence",
    cds_col: str = "cds_sequence",
    utr3_col: str = "utr3_sequence",
) -> str:
    """
    - Requires tx_id in df_long.
    - Supports TXT/TSV for sequence file (aligned by tx_id).
    - Fits ENV preprocessor once and saves it.
    """
    df = df_long.copy()

    # 1) Check if tx_id exists
    if not tx_id_col or tx_id_col not in df.columns:
        raise ValueError(
            f"df_long must contain tx_id column (current tx_id_col={tx_id_col})."
        )
    df["tx_id"] = df[tx_id_col].astype(str)

    # 2) Read sequence mapping
    gene2seq = load_seq_table(
        seq_table_path,
        key_col=df_key_col,
        utr5_col=utr5_col,
        cds_col=cds_col,
        utr3_col=utr3_col,
        sep=seq_sep,
    )

    # 3) Match sequences row by row (prioritize alignment by tx_id)
    utr5, cds, utr3 = [], [], []
    for _, r in df.iterrows():
        key = str(r[df_key_col]) if df_key_col in df.columns else str(r[gene_col])
        s = gene2seq.get(
            key, {"utr5_sequence": "", "cds_sequence": "", "utr3_sequence": ""}
        )
        utr5.append(s["utr5_sequence"])
        cds.append(s["cds_sequence"])
        utr3.append(s["utr3_sequence"])
    df["utr5_sequence"] = utr5
    df["cds_sequence"] = cds
    df["utr3_sequence"] = utr3

    # 4) Concatenate full transcript + lengths
    df["tx_sequence"] = (
        df["utr5_sequence"].fillna("")
        + df["cds_sequence"].fillna("")
        + df["utr3_sequence"].fillna("")
    )
    df["utr5_size"] = df["utr5_sequence"].str.len().fillna(0).astype(int)
    df["cds_size"] = df["cds_sequence"].str.len().fillna(0).astype(int)
    df["utr3_size"] = df["utr3_sequence"].str.len().fillna(0).astype(int)
    df["tx_size"] = df["tx_sequence"].str.len().fillna(0).astype(int)

    # 5) TE metrics
    df["TE_value"] = df[TE_value_col].astype(float).fillna(0.0)

    # 6) Package ENV + Fit & Save
    env_cols = list(env_numeric_cols) + list(env_categorical_cols)

    def row_env_json(row):
        d = {c: (row[c] if c in row.index else None) for c in env_cols}
        return json.dumps(d, ensure_ascii=False)

    df["ENV_JSON"] = df.apply(row_env_json, axis=1)

    # Use reindex to ensure missing columns are created (filled later in preprocessor)
    env_df = df.reindex(columns=env_cols)
    ep = EnvPreprocessor(env_numeric_cols, env_categorical_cols).fit(env_df)
    ep.save(env_preproc_out)

    # 7) Write out CSV
    keep_cols = [
        "tx_id",
        "utr5_sequence",
        "cds_sequence",
        "utr3_sequence",
        "utr5_size",
        "cds_size",
        "utr3_size",
        "tx_sequence",
        "tx_size",
        "TE_value",
        "ENV_JSON",
        gene_col,
        sample_col,
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(out_csv, sep=",", index=False)
    return out_csv


# ====== 4. Build PyTorch Dataset/DataModule ======
class TRIMDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        env_preproc_path: str,
        # === Key parameters for alignment with TRIM ===
        max_utr5: int = 1381,
        max_cds_utr3: int = 11937,
        targets: Tuple[str, ...] = ("TE_value",),
        standardize_target: bool = False,
        target_scaler_path: Optional[str] = None,
        # UTR secondary structure related
        utr_struct_jsonl: Optional[str] = None,
        utr_key_col: str = "SystematicName",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path, sep=",")
        self.CODON2SCORE = dict(CODON2SCORE)

        # —— If SystematicName is missing, generate it automatically from tx_id and write back ——
        utr_key = getattr(self, "_utr_key_col", "SystematicName")
        need_save_back = False

        if utr_key not in self.df.columns:
            if "tx_id" in self.df.columns:
                # Extraction rule: Take the part before the first "_"
                self.df[utr_key] = (
                    self.df["tx_id"].astype(str).str.strip().str.split("_", n=1).str[0]
                )
                need_save_back = True
            else:
                self.df[utr_key] = ""

        # Fill NaNs with empty strings for the generated column
        self.df[utr_key] = self.df[utr_key].fillna("")

        # === Automatically write back to TSV file to avoid re-extraction next time ===
        if need_save_back:
            try:
                tmp_path = Path(csv_path)
                bak_path = tmp_path.with_suffix(".bak.csv")
                tmp_path.rename(bak_path)  # Backup original file
                self.df.to_csv(csv_path, sep=",", index=False)
                print(
                    f"[INFO] Automatically added column '{utr_key}' to {csv_path}. Original file backed up to {bak_path.name}"
                )
            except Exception as e:
                print(f"[WARN] Failed to write back file: {e}")

        # === Build/Load ENV Preprocessor ===
        # 1) Restore ENV_JSON to DataFrame (if exists)
        env_json_series = self.df.get("ENV", pd.Series(["{}"] * len(self.df))).fillna(
            "{}"
        )

        env_rows = []
        for s in env_json_series.tolist():
            rec = json.loads(s) if isinstance(s, str) else (s or {})
            env_rows.append(_normalize_env_record(rec))

        self.env_df = pd.DataFrame(env_rows)

        # —— Dynamically infer numeric/categorical columns:
        # Rule: Columns with specific suffixes are numeric; others are numeric if convertible, else categorical.
        num_like_suffix = (
            ".时间_h",
            ".温度_C",
            ".PH",
            "_μM",
            "_μg_mL",
            "浓度_%",
            "处理时间_h",
        )
        numeric_cols, categorical_cols = [], []
        for c in self.env_df.columns:
            if any(c.endswith(suf) for suf in num_like_suffix):
                numeric_cols.append(c)
            else:
                # Try converting to numeric
                try:
                    pd.to_numeric(self.env_df[c])
                    numeric_cols.append(c)
                except Exception:
                    categorical_cols.append(c)

        # Unified numeric cleaning (idempotent)
        self.env_df = preprocess_numeric_data(self.env_df)

        # === Build/Load Preprocessor ===
        # If preprocessor file does not exist, automatically fit one based on column types and save it.
        if (env_preproc_path is None) or (not os.path.exists(env_preproc_path)):
            print("[INFO] Current ENV Preprocessor doesn't exist. Creating a NEW ONE.")
            # Based on DEFAULT_SCHEMA, pick "known" numeric/categorical columns, then infer others
            known_num = [
                c
                for c, t in DEFAULT_SCHEMA.items()
                if t == "num" and c in self.env_df.columns
            ]
            known_cat = [
                c
                for c, t in DEFAULT_SCHEMA.items()
                if t == "cat" and c in self.env_df.columns
            ]

            other_cols = [
                c for c in self.env_df.columns if c not in known_num + known_cat
            ]

            if other_cols is not None:
                print(
                    ">>>> Found undefined numeric/categorical columns, program will classify automatically <<<<"
                )
                print(f"{other_cols}")

            # Simple inference: columns convertible to numeric are numeric, else categorical
            inferred_num, inferred_cat = [], []
            for c in other_cols:
                try:
                    pd.to_numeric(self.env_df[c])
                    inferred_num.append(c)
                except Exception:
                    inferred_cat.append(c)

            num_cols = known_num + inferred_num
            cat_cols = known_cat + inferred_cat

            self.ep = EnvPreprocessor(num_cols, cat_cols).fit(self.env_df)

            # Ensure directory exists before saving
            if env_preproc_path is not None:
                os.makedirs(os.path.dirname(env_preproc_path) or ".", exist_ok=True)
                self.ep.save(env_preproc_path)
                print(f"[INFO] New ENV Preprocessor saved to Path: {env_preproc_path}")
        else:
            # Load if exists
            self.ep = EnvPreprocessor.load(env_preproc_path)

        # Standardize Case / Replace U->T, and create length columns
        for c in (
            "utr5_sequence",
            "cds_sequence",
            "utr3_sequence",
        ):
            if c in self.df.columns:
                self.df[c] = (
                    self.df[c].astype(str).str.strip().str.upper().str.replace("U", "T")
                )
            else:
                self.df[c] = ""
        self.df["tx_sequence"] = (
            self.df["utr5_sequence"]
            + self.df["cds_sequence"]
            + self.df["utr3_sequence"]
        )
        self.df["utr5_size"] = self.df["utr5_sequence"].str.len().fillna(0).astype(int)
        self.df["cds_size"] = self.df["cds_sequence"].str.len().fillna(0).astype(int)
        self.df["utr3_size"] = self.df["utr3_sequence"].str.len().fillna(0).astype(int)
        self.df["tx_size"] = self.df["tx_sequence"].str.len().fillna(0).astype(int)

        # === Align input values with dictionary keys ===
        self.max_utr5 = int(max_utr5)
        self.max_cds_utr3 = int(max_cds_utr3)

        # 5'UTR is always right-aligned (end of UTR5 aligns with end of max_utr5)
        self.padded_len = self.max_utr5 + self.max_cds_utr3
        # Filter out-of-bound samples: require utr5 <= max_utr5; cds + utr3 <= max_cds_utr3
        self.df = self.df[
            (self.df["utr5_size"] <= self.max_utr5)
            & ((self.df["tx_size"] - self.df["utr5_size"]) <= self.max_cds_utr3)
        ].reset_index(drop=True)

        # 12 channels or 4 channels
        self.seq_channels = 12

        self.targets = [t for t in targets if t in self.df.columns]

        self.standardize_target = bool(standardize_target)
        self._target_scaler = None
        if (
            self.standardize_target
            and target_scaler_path
            and os.path.exists(target_scaler_path)
        ):
            try:
                self._target_scaler = TargetZScaler.load(target_scaler_path)
            except Exception:
                print("[INFO] Unable to Process Target Scaling.")
                self._target_scaler = (
                    None  # Fallback to "no standardization" on failure
                )

        # UTR Secondary structure related keys
        self._utr_key_col = utr_key_col
        self.u5_pp_dict = {}
        self.u3_pp_dict = {}
        if utr_struct_jsonl is not None and os.path.exists(utr_struct_jsonl):
            with open(utr_struct_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    gid = str(rec.get("SystematicName") or rec.get("gene") or "")
                    if not gid:
                        continue
                    p5 = (rec.get("five_utr", {}) or {}).get("p_paired", None)
                    p3 = (rec.get("three_utr", {}) or {}).get("p_paired", None)
                    if isinstance(p5, list):
                        self.u5_pp_dict[gid] = np.asarray(p5, dtype=np.float32)
                    if isinstance(p3, list):
                        self.u3_pp_dict[gid] = np.asarray(p3, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _encode_into(self, x: np.ndarray, seq: str, ch_offset: int, pos_offset: int):
        # x: (C, L_pad), encode seq into 4 channels (ATCG) starting from ch_offset, writing from pos_offset
        for i, ch in enumerate(seq):
            j = NUC2IDX.get(ch)
            if j is not None:
                x[ch_offset + j, pos_offset + i] = 1.0

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        u5, cd, u3 = r.utr5_sequence, r.cds_sequence, r.utr3_sequence
        L5, Lc, L3 = int(r.utr5_size), int(r.cds_size), int(r.utr3_size)

        # === Build (C, L_pad) ===
        x = np.zeros((self.seq_channels, self.padded_len), dtype=np.uint8)

        pos_u5 = off5 = self.max_utr5 - L5
        pos_cds = self.max_utr5
        pos_u3 = self.max_utr5 + Lc

        # 5'UTR → Channels 0..3
        self._encode_into(x, u5, ch_offset=0, pos_offset=pos_u5)
        # CDS  → Channels 4..7, starting from max_utr5
        self._encode_into(x, cd, ch_offset=4, pos_offset=pos_cds)
        # 3'UTR → Channels 8..11, following CDS
        self._encode_into(x, u3, ch_offset=8, pos_offset=pos_u3)

        # Add three channels: u5_pp, cds_aux (bias/triplet), u3_pp
        ch_u5_pp = np.zeros((1, self.padded_len), dtype=np.float32)
        ch_cds_aux = np.zeros((1, self.padded_len), dtype=np.float32)
        ch_u3_pp = np.zeros((1, self.padded_len), dtype=np.float32)

        # 5'UTR p_paired written in right-aligned interval
        gid = str(r.get(self._utr_key_col, ""))
        if gid in self.u5_pp_dict:
            arr = self.u5_pp_dict[gid]
            n = min(L5, int(arr.shape[0]))
            if n > 0:
                ch_u5_pp[0, pos_u5 : pos_u5 + n] = arr[:n]

        # CDS triplet auxiliary (broadcast by preference value)
        for i in range(0, Lc, 3):
            codon = cd[i : i + 3].upper()
            s = self.CODON2SCORE.get(codon, 0.0)
            ch_cds_aux[0, pos_cds + i : pos_cds + i + 3] = s

        # 3'UTR p_paired
        if gid in self.u3_pp_dict:
            arr = self.u3_pp_dict[gid]
            n = min(L3, int(arr.shape[0]))
            if n > 0:
                ch_u3_pp[0, pos_u3 : pos_u3 + n] = arr[:n]

        # Assemble into 15 channels (12 one-hot + 3 new channels)
        x_full = np.concatenate(
            [x.astype(np.float32), ch_u5_pp, ch_cds_aux, ch_u3_pp], axis=0
        )

        # ENV vector
        env_vec = self.ep.transform(self.env_df.iloc[[idx]])[0].astype(np.float32)

        item = {
            "tx_id": r.get("tx_id", f"row_{idx}"),
            "x": torch.from_numpy(x_full),
            "env": torch.from_numpy(env_vec),
            "lens": torch.tensor([L5, Lc, L3], dtype=torch.long),
        }

        y_raw = float(r["TE_value"])

        # If configured and scaler loaded successfully, z-score transform y
        if self.standardize_target and (self._target_scaler is not None):
            y_use = float(self._target_scaler.transform(y_raw))
        else:
            y_use = y_raw

        item["TE_value"] = torch.tensor(y_use, dtype=torch.float32)

        # Assemble item
        for t in self.targets:
            if t == "TE_value":
                # y_use might be z-scored (if standardize_target is on)
                item[t] = torch.tensor(y_use, dtype=torch.float32)
            else:
                # Keep generic branch for future expansion
                v = float(r[t])
                item[t] = torch.tensor(v, dtype=torch.float32)

        item["TE_value_raw"] = torch.tensor(y_raw, dtype=torch.float32)
        return item


class TRIMDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        all_csv: Optional[
            str
        ] = None,  # If only one csv provided, split internally into train/val/test
        env_preproc_path: Optional[str] = None,  # Path to saved env preprocessor
        target_scaler_path: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 16,
        max_utr5: int = 1381,
        max_cds_utr3: int = 11937,
        targets: Tuple[str, ...] = ("TE_value",),
        utr_struct_jsonl: Optional[str] = None,
        utr_key_col: str = "SystematicName",
    ):
        super().__init__()
        self.train_csv, self.val_csv, self.test_csv = train_csv, val_csv, test_csv
        self.all_csv, self.env_preproc_path = all_csv, env_preproc_path
        self.bs, self.nw = batch_size, num_workers
        self.target_scaler_path = target_scaler_path

        self.utr_struct_jsonl = utr_struct_jsonl
        self.utr_key_col = utr_key_col

        self.kw = dict(
            max_utr5=max_utr5,
            max_cds_utr3=max_cds_utr3,
            targets=targets,
            utr_struct_jsonl=self.utr_struct_jsonl,
            utr_key_col=self.utr_key_col,
        )

    def setup(self, stage: Optional[str] = None):
        # --- Priority Protection: If 3 splits provided, ignore all_csv ---
        has_3splits = (
            (self.train_csv is not None)
            and (self.val_csv is not None)
            and (self.test_csv is not None)
        )

        if (self.all_csv is not None) and (not has_3splits):
            # Use all_csv for splitting/dumping only when 3 splits are NOT provided
            df = pd.read_csv(self.all_csv, sep=",")

            # If no split column: auto split 8:1:1; if present, respect existing split
            if "split" not in df.columns:
                # Split 80% train + 20% temp
                df_train, df_tmp = train_test_split(
                    df, test_size=0.2, random_state=42, shuffle=True
                )
                # Split temp into val/test (10% each)
                df_val, df_test = train_test_split(
                    df_tmp, test_size=0.5, random_state=42, shuffle=True
                )

                df_train = df_train.copy()
                df_train["split"] = "train"
                df_val = df_val.copy()
                df_val["split"] = "val"
                df_test = df_test.copy()
                df_test["split"] = "test"
                df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

                base = os.path.splitext(self.all_csv)[0]
                df.to_csv(f"{base}.withsplit.csv", sep=",", index=False)

            # Dump three files regardless of split source
            base = os.path.splitext(self.all_csv)[0]
            paths = {}
            for sp in ["train", "val", "test"]:
                sub = df[df["split"] == sp]
                if sub.empty:
                    raise ValueError(
                        f"[all_csv Split] split='{sp}' is empty. Check data or split column."
                    )
                p = f"{base}.{sp}.csv"
                sub.to_csv(p, sep=",", index=False)
                paths[sp] = p

            self.train_csv, self.val_csv, self.test_csv = (
                paths["train"],
                paths["val"],
                paths["test"],
            )

        elif has_3splits:
            # Three splits provided: just check existence
            for p, name in [
                (self.train_csv, "train_csv"),
                (self.val_csv, "val_csv"),
                (self.test_csv, "test_csv"),
            ]:
                if (p is None) or (not os.path.exists(p)):
                    raise FileNotFoundError(f"{name} file does not exist: {p}")
        else:
            raise ValueError(
                "Please provide (train_csv, val_csv, test_csv) or provide all_csv for auto-split."
            )

        # === General part independent of split method ===
        base_dir = (
            os.path.dirname(self.env_preproc_path) if self.env_preproc_path else "."
        )
        target_scaler_path = self.target_scaler_path

        # If not exists, fit z-score (mean, var) using training set
        if not os.path.exists(target_scaler_path):
            tmp_df = pd.read_csv(self.train_csv)
            if "TE_value" not in tmp_df.columns:
                raise KeyError(
                    f"Training set missing TE_value column: {self.train_csv}"
                )
            scaler = TargetZScaler().fit(tmp_df["TE_value"].values)
            os.makedirs(base_dir or ".", exist_ok=True)
            scaler.save(target_scaler_path)

        self.target_scaler_path = target_scaler_path

        # Build three Datasets and enable standardize_target
        self.ds_train = TRIMDataset(
            self.train_csv,
            self.env_preproc_path,
            **self.kw,
            standardize_target=True,
            target_scaler_path=target_scaler_path,
        )
        self.ds_val = TRIMDataset(
            self.val_csv,
            self.env_preproc_path,
            **self.kw,
            standardize_target=True,
            target_scaler_path=target_scaler_path,
        )
        self.ds_test = TRIMDataset(
            self.test_csv,
            self.env_preproc_path,
            **self.kw,
            standardize_target=True,
            target_scaler_path=target_scaler_path,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
        )
