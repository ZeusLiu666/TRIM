"""
Stochastic Hill Climbing Optimization (Synonymous Mutation) + Batch Inference + Per-Round Aggregation

Dependencies:
- Reuses predict_csv_with_ckpt() from predict_with_rand_seq.py
- Reuses CODON2SCORE (for sampling probabilities) and input column definitions from data.py
- numpy / pandas / torch / pytorch_lightning / joblib

Inputs:
- Paths to trained model checkpoint (ckpt), ENV preprocessor, and target z-scaler
- Initial 5'UTR / CDS / 3'UTR sequences (DNA, T/A/C/G)
"""

import os, json, random, string, math, shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# === Import functions and constants from your project ===
from predict_with_rand_seq import predict_csv_with_ckpt  # noqa
from src.data import (
    CODON2SCORE,
)  # Used for sampling probabilities and auxiliary scoring  # noqa


def _aa_table() -> Dict[str, List[str]]:
    """Standard Genetic Code: Amino Acid -> List of Synonymous Codons (using T instead of U)"""
    table = {
        "A": ["GCT", "GCC", "GCA", "GCG"],
        "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
        "N": ["AAT", "AAC"],
        "D": ["GAT", "GAC"],
        "C": ["TGT", "TGC"],
        "Q": ["CAA", "CAG"],
        "E": ["GAA", "GAG"],
        "G": ["GGT", "GGC", "GGA", "GGG"],
        "H": ["CAT", "CAC"],
        "I": ["ATT", "ATC", "ATA"],
        "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
        "K": ["AAA", "AAG"],
        "M": ["ATG"],  # Start codon
        "F": ["TTT", "TTC"],
        "P": ["CCT", "CCC", "CCA", "CCG"],
        "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
        "T": ["ACT", "ACC", "ACA", "ACG"],
        "W": ["TGG"],
        "Y": ["TAT", "TAC"],
        "V": ["GTT", "GTC", "GTA", "GTG"],
        "*": ["TAA", "TAG", "TGA"],  # Stop codons
    }
    return table


def _trans_table() -> Dict[str, str]:
    """Codon -> Amino Acid (Single Letter)"""
    out = {}
    for aa, codons in _aa_table().items():
        for c in codons:
            out[c] = aa
    return out


CODON2AA = _trans_table()


def cds_to_aa(cds: str) -> str:
    """Translate DNA CDS (ATCG) to amino acid sequence; ignore trailing partial codons."""
    cds = (cds or "").strip().upper().replace("U", "T")
    aas = []
    for i in range(0, len(cds) - len(cds) % 3, 3):
        codon = cds[i : i + 3]
        aa = CODON2AA.get(codon, "X")  # 'X' for unknown
        if aa == "*":
            # Keep the stop position, but generating random sequences will fix the last codon to stop
            aas.append("*")
            break
        aas.append(aa)
    return "".join(aas)


def aa_to_random_cds(aa_seq: str, rng: np.random.Generator) -> str:
    """Generate a random synonymous CDS from an AA sequence, weighted by CODON2SCORE."""
    table = _aa_table()
    out = []
    for i, aa in enumerate(aa_seq):
        choices = table.get(aa)
        if not choices:
            # Fallback for unknown AA
            out.append("NNN")
            continue
        # Start (M) and Stop (*) handling
        if aa == "M":
            out.append("ATG")
            continue
        if aa == "*":
            # Choose a random stop codon
            w = np.array([CODON2SCORE.get(c, 1e-6) for c in choices], dtype=float)
            w = w / w.sum()
            out.append(rng.choice(choices, p=w))
            break
        # General case: sample based on CODON2SCORE weights
        w = np.array([CODON2SCORE.get(c, 1e-6) for c in choices], dtype=float)
        w = w / w.sum()
        out.append(rng.choice(choices, p=w))
    return "".join(out)


def mutate_synonymous(cds: str, k_sites: int, rng: np.random.Generator) -> str:
    """Randomly select k codon sites in the CDS and replace them with 'different synonymous codons'.
    - Keep ATG (start) unchanged.
    - If the end is a stop codon, keep it unchanged.
    """
    cds = cds.strip().upper().replace("U", "T")
    assert len(cds) % 3 == 0, "CDS length must be a multiple of 3"
    n_codon = len(cds) // 3
    codons = [cds[i * 3 : (i + 1) * 3] for i in range(n_codon)]

    # Protect start and stop codons
    protect_head = codons[0] == "ATG"
    protect_tail = codons[-1] in ("TAA", "TAG", "TGA")
    candidate_idx = list(range(n_codon))
    if protect_head:
        candidate_idx = candidate_idx[1:]
    if protect_tail and len(candidate_idx) > 0:
        candidate_idx = candidate_idx[:-1]
    if len(candidate_idx) == 0:
        return cds  # No sites to mutate

    # Randomly pick k distinct sites
    k = min(k_sites, len(candidate_idx))
    pick = rng.choice(candidate_idx, size=k, replace=False)

    for idx in pick:
        old = codons[idx]
        aa = CODON2AA.get(old, None)
        if aa is None or aa == "*" or aa == "M":
            continue
        syn = [c for c in _aa_table()[aa] if c != old]
        if not syn:
            continue
        # Pick a different synonymous codon based on preference weights
        w = np.array([CODON2SCORE.get(c, 1e-6) for c in syn], dtype=float)
        w = w / w.sum()
        codons[idx] = rng.choice(syn, p=w)
    return "".join(codons)


def _mk_tx_id(prefix: str, i: int) -> str:
    return f"{prefix}_{i:07d}"


def _write_candidates_csv(rows: List[Dict], path: str):
    df = pd.DataFrame(rows)
    # Required columns: tx_id, utr5_sequence, cds_sequence, utr3_sequence, ENV, TE_value, Is_Translated
    for c in ["tx_id", "utr5_sequence", "cds_sequence", "utr3_sequence"]:
        assert c in df.columns, f"Missing column: {c}"
    if "ENV" not in df.columns:
        df["ENV"] = "{}"
    if "Is_Translated" not in df.columns:
        df["Is_Translated"] = 1
    if "TE_value" not in df.columns:
        df["TE_value"] = 0.0
    df.to_csv(path, index=False)


# ---------------- Main Function ----------------
def random_hill_climb_optimize(
    ckpt_path: str,
    env_preproc_path: str,
    target_scaler_path: str,
    utr5: str,
    cds_start: str,
    utr3: str,
    *,
    n_starts: int = 64,  # Number of initial synonymous starting points
    top_keep: int = 16,  # Number to keep per round (and write to summary)
    neighbors_per_round: int = 256,  # Number of neighbors (mutants) per "parent"
    mutate_sites: int = 64,  # Number of codons to mutate per neighbor
    steps: int = 128,  # Number of iteration rounds
    batch_size: int = 2048,
    num_workers: int = 16,
    out_dir: str = "outputs/cds_opt",
    rng_seed: int = 2025,
    tx_prefix: str = "YGL103W",
) -> str:
    """
    Returns: Path to the final "summary CSV" (new candidates and TE recorded each round).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    # 1) Get target amino acid sequence from start CDS
    aa_seq = cds_to_aa(cds_start)
    if len(aa_seq) == 0:
        raise ValueError("AA sequence translated from start CDS is empty.")

    # 2) Generate initial synonymous start points (including the start itself)
    starts: List[Tuple[str, float]] = []
    uniq = set()

    uniq.add(cds_start)
    starts.append((cds_start, math.nan))
    for _ in range(n_starts - 1):
        cds_new = aa_to_random_cds(aa_seq, rng)
        if cds_new not in uniq:
            uniq.add(cds_new)
            starts.append((cds_new, math.nan))

    # 3) Write round=0 candidates and infer
    tmp_dir = Path(out_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    round_log = Path(out_dir) / "round_log.csv"  # Top retained list per round

    def _score_and_rank(cds_list: List[str], round_idx: int, tag: str) -> pd.DataFrame:
        rows = []
        env_dict = {
            "菌株": "SUB592",
            "预培养": {
                "培养基": "YPD",
                "碳源": "glucose",
                "碳源浓度": "2%",
                "氮源": None,
                "氮源浓度": "0%",
                "时间": "20h",
                "温度": 30,
                "RNA终点": "1.5",
                "蛋白质终点": "1.5",
            },
            "加药培养": {
                "培养基": "",
                "碳源": "",
                "碳源浓度": "",
                "氮源": "",
                "氮源浓度": "",
                "PH": 7,
                "温度": "",
                "时间": "",
            },
            "药物": {
                "药物A": None,
                "浓度A": 0,
                "药物B": None,
                "浓度B": 0,
                "药物C": None,
                "浓度C": 0,
            },
            "遗传改造": {
                "预留1": None,
                "预留2": None,
                "预留3": None,
                "预留4": None,
                "预留5": 0,
            },
            "处理": None,
            "处理时间": "0h",
        }

        # Convert dict to JSON string (ensure_ascii=False for Chinese characters)
        env_json_str = json.dumps(env_dict, ensure_ascii=False)
        for i, cds in enumerate(cds_list):
            rows.append(
                {
                    "tx_id": _mk_tx_id(f"{tx_prefix}_r{round_idx}_{tag}", i),
                    "utr5_sequence": utr5,
                    "cds_sequence": cds,
                    "utr3_sequence": utr3,
                    "ENV": env_json_str,
                    "Is_Translated": 1,
                    "TE_value": 0.0,
                    "SystematicName": f"{tx_prefix}",
                }
            )
        tsv = tmp_dir / f"round_{tag}.csv"
        _write_candidates_csv(rows, str(tsv))
        out = predict_csv_with_ckpt(
            CKPT_PATH=ckpt_path,
            CSV_PATH=str(tsv),
            ENV_PREPROC_PATH=env_preproc_path,
            TARGET_SCALER_PATH=target_scaler_path,
            OUT_PATH=str(tsv).replace(".csv", "_pred.csv"),
            BATCH_SIZE=batch_size,
            NUM_WORKERS=num_workers,
            MAX_UTR5=1381,
            MAX_CDS_UTR3=11937,
        )
        df = pd.read_csv(out)
        # Note: predict_csv_with_ckpt writes "denormalized" results back to TE_value, keeping te_pred (z-score).
        df["round"] = round_idx
        return (
            df[
                [
                    "round",
                    "tx_id",
                    "utr5_sequence",
                    "cds_sequence",
                    "utr3_sequence",
                    "TE_value",
                    "te_pred",
                    "SystematicName",
                ]
            ]
            .sort_values("TE_value", ascending=False)
            .reset_index(drop=True)
        )

    # round 0
    df0 = _score_and_rank([s for s, _ in starts], round_idx=0, tag="seed")
    df0.head(top_keep).to_csv(round_log, index=False)
    print(f"[INFO] round=0 completed, Top={top_keep} recorded in {round_log}")

    # ★ Select Global Top-K from round_log as next generation parents (unique CDS)
    def _pick_global_elite(round_log_path: Path, k: int) -> List[str]:
        elite = pd.read_csv(round_log_path)
        elite = (
            elite.sort_values("TE_value", ascending=False)
            .drop_duplicates(subset="cds_sequence", keep="first")
            .head(k)
        )
        return elite["cds_sequence"].tolist()

    # 4) Iterative Hill Climbing
    parents = df0.head(top_keep)["cds_sequence"].tolist()
    for r in range(1, steps + 1):
        # 4.1 Generate neighbors for each "parent"
        neigh = []
        for cds in parents:
            for _ in range(neighbors_per_round):
                neigh.append(mutate_synonymous(cds, mutate_sites, rng))
        # Deduplicate (preserve order)
        seen = set()
        neigh = [x for x in neigh if not (x in seen or seen.add(x))]

        # 4.2 Inference and scoring
        dfr = _score_and_rank(neigh, round_idx=r, tag="neigh")
        keep_df = dfr.head(top_keep)
        keep_df.to_csv(round_log, mode="a", header=False, index=False)

        # ★ Update parents using "Global Top-K" after each round, not just current round Top-K
        parents = _pick_global_elite(round_log, top_keep)
        best = float(keep_df["TE_value"].iloc[0])
        print(f"[INFO] round={r} completed, current best TE={best:.4f}")

    print(f"[OK] Random hill climbing finished.")
    print(f"[OK] Per-round Top={top_keep} summary at: {round_log}")
    return


# ---------------- Minimal Example (Can be removed) ----------------
if __name__ == "__main__":
    CKPT = "outputs/logs/version_0/checkpoints/best-epoch=011-val/r2_reg=0.770.ckpt"
    ENV_P = "data/preprocessor/env_preproc.joblib"
    TGT_S = "data/preprocessor/target_TE_value_zscaler.pkl"
    U5 = "ACTATGCTACGTACCTGTTTAACTCTTCTCATTTTATCCGTTTTCTTTCTTCACCGTTCCTCTTAAGTTCTTTATTTTTTTCATAACAGAATGATCACCCCTTTCACTTTGCCGCCAATATAATATTAACACACAAGAAATAAGTATGAGGTAGTTTGCTCTCGAAAAAACCAAGTAATAGTTCAAAAAATGAAAAAAAGAAAACAAATACCAAATTATGCCACCGTTACCTTACGTTTCATGGTTAATCATCGTTTACTGCCGCCTATGAGCGTAAGCTAATGTTATAAAGAAACAAGCTATAATATTGTTAAATATAGTTGATCAACAGCATTGTAATGATTACAAGAGACGAGGTGGAATGAACCTTATGAAATGCGTATTATATATAAACTGTAATAAGAGCTAAGTTGAATTGAAATCTACGATACTTGATGTTGACATTATAGCACTAGTTCCCAGGAAACCCTTTCGAAAAACACAGCAAAAACAAGAGTACTGTAACCAATGTAACATCTGTACACCAGGGACCCACACATTACCAAAATCAAAATTATTTTTCTAATGCCTGTTATTTTTCCTATTTTTCCTCTGGCGCGTGAATAGCCCGCAGAGACGCAAACAATTTTCCTCGCAGTTTTTCGCTTGTTTAATGCGTATTTTCCCAGATAGGTTCAAACCTTTCATCTGTATCCCGTATATTTAAGATGGCGTTTGCTTTCTCCGTTGATTTTTTTCCTTCTTAGTGATTTTTTTGCATTAAATCCCAGAACAATCATCCAACTAATCAAGA"
    CDS = "ATGACCATGATTACCGATAGCCTTGCAGTCGTTTTGCAACGTCGAGACTGGGAAAACCCTGGGGTAACTCAACTTAACCGTCTGGCTGCTCATCCTCCATTCGCCTCCTGGCGGAACTCTGAGGAGGCCAGAACCGACCGTCCTAGCCAGCAGCTCAGGTCGTTGAACGGGGAATGGAGATTTGCATGGTTCCCTGCACCAGAGGCGGTTCCTGAGTCCTGGTTAGAGTGTGATTTGCCTGAGGCAGATACTGTTGTTGTTCCTAGTAACTGGCAAATGCATGGGTACGACGCCCCTATTTACACAAATGTCACCTACCCTATCACGGTTAACCCTCCATTTGTTCCCACAGAAAACCCAACGGGTTGTTATTCATTGACCTTTAACGTTGATGAAAGTTGGTTGCAAGAAGGACAGACAAGAATCATCTTCGATGGAGTTAACTCAGCCTTCCACCTTTGGTGTAACGGCCGTTGGGTCGGATATGGGCAGGATTCTAGACTCCCCTCGGAGTTTGATCTGAGCGCCTTTTTGCGTGCGGGTAAGAATCGACTAGCAGTCATGGTCCTCAGGTGGTCGGACGGATCATATTTGGAAGATCAGGATATGTGGAGGATGAGTGGTATCTTTCGTGATGTTTCTTTACTACATAAGCCAACCACGCAAATTTCGGATTTTCATGTGGCAACTCGTTTTAATGACGATTTCTCGAGAGCCGTTCTCGAGGCCGAGGTTCAAATGTGTGGCGAATTGAGAGACTACTTGCGTGTTACTGTCTCCTTGTGGCAAGGGGAAACGCAAGTTGCTTCCGGCACCGCCCCTTTCGGCGGAGAAATTATTGATGAGCGAGGAGGCTATGCTGATAGGGTAACTCTTAGGTTGAACGTGGAAAATCCAAAGTTGTGGTCCGCTGAAATTCCCAATCTGTACAGAGCGGTAGTAGAGTTGCATACAGCTGATGGTACGTTGATCGAGGCCGAAGCGTGTGATGTCGGCTTCAGAGAGGTCAGGATTGAAAATGGCTTGCTCCTCTTAAATGGCAAACCCCTCTTAATACGAGGTGTAAACAGACATGAGCATCACCCTTTGCATGGGCAGGTTATGGACGAGCAAACTATGGTTCAAGATATCCTGTTAATGAAACAGAACAATTTTAACGCTGTAAGGTGTTCTCACTACCCAAACCATCCCTTGTGGTACACTCTGTGTGATAGATATGGCCTATACGTCGTTGATGAGGCAAATATTGAGACACATGGAATGGTGCCTATGAATAGATTAACTGATGATCCTCGTTGGCTTCCTGCGATGAGTGAGCGAGTCACTAGGATGGTTCAACGTGACAGGAATCATCCTAGTGTGATAATTTGGTCCTTAGGGAACGAATCAGGGCATGGGGCTAATCACGATGCACTATATAGATGGATCAAGTCCGTAGACCCTTCTCGTCCGGTTCAATACGAAGGAGGCGGTGCAGACACAACTGCCACCGACATTATATGTCCCATGTATGCTCGTGTGGATGAAGACCAGCCCTTCCCAGCAGTGCCTAAATGGTCTATAAAAAAATGGTTGTCCTTACCGGGTGAGACTAGACCTTTGATTTTGTGCGAATATGCTCATGCGATGGGCAATTCTCTTGGCGGCTTTGCCAAGTATTGGCAGGCCTTCAGACAATATCCAAGATTACAAGGTGGCTTCGTGTGGGATTGGGTTGATCAATCTTTAATAAAATACGACGAAAACGGTAATCCGTGGTCAGCTTACGGTGGAGACTTCGGTGATACCCCGAATGATAGGCAGTTCTGCATGAACGGCTTAGTATTTGCAGACAGGACTCCGCACCCGGCGTTGACGGAAGCAAAGCATCAGCAGCAATTTTTTCAATTCAGATTATCCGGACAGACTATTGAAGTGACTTCTGAATACTTATTCCGACACTCTGACAACGAGCTATTACACTGGATGGTTGCTTTGGACGGGAAACCTCTGGCTAGTGGTGAAGTTCCTTTAGACGTGGCGCCTCAAGGAAAGCAACTGATCGAATTGCCAGAGCTACCGCAACCTGAGTCTGCTGGCCAGCTATGGCTCACGGTGCGTGTTGTACAGCCTAACGCAACTGCTTGGTCCGAGGCCGGGCACATATCGGCATGGCAACAATGGCGTCTGGCTGAGAACCTTTCAGTGACACTCCCTGCAGCCTCACATGCAATCCCATATTTAACCACATCCGAGATGGATTTCTGCATCGAACTCGGCAACAAGAGATGGCAATTCAACCGTCAAAGCGGTTTCCTTAGTCAAATGTGGATCGGTGATAAGAAGCAACTATTAACGCCCCTCAGAGATCAATTCACAAGGGCCCCACTGGACAATGACATAGGTGTCTCGGAAGCCACAAGGATTGACCCCAATGCCTGGGTGGAGAGATGGAAAGCCGCTGGCCATTACCAAGCCGAAGCCGCCTTATTACAATGTACCGCTGACACGTTGGCTGATGCGGTGCTCATCACCACCGCTCACGCTTGGCAACACCAAGGCAAGACCTTGTTTATCAGCAGAAAAACTTATAGGATTGATGGCAGCGGTCAAATGGCTATAACGGTCGATGTCGAAGTCGCCTCCGATACCCCTCATCCGGCTAGGATCGGTCTAAACTGTCAGCTGGCCCAAGTTGCTGAAAGAGTGAATTGGCTAGGTCTTGGGCCACAAGAAAATTATCCGGATAGACTGACAGCGGCCTGTTTTGATCGGTGGGATCTGCCTCTTTCTGATATGTACACCCCCTATGTCTTTCCTAGTGAAAACGGGCTAAGATGTGGTACTAGGGAGTTAAATTATGGACCTCACCAGTGGAGAGGTGATTTTCAGTTTAATATCTCAAGATATAGCCAACAACAATTAATGGAGACCAGTCACAGACATTTGTTGCATGCCGAGGAAGGTACTTGGTTGAACATAGATGGGTTCCACATGGGTATCGGCGGCGATGACAGTTGGTCGCCGAGTGTATCGGCAGAATTTCAATTGAGTGCAGGGAGATACCATTATCAGTTAGTATGGTGTCAAAAATGA"
    U3 = "GCGCATCAACAAAAACTCTATGTATTTTCCAATAAATTATATATCTTCAGTTTAATCTAATTCAACATCTACTTCTGTATTATTTCTATCACCCATTTTCACCGTTTTTTCGTCTTGTCTATTTTGAAGTTACCAGTGGCAGATCCCAACGCTTTTATCGCTTTTTTAAAAGATTTTCAAAATCCATATATAAACATAGGGTCCAAAAAGAAAACAAATAACAAAACAAGAAAACCTATTCGAGGTAAAGACCAGGAAAAACAAA"

    random_hill_climb_optimize(
        ckpt_path=CKPT,
        env_preproc_path=ENV_P,
        target_scaler_path=TGT_S,
        utr5=U5,
        cds_start=CDS,
        utr3=U3,
        n_starts=256,
        top_keep=16,  # Keep top K sequences per round
        neighbors_per_round=128,  # Generate N neighbors per sequence per round
        mutate_sites=8,  # Number of codons to mutate per neighbor
        steps=4096,
        batch_size=1024,
        num_workers=16,
        out_dir="outputs/cds_optimization",
        rng_seed=2025,
        tx_prefix="YGL103W",
    )
