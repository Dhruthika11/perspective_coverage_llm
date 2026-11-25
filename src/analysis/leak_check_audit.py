"""
Automated Verbatim-Containment & Lexical Overlap Leak Check Audit.

Standing pipeline test that verifies datasets are free from artificial
verbatim concatenation shortcuts and ensures token overlap is matched between
positive and negative pairs.
"""

import os
import sys
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv
from src.utils.logging_utils import get_logger

logger = get_logger("leak_check_audit")


def audit_dataset_leakage(
    name: str,
    path: str,
    p_col: str = "premise",
    h_col: str = "hypothesis",
    l_col: str = None
) -> Dict[str, Any]:
    resolved_path = resolve(path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Dataset not found at {resolved_path}")

    df = load_csv(resolved_path)

    # Determine positive mask
    if l_col and l_col in df.columns:
        lbl = df[l_col].astype(str).str.lower()
        pos_mask = lbl.isin(["1", "1.0", "entailment", "full_coverage"])
    elif "label_4class" in df.columns:
        pos_mask = df["label_4class"] == 1
    elif "label" in df.columns:
        pos_mask = df["label"].astype(str).str.lower().isin(["entailment", "full_coverage"])
    elif "proposed_label" in df.columns:
        pos_mask = df["proposed_label"].astype(str).str.lower().isin(["entailment", "full_coverage"])
    elif "ground_truth_label" in df.columns:
        pos_mask = df["ground_truth_label"].astype(int) == 1
    else:
        raise ValueError(f"No recognized label column in {name}: {df.columns.tolist()}")

    def get_metrics(sub_df: pd.DataFrame) -> Dict[str, Any]:
        if len(sub_df) == 0:
            return {"count": 0, "exact_substr_rate": 0.0, "mean_token_overlap": 0.0, "mean_p_words": 0.0, "mean_h_words": 0.0}

        exact_substr = []
        token_overlap = []
        p_lens = []
        h_lens = []
        for _, r in sub_df.iterrows():
            p = str(r[p_col]).lower().strip()
            h = str(r[h_col]).lower().strip()
            # Check exact substring containment
            exact_substr.append(h in p)

            p_toks = set(p.split())
            h_toks = set(h.split())
            p_lens.append(len(str(r[p_col]).strip().split()))
            h_lens.append(len(str(r[h_col]).strip().split()))

            if len(h_toks) > 0:
                token_overlap.append(len(h_toks.intersection(p_toks)) / len(h_toks))
            else:
                token_overlap.append(0.0)

        return {
            "count": len(sub_df),
            "exact_substr_rate": round(float(np.mean(exact_substr) * 100), 2),
            "mean_token_overlap": round(float(np.mean(token_overlap) * 100), 2),
            "mean_p_words": round(float(np.mean(p_lens)), 1),
            "mean_h_words": round(float(np.mean(h_lens)), 1)
        }

    pos_res = get_metrics(df[pos_mask])
    neg_res = get_metrics(df[~pos_mask])

    overlap_gap = round(pos_res["mean_token_overlap"] - neg_res["mean_token_overlap"], 2)
    # Leak criteria: exact substring rate > 5% or overlap gap > 20%
    has_leak = (pos_res["exact_substr_rate"] > 5.0) or (overlap_gap > 20.0)

    result = {
        "dataset_name": name,
        "dataset_path": path,
        "total_pairs": len(df),
        "positives": pos_res,
        "negatives": neg_res,
        "overlap_gap_pos_minus_neg": overlap_gap,
        "verbatim_containment_detected": pos_res["exact_substr_rate"] > 5.0,
        "clean_standard_verified": not has_leak
    }
    return result


def main():
    datasets_to_audit = [
        ("Locked Test Set (Old Standard)", "data/nli_training_v3/test.csv"),
        ("V5 Training Set", "data/nli_training_v5/train.csv"),
        ("V5 Validation Set", "data/nli_training_v5/val.csv"),
        ("Fresh Claims Benchmark", "data/fresh_claim_benchmark/test_fresh_claims.csv"),
        ("Mistral Adjudicated Benchmark", "data/mistral_benchmark/mistral_adjudicated_benchmark.csv"),
        ("Clean Balanced LOGFO Benchmark", "data/logfo_benchmark/clean_balanced_logfo_benchmark.csv"),
        ("Rebalanced LOGFO Benchmark (Suspect)", "data/logfo_benchmark/balanced_multifamily_benchmark.csv")
    ]

    print("=" * 110)
    print("VERBATIM-CONTAINMENT & LEXICAL OVERLAP AUDIT REPORT")
    print("=" * 110)
    print(f"{'Dataset Name':<35} | {'Pairs (Pos/Neg)':<16} | {'Pos Substr%':<11} | {'Pos Overlap%':<12} | {'Neg Overlap%':<12} | {'Status'}")
    print("-" * 110)

    for name, path in datasets_to_audit:
        try:
            res = audit_dataset_leakage(name, path)
            pos = res["positives"]
            neg = res["negatives"]
            pairs_str = f"{res['total_pairs']} ({pos['count']}/{neg['count']})"
            status = "CLEAN [PASSED]" if res["clean_standard_verified"] else "LEAKED [FAIL]"
            print(f"{name:<35} | {pairs_str:<16} | {pos['exact_substr_rate']:>10.2f}% | {pos['mean_token_overlap']:>11.2f}% | {neg['mean_token_overlap']:>11.2f}% | {status}")
        except Exception as e:
            print(f"{name:<35} | ERROR: {e}")

    print("=" * 110)


if __name__ == "__main__":
    main()
