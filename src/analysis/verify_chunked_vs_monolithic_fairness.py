"""
Methodological Verification: Monolithic V4 vs. Chunked V4 under Identical Evaluation Protocols.

Strictly verifies:
  1. Identical Argmax Rule:
       Accept iff argmax(p_contra, p_full, p_no_cov, p_partial) == full_coverage
  2. Identical Pure Full-Coverage Probability Threshold:
       Accept iff P(full_coverage) >= theta (evaluated over the identical grid)
  3. Identical Blended Coverage Probability Threshold:
       Accept iff P(full) + 0.5 * P(partial) >= theta (identical grid)
  4. Exact paired category shifts:
       Where did chunking gain true matches?
       Where did chunking introduce false positives?
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("verify_chunked_vs_monolithic_fairness")


def compute_metrics(accept_mask, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask):
    tp = int((accept_mask & is_pos).sum())
    fp = int((accept_mask & is_neg).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / is_pos.sum() if is_pos.sum() > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "llm_188_matches": int((accept_mask & llm_mask).sum()),
        "same_pol_fps": int((accept_mask & same_pol_mask).sum()),
        "same_top_fps": int((accept_mask & same_top_mask).sum()),
        "paraphrase_matches": int((accept_mask & para_mask).sum())
    }


def verify_fairness(
    mono_csv: str = "results/metrics/v4_1118_predictions.csv",
    chunk_csv: str = "results/metrics/v4_chunked_1118_predictions.csv",
    output_report: str = "results/metrics/v4_fair_comparison_report.json"
):
    p_mono = resolve(mono_csv)
    p_chunk = resolve(chunk_csv)

    if not os.path.exists(p_mono) or not os.path.exists(p_chunk):
        raise FileNotFoundError(f"Missing one or both prediction files: {p_mono}, {p_chunk}")

    df_m = load_csv(p_mono)
    df_c = load_csv(p_chunk)
    assert len(df_m) == len(df_c) == 1118, "Row counts must be exactly 1,118"

    # Ground truth masks
    is_pos = (df_m["label"].str.lower() == "entailment").to_numpy()
    is_neg = ~is_pos
    llm_mask = df_m["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_m["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_m["pair_category"] == "same_topic_diff_concept").to_numpy()
    para_mask = (df_m["pair_category"] == "within_cluster_paraphrase").to_numpy()

    # ---------------------------------------------------------
    # 1. STRICT ARGMAX COMPARISON (Identical Rule: argmax == 1)
    # ---------------------------------------------------------
    m_argmax = (df_m["v4_pred_label_4class"] == "full_coverage").to_numpy()
    c_argmax = (df_c["v4_chunk_pred_label_4class"] == "full_coverage").to_numpy()

    m_strict = compute_metrics(m_argmax, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask)
    c_strict = compute_metrics(c_argmax, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask)

    print("\n" + "="*95)
    print("PROTOCOL 1: STRICT ARGMAX EVALUATION (argmax == 'full_coverage')")
    print("="*95)
    print(f"{'Metric':<32} | {'Monolithic V4':<18} | {'Chunked V4':<18} | {'Delta':<18}")
    print("-" * 95)
    print(f"{'LLM-to-Human Matches (/188)':<32} | {m_strict['llm_188_matches']:>4d} ({m_strict['llm_188_matches']/188*100:>5.2f}%)   | {c_strict['llm_188_matches']:>4d} ({c_strict['llm_188_matches']/188*100:>5.2f}%)   | {c_strict['llm_188_matches']-m_strict['llm_188_matches']:>+4d} ({(c_strict['llm_188_matches']-m_strict['llm_188_matches'])/188*100:>+5.2f}%)")
    print(f"{'Paraphrase Matches (/352)':<32} | {m_strict['paraphrase_matches']:>4d} ({m_strict['paraphrase_matches']/352*100:>5.2f}%)   | {c_strict['paraphrase_matches']:>4d} ({c_strict['paraphrase_matches']/352*100:>5.2f}%)   | {c_strict['paraphrase_matches']-m_strict['paraphrase_matches']:>+4d}")
    print(f"{'Total True Positives (/540)':<32} | {m_strict['tp']:>4d} ({m_strict['recall']*100:>5.2f}%)   | {c_strict['tp']:>4d} ({c_strict['recall']*100:>5.2f}%)   | {c_strict['tp']-m_strict['tp']:>+4d}")
    print(f"{'Total False Positives (/578)':<32} | {m_strict['fp']:>4d}               | {c_strict['fp']:>4d}               | {c_strict['fp']-m_strict['fp']:>+4d}")
    print(f"{'  - Same-Policy FP (/240)':<32} | {m_strict['same_pol_fps']:>4d} ({m_strict['same_pol_fps']/240*100:>5.2f}%)   | {c_strict['same_pol_fps']:>4d} ({c_strict['same_pol_fps']/240*100:>5.2f}%)   | {c_strict['same_pol_fps']-m_strict['same_pol_fps']:>+4d}")
    print(f"{'  - Same-Topic FP (/180)':<32} | {m_strict['same_top_fps']:>4d} ({m_strict['same_top_fps']/180*100:>5.2f}%)   | {c_strict['same_top_fps']:>4d} ({c_strict['same_top_fps']/180*100:>5.2f}%)   | {c_strict['same_top_fps']-m_strict['same_top_fps']:>+4d}")
    print(f"{'Overall Binary Precision':<32} | {m_strict['precision']*100:>5.2f}%             | {c_strict['precision']*100:>5.2f}%             | {(c_strict['precision']-m_strict['precision'])*100:>+5.2f}%")
    print(f"{'Overall Binary Recall':<32} | {m_strict['recall']*100:>5.2f}%             | {c_strict['recall']*100:>5.2f}%             | {(c_strict['recall']-m_strict['recall'])*100:>+5.2f}%")
    print(f"{'Binary F1 Score':<32} | {m_strict['f1']:>6.4f}             | {c_strict['f1']:>6.4f}             | {c_strict['f1']-m_strict['f1']:>+6.4f}")

    # ---------------------------------------------------------
    # 2. IDENTICAL PROBABILITY SWEEP ON P(full_coverage) >= theta
    # ---------------------------------------------------------
    p_m_full = df_m["p_v4_full_cov"].to_numpy()
    p_c_full = df_c["p_v4_chunk_full_cov"].to_numpy()

    print("\n" + "="*95)
    print("PROTOCOL 2: IDENTICAL P(full_coverage) >= theta SWEEP")
    print("="*95)
    print(f"{'Theta':<7} | {'Mono Prec':<10} {'Mono Rec':<10} {'Mono F1':<9} {'Mono LLM':<10} | {'Chunk Prec':<10} {'Chunk Rec':<10} {'Chunk F1':<9} {'Chunk LLM':<10}")
    print("-" * 95)
    sweep_full = {}
    for theta in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        m_res = compute_metrics(p_m_full >= theta, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask)
        c_res = compute_metrics(p_c_full >= theta, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask)
        sweep_full[f"theta_{theta:.2f}"] = {"monolithic": m_res, "chunked": c_res}
        print(f"{theta:<7.2f} | {m_res['precision']*100:>8.2f}% {m_res['recall']*100:>8.2f}% {m_res['f1']:>8.4f} {m_res['llm_188_matches']:>4d}/188  | {c_res['precision']*100:>8.2f}% {c_res['recall']*100:>8.2f}% {c_res['f1']:>8.4f} {c_res['llm_188_matches']:>4d}/188")

    # ---------------------------------------------------------
    # 3. IDENTICAL BLENDED SWEEP: P(full) + 0.5 * P(partial) >= theta
    # ---------------------------------------------------------
    b_m = df_m["p_v4_full_cov"].to_numpy() + 0.5 * df_m["p_v4_partial_cov"].to_numpy()
    b_c = df_c["p_v4_chunk_full_cov"].to_numpy() + 0.5 * df_c["p_v4_chunk_partial_cov"].to_numpy()

    print("\n" + "="*95)
    print("PROTOCOL 3: IDENTICAL BLENDED COVERAGE SWEEP (P(full) + 0.5*P(partial) >= theta)")
    print("="*95)
    print(f"{'Theta':<7} | {'Mono Prec':<10} {'Mono Rec':<10} {'Mono F1':<9} {'Mono LLM':<10} | {'Chunk Prec':<10} {'Chunk Rec':<10} {'Chunk F1':<9} {'Chunk LLM':<10}")
    print("-" * 95)
    sweep_blend = {}
    for theta in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        m_res = compute_metrics(b_m >= theta, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask)
        c_res = compute_metrics(b_c >= theta, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask, para_mask)
        sweep_blend[f"theta_{theta:.2f}"] = {"monolithic": m_res, "chunked": c_res}
        print(f"{theta:<7.2f} | {m_res['precision']*100:>8.2f}% {m_res['recall']*100:>8.2f}% {m_res['f1']:>8.4f} {m_res['llm_188_matches']:>4d}/188  | {c_res['precision']*100:>8.2f}% {c_res['recall']*100:>8.2f}% {c_res['f1']:>8.4f} {c_res['llm_188_matches']:>4d}/188")

    report = {
        "strict_argmax_comparison": {"monolithic": m_strict, "chunked": c_strict},
        "sweep_pure_full_coverage": sweep_full,
        "sweep_blended_coverage": sweep_blend
    }
    save_json(report, resolve(output_report))
    print("="*95 + "\n")
    return report


if __name__ == "__main__":
    verify_fairness()

