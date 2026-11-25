"""
Argument-NLI V5 Disaggregated Benchmark Evaluation Protocol.

Evaluates the V5 model across decoupled regimes to prevent confounding:
  1. Stratum 1: Short Human Benchmark (N = 750)
     - Paraphrase Positives (N = 352) vs. Same-Stance Negatives (N = 240),
       Opposite-Stance Conflicts (N = 120), and Unrelated Claims (N = 38).
     - Evaluates whether Warm-Started MNLI Head successfully restores short-sentence AUROC and contradiction detection.
  2. Stratum 2: Long LLM Arguments Benchmark (N = 368)
     - LLM Positives (N = 188) vs. Same-Topic Negatives (N = 180).
     - Compares Monolithic V5 vs. 2-Sentence Chunked V5 (Max, Top-2, LogSumExp pooling).
  3. Stratum 3: Stance-Conflation Separation
     - LLM Positives (N = 188) vs. Same-Policy Negatives (N = 240).
     - Verifies whether chunking maintains high reasoning separation (AUC > 0.70).
  4. Stratum 4: Macro Combined Test Benchmark (N = 1,118)
     - Overall AUROC, Recall @ 10% FPR, and Strict Argmax Performance.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.special
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.analysis.evaluate_zeroshot_chunked_inference import split_into_sentence_chunks, ChunkedPairDataset

logger = get_logger("evaluate_v5_benchmark")

ID_TO_LABEL_V5 = {
    0: "contradiction",
    1: "full_coverage",
    2: "no_coverage",
    3: "partial_coverage"
}


def logsumexp_pool(scores: np.ndarray, tau: float = 2.0, normalize: bool = True) -> float:
    """
    Computes LogSumExp pooling.
    When normalize=True, subtracts log(M)/tau to remove the chunk-count length bias:
      LSE_norm = (1/tau) * log(mean(exp(tau * scores)))
    This prevents multi-chunk arguments from gaining an artificial mathematical advantage over 1-chunk arguments.
    """
    if len(scores) <= 1:
        return float(scores[0]) if len(scores) == 1 else 0.0
    raw_lse = float(scipy.special.logsumexp(tau * scores) / tau)
    if normalize:
        return float(raw_lse - (np.log(len(scores)) / tau))
    return raw_lse


def compute_recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10) -> Tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0, float(thresholds[0])
    best_i = valid_idx[-1]
    return float(tpr[best_i]), float(fpr[best_i]), float(thresholds[best_i])


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def run_v5_disaggregated_evaluation(
    model_dir: str = "models/argument_nli_deberta_v3_v5",
    test_csv: str = "data/nli_training_v3/test.csv",
    output_pred_csv: str = "results/metrics/v5_1118_predictions.csv",
    output_report_json: str = "results/metrics/v5_disaggregated_benchmark_report.json",
    window_size: int = 2,
    batch_size: int = 64
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = resolve(model_dir)
    test_path = resolve(test_csv)
    out_csv = resolve(output_pred_csv)
    out_json = resolve(output_report_json)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test CSV at {test_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing V5 model at {model_path}. Please train V5 first.")

    df_test = load_csv(test_path)
    logger.info(f"Loaded {len(df_test)} pairs from {test_path}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading V5 model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # 1. Monolithic Forward Pass (Full Premise)
    logger.info("Computing monolithic forward pass on full premises...")
    mono_pairs = [(str(row["premise"]), str(row["hypothesis"])) for _, row in df_test.iterrows()]
    mono_dataset = ChunkedPairDataset(mono_pairs, tokenizer, max_length=256)
    mono_loader = DataLoader(mono_dataset, batch_size=batch_size, shuffle=False)

    mono_probs = []
    with torch.no_grad():
        for batch in mono_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            mono_probs.append(probs)

    mono_probs = np.concatenate(mono_probs, axis=0)  # Shape (1118, 4)

    # 2. Chunked Forward Pass (Window size = 2)
    logger.info(f"Chunking premises (window_size={window_size})...")
    flat_chunk_pairs = []
    pair_chunk_ranges = []
    pair_chunk_texts = []

    for _, row in df_test.iterrows():
        premise = str(row["premise"])
        hypo = str(row["hypothesis"])
        chunks = split_into_sentence_chunks(premise, window_size=window_size)
        pair_chunk_texts.append(chunks)
        start = len(flat_chunk_pairs)
        for c in chunks:
            flat_chunk_pairs.append((c, hypo))
        end = len(flat_chunk_pairs)
        pair_chunk_ranges.append((start, end))

    logger.info(f"Generated {len(flat_chunk_pairs)} chunks across 1,118 pairs. Scoring chunks...")
    chunk_dataset = ChunkedPairDataset(flat_chunk_pairs, tokenizer, max_length=192)
    chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=False)

    all_chunk_probs = []
    with torch.no_grad():
        for batch in chunk_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_chunk_probs.append(probs)

    all_chunk_probs = np.concatenate(all_chunk_probs, axis=0)

    # 3. Pooling Aggregations per Pair
    chunk_max_scores = []
    chunk_top2_scores = []
    chunk_lse_scores = []
    chunk_raw_lse_scores = []
    chunk_mean_scores = []
    best_chunk_texts = []

    for i, (start, end) in enumerate(pair_chunk_ranges):
        pair_probs = all_chunk_probs[start:end]
        chunks = pair_chunk_texts[i]
        full_p = pair_probs[:, 1]  # Class 1: full_coverage

        best_j = int(np.argmax(full_p))
        best_chunk_texts.append(chunks[best_j])
        chunk_max_scores.append(float(full_p[best_j]))
        if len(full_p) >= 2:
            chunk_top2_scores.append(float(np.mean(np.sort(full_p)[-2:])))
        else:
            chunk_top2_scores.append(float(full_p[0]))
        chunk_lse_scores.append(logsumexp_pool(full_p, tau=2.0, normalize=True))
        chunk_raw_lse_scores.append(logsumexp_pool(full_p, tau=2.0, normalize=False))
        chunk_mean_scores.append(float(np.mean(full_p)))

    chunk_max_scores = np.array(chunk_max_scores)
    chunk_top2_scores = np.array(chunk_top2_scores)
    chunk_lse_scores = np.array(chunk_lse_scores)
    chunk_raw_lse_scores = np.array(chunk_raw_lse_scores)
    chunk_mean_scores = np.array(chunk_mean_scores)

    # Record to DataFrame
    df_test["p_v5_mono_contra"] = mono_probs[:, 0]
    df_test["p_v5_mono_full"] = mono_probs[:, 1]
    df_test["p_v5_mono_none"] = mono_probs[:, 2]
    df_test["p_v5_mono_partial"] = mono_probs[:, 3]
    df_test["v5_mono_pred"] = [ID_TO_LABEL_V5[i] for i in np.argmax(mono_probs, axis=1)]

    df_test["p_v5_chunk_max_full"] = chunk_max_scores
    df_test["p_v5_chunk_top2_full"] = chunk_top2_scores
    df_test["p_v5_chunk_lse_full"] = chunk_lse_scores
    df_test["p_v5_chunk_raw_lse_full"] = chunk_raw_lse_scores
    df_test["p_v5_chunk_mean_full"] = chunk_mean_scores
    df_test["v5_best_chunk_text"] = best_chunk_texts

    # Length-Routed Hybrid Strategy (Mono on <50w, Chunked Max on >=50w)
    is_short_premise = (df_test["premise_words"] < 50).to_numpy()
    hybrid_scores = np.where(is_short_premise, df_test["p_v5_mono_full"], df_test["p_v5_chunk_max_full"])
    df_test["p_v5_hybrid_full"] = hybrid_scores

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_test.to_csv(out_csv, index=False)
    logger.info(f"Saved V5 test predictions to {out_csv}")

    # Masks
    is_pos = (df_test["label"].str.lower() == "entailment").to_numpy()
    is_neg = ~is_pos
    
    # Adaptive short and long masks
    short_cats = [
        "within_cluster_paraphrase",
        "same_stance_different_reasoning",
        "opposite_stance_conflict",
        "unrelated_different_claim"
    ]
    if "length_regime" in df_test.columns:
        short_mask = (df_test["length_regime"] == "short_human").to_numpy() | df_test["pair_category"].isin(short_cats).to_numpy()
        long_mask = df_test["length_regime"].isin(["long_llm", "long_argument"]).to_numpy() | df_test["pair_category"].str.contains("long|discourse|concept", case=False, na=False).to_numpy()
    else:
        short_mask = df_test["pair_category"].isin(short_cats).to_numpy()
        long_cats = [
            "long_llm_to_human_positive_qwen2_5_1_5b_instruct",
            "long_llm_to_human_positive_qwen2_5_32b_instruct",
            "long_llm_to_human_positive_qwen2_5_3b_instruct",
            "same_topic_diff_concept"
        ]
        long_mask = df_test["pair_category"].isin(long_cats).to_numpy()

    # Fallback to word counts if empty
    if np.sum(long_mask) == 0 and "premise_words" in df_test.columns:
        long_mask = (df_test["premise_words"] >= 50).to_numpy()
    if np.sum(short_mask) == 0 and "premise_words" in df_test.columns:
        short_mask = (df_test["premise_words"] < 50).to_numpy()

    llm_pos_mask = df_test["pair_category"].str.contains("long_llm_to_human_positive|extended_discourse", case=False, na=False).to_numpy()
    same_pol_mask = (df_test["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_test["pair_category"] == "same_topic_diff_concept").to_numpy()

    # -------------------------------------------------------------------------
    # STRATUM 1: Short Human Benchmark
    # -------------------------------------------------------------------------
    y_short = is_pos[short_mask].astype(int)
    s_short_mono = df_test.loc[short_mask, "p_v5_mono_full"].to_numpy()
    if len(np.unique(y_short)) == 2:
        auc_short = float(roc_auc_score(y_short, s_short_mono))
        rec10_short, fpr_short, th_short = compute_recall_at_fpr(y_short, s_short_mono, target_fpr=0.10)
    else:
        auc_short, rec10_short, fpr_short, th_short = 0.5, 0.0, 0.0, 0.5

    # -------------------------------------------------------------------------
    # STRATUM 2: Long Arguments Benchmark
    # -------------------------------------------------------------------------
    y_long = is_pos[long_mask].astype(int)
    s_long_mono = df_test.loc[long_mask, "p_v5_mono_full"].to_numpy()
    s_long_chunk = df_test.loc[long_mask, "p_v5_chunk_max_full"].to_numpy()
    if np.sum(long_mask) > 0 and len(np.unique(y_long)) == 2:
        auc_long_mono = float(roc_auc_score(y_long, s_long_mono))
        auc_long_chunk = float(roc_auc_score(y_long, s_long_chunk))
        rec10_long_chunk, _, _ = compute_recall_at_fpr(y_long, s_long_chunk, target_fpr=0.10)
    else:
        auc_long_mono, auc_long_chunk, rec10_long_chunk = 0.5, 0.5, 0.0

    # -------------------------------------------------------------------------
    # STRATUM 3: Stance-Conflation Separation
    # -------------------------------------------------------------------------
    sp_mask = llm_pos_mask | same_pol_mask
    y_sp = is_pos[sp_mask].astype(int)
    if np.sum(sp_mask) > 0 and len(np.unique(y_sp)) == 2:
        auc_sp_mono = float(roc_auc_score(y_sp, df_test.loc[sp_mask, "p_v5_mono_full"].to_numpy()))
        auc_sp_chunk = float(roc_auc_score(y_sp, df_test.loc[sp_mask, "p_v5_chunk_max_full"].to_numpy()))
        auc_sp_lse = float(roc_auc_score(y_sp, df_test.loc[sp_mask, "p_v5_chunk_lse_full"].to_numpy()))
        auc_sp_raw_lse = float(roc_auc_score(y_sp, df_test.loc[sp_mask, "p_v5_chunk_raw_lse_full"].to_numpy()))
    else:
        auc_sp_mono, auc_sp_chunk, auc_sp_lse, auc_sp_raw_lse = 0.5, 0.5, 0.5, 0.5

    # -------------------------------------------------------------------------
    # STRATUM 4: Overall Combined Benchmark
    # -------------------------------------------------------------------------
    y_all = is_pos.astype(int)
    auc_all_mono = float(roc_auc_score(y_all, df_test["p_v5_mono_full"].to_numpy())) if len(np.unique(y_all)) == 2 else 0.5
    auc_all_chunk = float(roc_auc_score(y_all, df_test["p_v5_chunk_max_full"].to_numpy())) if len(np.unique(y_all)) == 2 else 0.5
    auc_all_hybrid = float(roc_auc_score(y_all, hybrid_scores)) if len(np.unique(y_all)) == 2 else 0.5
    rec10_all_chunk, _, _ = compute_recall_at_fpr(y_all, df_test["p_v5_chunk_max_full"].to_numpy(), target_fpr=0.10) if len(np.unique(y_all)) == 2 else (0.0, 0.0, 0.5)
    rec10_all_hybrid, _, th_10_hybrid = compute_recall_at_fpr(y_all, hybrid_scores, target_fpr=0.10) if len(np.unique(y_all)) == 2 else (0.0, 0.0, 0.5)

    # Contradiction Detection Recall on Opposite-Stance Conflicts
    opp_mask = (df_test["pair_category"] == "opposite_stance_conflict").to_numpy()
    n_opp_total = int(np.sum(opp_mask))
    opp_preds = df_test.loc[opp_mask, "v5_mono_pred"].value_counts().to_dict() if n_opp_total > 0 else {}
    contra_caught = opp_preds.get("contradiction", 0)

    # -------------------------------------------------------------------------
    # PRECISION & CLASSIFICATION METRICS AUDIT
    # -------------------------------------------------------------------------
    def calc_prec_rec_f1(pred_mask: np.ndarray, truth_mask: np.ndarray, neg_mask: np.ndarray):
        if len(pred_mask) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
        tp = int((pred_mask & truth_mask).sum())
        fp = int((pred_mask & neg_mask).sum())
        fn = int((~pred_mask & truth_mask).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / truth_mask.sum() if truth_mask.sum() > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    strict_mono_pred = (df_test["v5_mono_pred"] == "full_coverage").to_numpy()
    strict_overall = calc_prec_rec_f1(strict_mono_pred, is_pos, is_neg)
    strict_short = calc_prec_rec_f1(strict_mono_pred[short_mask], is_pos[short_mask], is_neg[short_mask]) if np.sum(short_mask) > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    strict_long = calc_prec_rec_f1(strict_mono_pred[long_mask], is_pos[long_mask], is_neg[long_mask]) if np.sum(long_mask) > 0 else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    # Qualified Policy (full_coverage OR partial_coverage)
    qual_mono_pred = df_test["v5_mono_pred"].isin(["full_coverage", "partial_coverage"]).to_numpy()
    qual_overall = calc_prec_rec_f1(qual_mono_pred, is_pos, is_neg)

    # Precision at 10% FPR constraint
    rec_10_all, act_fpr_all, th_10_all = compute_recall_at_fpr(y_all, df_test["p_v5_mono_full"].to_numpy(), target_fpr=0.10)
    pass_10_all = (df_test["p_v5_mono_full"] >= th_10_all).to_numpy()
    prec_at_10_fpr = calc_prec_rec_f1(pass_10_all, is_pos, is_neg)

    # Operating Threshold Sweep (theta in 0.20 to 0.90) for Hybrid Router
    thetas = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    hybrid_sweep_table = []
    mono_sweep_table = []
    for th in thetas:
        # Hybrid
        h_mask = (hybrid_scores >= th)
        hm = calc_prec_rec_f1(h_mask, is_pos, is_neg)
        hybrid_sweep_table.append({
            "theta": th, "precision": hm["precision"], "recall": hm["recall"], "f1": hm["f1"], "tp": hm["tp"], "fp": hm["fp"]
        })
        # Mono
        m_mask = (df_test["p_v5_mono_full"] >= th).to_numpy()
        mm = calc_prec_rec_f1(m_mask, is_pos, is_neg)
        mono_sweep_table.append({
            "theta": th, "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"], "tp": mm["tp"], "fp": mm["fp"]
        })

    report = {
        "stratum_1_short_human_750": {
            "total_pairs": int(short_mask.sum()),
            "auroc": round(float(auc_short), 4),
            "recall_at_10_fpr": round(float(rec10_short), 4),
            "threshold_at_10_fpr": round(float(th_short), 4),
            "opposite_stance_contradiction_detections": f"{contra_caught}/120",
            "strict_argmax": strict_short
        },
        "stratum_2_long_llm_368": {
            "total_pairs": int(long_mask.sum()),
            "monolithic_auroc": round(float(auc_long_mono), 4),
            "chunked_max_auroc": round(float(auc_long_chunk), 4),
            "chunked_recall_at_10_fpr": round(float(rec10_long_chunk), 4),
            "strict_argmax": strict_long
        },
        "stratum_3_stance_conflation_428": {
            "monolithic_auroc": round(float(auc_sp_mono), 4),
            "chunked_max_auroc": round(float(auc_sp_chunk), 4),
            "chunked_lse_auroc": round(float(auc_sp_lse), 4)
        },
        "stratum_4_macro_combined_1118": {
            "monolithic_auroc": round(float(auc_all_mono), 4),
            "chunked_max_auroc": round(float(auc_all_chunk), 4),
            "hybrid_router_auroc": round(float(auc_all_hybrid), 4),
            "chunked_recall_at_10_fpr": round(float(rec10_all_chunk), 4),
            "hybrid_recall_at_10_fpr": round(float(rec10_all_hybrid), 4),
            "strict_argmax": strict_overall,
            "qualified_argmax": qual_overall,
            "precision_at_10_fpr_constraint": prec_at_10_fpr
        },
        "threshold_sweep_monolithic": mono_sweep_table,
        "threshold_sweep_hybrid_router": hybrid_sweep_table
    }

    save_json(report, out_json)
    logger.info(f"Saved disaggregated benchmark report to {out_json}")

    print("\n" + "=" * 105)
    print("ARGUMENT-NLI V5 DISAGGREGATED BENCHMARK EVALUATION")
    print("=" * 105)
    n_short_total = int(np.sum(short_mask))
    n_long_total = int(np.sum(long_mask))
    n_sp_total = int(np.sum(sp_mask))
    print(f"Stratum 1: Short Human (N={n_short_total}) AUROC:                {auc_short:.4f} (Rec@10%: {rec10_short*100:.2f}%)")
    if n_opp_total > 0:
        print(f"           Contradictions Detected (/{n_opp_total}):          {contra_caught}/{n_opp_total} ({contra_caught/n_opp_total*100:.1f}%)")
    print(f"           Strict Argmax Precision:                  {strict_short['precision']*100:.2f}% | Recall: {strict_short['recall']*100:.2f}% | F1: {strict_short['f1']:.4f}")
    print("-" * 105)
    print(f"Stratum 2: Long LLM Arguments (N={n_long_total}) Mono AUROC:     {auc_long_mono:.4f}")
    print(f"           Long LLM Arguments (N={n_long_total}) Chunk AUROC:    {auc_long_chunk:.4f}")
    print(f"           Strict Argmax Precision:                  {strict_long['precision']*100:.2f}% | Recall: {strict_long['recall']*100:.2f}% | F1: {strict_long['f1']:.4f}")
    print("-" * 105)
    print(f"Stratum 3: Stance Separation (N={n_sp_total}):   Mono: {auc_sp_mono:.4f} | Chunk Max: {auc_sp_chunk:.4f} | Norm LSE: {auc_sp_lse:.4f} | Raw LSE: {auc_sp_raw_lse:.4f}")
    print("-" * 105)
    print(f"Stratum 4: Macro Combined (N={len(df_test)}) Mono AUROC:       {auc_all_mono:.4f}")
    print(f"           Macro Combined (N={len(df_test)}) Chunk Max AUROC:  {auc_all_chunk:.4f}")
    print(f"           Macro Combined (N={len(df_test)}) Hybrid AUROC:     {auc_all_hybrid:.4f}")
    print(f"           Strict Argmax Precision (full_cov):       {strict_overall['precision']*100:.2f}% | Recall: {strict_overall['recall']*100:.2f}% | F1: {strict_overall['f1']:.4f}")
    print(f"           Precision at 10% FPR constraint:          {prec_at_10_fpr['precision']*100:.2f}% | Recall: {prec_at_10_fpr['recall']*100:.2f}% | F1: {prec_at_10_fpr['f1']:.4f}")
    print("=" * 105)

    print("\n" + "-" * 95)
    print("LENGTH-ROUTED HYBRID SWEEP (Mono on <50w, Chunk on >=50w)")
    print("-" * 95)
    print(f"{'Theta':<8} | {'Precision':<12} | {'Recall':<12} | {'F1':<10} | {'True Pos':<10} | {'False Pos'}")
    print("-" * 95)
    for row in hybrid_sweep_table:
        print(f"{row['theta']:<8.2f} | {row['precision']*100:>10.2f}% | {row['recall']*100:>10.2f}% | {row['f1']:>8.4f} | {row['tp']:>8d}   | {row['fp']:>9d}")
    print("=" * 95 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Argument-NLI V5 Disaggregated Benchmark")
    parser.add_argument("--model_dir", type=str, default="models/argument_nli_deberta_v3_v5")
    parser.add_argument("--test_csv", type=str, default="data/nli_training_v3/test.csv")
    parser.add_argument("--output_csv", type=str, default="results/metrics/v5_1118_predictions.csv")
    parser.add_argument("--output_report", type=str, default="results/metrics/v5_disaggregated_benchmark_report.json")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    run_v5_disaggregated_evaluation(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        output_pred_csv=args.output_csv,
        output_report_json=args.output_report,
        window_size=args.window_size,
        batch_size=args.batch_size
    )

