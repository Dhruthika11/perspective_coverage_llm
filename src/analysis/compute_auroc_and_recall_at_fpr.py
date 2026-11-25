"""
Threshold-Free Evaluation: AUROC and Recall @ 10% FPR Across Pooling Strategies.

Evaluates:
  1. Forward Baseline (Reference)
  2. Argument-NLI V3 (Reference)
  3. Monolithic V4 (Full Argument)
  4. Max Pooling V4 (max chunk)
  5. Top-2 Average Pooling V4 (mean of top 2 chunks)
  6. Mean Pooling V4 (mean of all chunks)
  7. LogSumExp Pooling V4 (smooth max, tau=2.0)

Computes:
  - Overall AUROC (540 Positives vs. 578 Negatives)
  - Overall Recall at FPR <= 10% (Target: max TPR at <= 58 total False Positives)
  - LLM-to-Human Recall at FPR <= 10% (out of 188 LLM positive pairs)
  - Paraphrase Recall at FPR <= 10% (out of 352 human pairs)
  - Category-Specific Separation AUROC:
      * LLM Positives vs. Same-Policy Negatives (188 vs. 240)
      * LLM Positives vs. Same-Topic Negatives (188 vs. 180)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.special
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("compute_auroc_and_recall_at_fpr")


def get_recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10):
    """Computes exact recall and threshold at FPR <= target_fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0, float(thresholds[0]), 0
    best_i = valid_idx[-1]
    thresh = float(thresholds[best_i])
    actual_fpr = float(fpr[best_i])
    overall_rec = float(tpr[best_i])
    return overall_rec, actual_fpr, thresh, best_i


def logsumexp_pool(scores: np.ndarray, tau: float = 2.0) -> float:
    if len(scores) == 1:
        return float(scores[0])
    return float(scipy.special.logsumexp(tau * scores) / tau)


def run_threshold_free_evaluation(
    model_dir: str = "models/argument_nli_deberta_v3_v4",
    test_csv: str = "data/nli_training_v3/test.csv",
    mono_csv: str = "results/metrics/v4_1118_predictions.csv",
    output_report: str = "results/metrics/pooling_auroc_and_recall_at_fpr_report.json",
    batch_size: int = 64
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src.analysis.evaluate_v4_chunked_inference import split_into_sentence_chunks, ChunkedPairDataset

    model_path = resolve(model_dir)
    test_path = resolve(test_csv)
    mono_path = resolve(mono_csv)
    out_json = resolve(output_report)

    df_test = load_csv(test_path)
    df_mono = load_csv(mono_path) if os.path.exists(mono_path) else None

    # Masks
    is_pos = (df_test["label"].str.lower() == "entailment").to_numpy()
    is_neg = ~is_pos
    llm_mask = df_test["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_test["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_test["pair_category"] == "same_topic_diff_concept").to_numpy()
    para_mask = (df_test["pair_category"] == "within_cluster_paraphrase").to_numpy()

    # Model inference for chunks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading V4 model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    flat_chunk_pairs = []
    pair_chunk_ranges = []

    for idx, row in df_test.iterrows():
        premise = str(row["premise"])
        hypo = str(row["hypothesis"])
        chunks = split_into_sentence_chunks(premise, window_size=2)
        start = len(flat_chunk_pairs)
        for c in chunks:
            flat_chunk_pairs.append((c, hypo))
        end = len(flat_chunk_pairs)
        pair_chunk_ranges.append((start, end))

    logger.info(f"Computing forward pass for {len(flat_chunk_pairs)} chunks...")
    dataset = ChunkedPairDataset(flat_chunk_pairs, tokenizer, max_length=192)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_chunk_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_chunk_probs.append(probs)

    all_chunk_probs = np.concatenate(all_chunk_probs, axis=0)  # (N_chunks, 4)

    # Compute aggregation score vectors
    max_scores = []
    top2_scores = []
    mean_scores = []
    lse_scores = []

    for i, (start, end) in enumerate(pair_chunk_ranges):
        pair_probs = all_chunk_probs[start:end]
        full_p = pair_probs[:, 1]  # P(full_coverage)

        max_scores.append(float(np.max(full_p)))
        if len(full_p) >= 2:
            top2_scores.append(float(np.mean(np.sort(full_p)[-2:])))
        else:
            top2_scores.append(float(full_p[0]))
        mean_scores.append(float(np.mean(full_p)))
        lse_scores.append(logsumexp_pool(full_p, tau=2.0))

    score_dict = {
        "Max Pooling": np.array(max_scores),
        "Top-2 Average Pooling": np.array(top2_scores),
        "LogSumExp Pooling": np.array(lse_scores),
        "Mean Pooling": np.array(mean_scores)
    }

    # Add Monolithic V4
    if df_mono is not None and "p_v4_full_cov" in df_mono.columns:
        score_dict["Monolithic V4"] = df_mono["p_v4_full_cov"].to_numpy()

    # Add Forward Baseline and V3 if available
    master_p = resolve("results/metrics/master_1118_predictions.csv")
    if os.path.exists(master_p):
        df_master = load_csv(master_p)
        score_dict["Forward Baseline (Zero-Shot)"] = df_master["p_forward_base_entail"].to_numpy()
        score_dict["Argument-NLI V3"] = df_master["p_v3_entail"].to_numpy()

    # Reorder dictionary to put baselines first
    preferred_order = [
        "Forward Baseline (Zero-Shot)",
        "Argument-NLI V3",
        "Monolithic V4",
        "Max Pooling",
        "Top-2 Average Pooling",
        "LogSumExp Pooling",
        "Mean Pooling"
    ]
    ordered_scores = {k: score_dict[k] for k in preferred_order if k in score_dict}

    print("\n" + "="*115)
    print("THRESHOLD-FREE EVALUATION: AUROC & RECALL @ 10% FPR (N = 1,118)")
    print("="*115)
    header = f"{'Method':<28} | {'Overall AUROC':<13} | {'Recall @ 10% FPR':<16} | {'LLM Rec @ 10% FPR':<18} | {'Para Rec @ 10% FPR':<18} | {'Thresh @ 10%':<12}"
    print(header)
    print("-" * 115)

    report_data = {}

    for name, s in ordered_scores.items():
        # 1. Overall AUROC
        overall_auc = roc_auc_score(is_pos, s)

        # 2. Recall at 10% FPR
        rec_10, act_fpr, thresh_10, _ = get_recall_at_fpr(is_pos, s, target_fpr=0.10)

        # Boolean accept mask at that threshold
        accept_10 = (s >= thresh_10)
        llm_caught = int((accept_10 & llm_mask).sum())
        llm_rec_10 = llm_caught / 188.0
        para_caught = int((accept_10 & para_mask).sum())
        para_rec_10 = para_caught / 352.0
        total_fps = int((accept_10 & is_neg).sum())

        # Category separation AUROCs
        # A. LLM Positives (188) vs. Same-Policy Negatives (240)
        sub_sp_mask = llm_mask | same_pol_mask
        y_sp = is_pos[sub_sp_mask]
        s_sp = s[sub_sp_mask]
        auc_llm_vs_sp = roc_auc_score(y_sp, s_sp) if len(np.unique(y_sp)) > 1 else 0.5

        # B. LLM Positives (188) vs. Same-Topic Negatives (180)
        sub_st_mask = llm_mask | same_top_mask
        y_st = is_pos[sub_st_mask]
        s_st = s[sub_st_mask]
        auc_llm_vs_st = roc_auc_score(y_st, s_st) if len(np.unique(y_st)) > 1 else 0.5

        report_data[name] = {
            "overall_auroc": round(float(overall_auc), 4),
            "recall_at_10_fpr": round(float(rec_10), 4),
            "actual_fpr": round(float(act_fpr), 4),
            "threshold_at_10_fpr": round(float(thresh_10), 4),
            "total_fps_at_10_fpr": total_fps,
            "llm_188_matches_at_10_fpr": llm_caught,
            "llm_188_recall_at_10_fpr": round(float(llm_rec_10), 4),
            "paraphrase_352_matches_at_10_fpr": para_caught,
            "paraphrase_352_recall_at_10_fpr": round(float(para_rec_10), 4),
            "auc_llm_vs_same_policy": round(float(auc_llm_vs_sp), 4),
            "auc_llm_vs_same_topic": round(float(auc_llm_vs_st), 4)
        }

        print(f"{name:<28} | {overall_auc:>11.4f}   | {rec_10*100:>13.2f}%   | {llm_caught:>3d}/188 ({llm_rec_10*100:>5.1f}%)   | {para_caught:>3d}/352 ({para_rec_10*100:>5.1f}%)   | {thresh_10:>10.4f}")

    print("\n" + "="*115)
    print("CATEGORY-SPECIFIC SEPARATION AUROC (LLM Positives vs. Hard Negatives)")
    print("="*115)
    print(f"{'Method':<28} | {'AUC (LLM Pos vs. Same-Policy Neg)':<35} | {'AUC (LLM Pos vs. Same-Topic Neg)':<35}")
    print("-" * 115)
    for name, d in report_data.items():
        print(f"{name:<28} | {d['auc_llm_vs_same_policy']:>30.4f}      | {d['auc_llm_vs_same_topic']:>30.4f}")
    print("="*115 + "\n")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    save_json(report_data, out_json)
    logger.info(f"Saved threshold-free report to {out_json}")
    return report_data


if __name__ == "__main__":
    run_threshold_free_evaluation()

