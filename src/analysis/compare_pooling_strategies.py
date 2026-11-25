"""
Comparison of Chunk Aggregation Strategies on Locked 1,118-Pair Test Set:
  1. Max Pooling: max(s_i)
  2. Top-2 Average Pooling: mean(top_2(s_i))
  3. Mean Pooling: mean(s_i)
  4. LogSumExp (Smooth Max): (1/tau) * log(sum(exp(tau * s_i)))

Investigates whether Max Pooling is the primary cause of the +91 topical false positives.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.special

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("compare_pooling_strategies")


def compute_metrics(accept_mask, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask):
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
    }


def logsumexp_pool(scores: np.ndarray, tau: float = 2.0) -> float:
    """Computes smooth max via LogSumExp: (1/tau) * log(sum(exp(tau * s)))"""
    if len(scores) == 1:
        return float(scores[0])
    return float(scipy.special.logsumexp(tau * scores) / tau)


def run_pooling_comparison(
    chunk_csv: str = "results/metrics/v4_chunked_1118_predictions.csv",
    mono_csv: str = "results/metrics/v4_1118_predictions.csv",
    output_report: str = "results/metrics/pooling_comparison_report.json"
):
    p_chunk = resolve(chunk_csv)
    p_mono = resolve(mono_csv)

    df_c = load_csv(p_chunk)
    df_m = load_csv(p_mono)

    is_pos = (df_c["label"].str.lower() == "entailment").to_numpy()
    is_neg = ~is_pos
    llm_mask = df_c["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_c["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_c["pair_category"] == "same_topic_diff_concept").to_numpy()

    # Re-evaluate chunk probabilities
    # We will compute the 4 pooling rules across pairs
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from torch.utils.data import DataLoader, Dataset
    from src.analysis.evaluate_v4_chunked_inference import split_into_sentence_chunks, ChunkedPairDataset

    model_path = resolve("models/argument_nli_deberta_v3_v4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {model_path} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    flat_chunk_pairs = []
    pair_chunk_ranges = []

    for idx, row in df_c.iterrows():
        premise = str(row["premise"])
        hypo = str(row["hypothesis"])
        chunks = split_into_sentence_chunks(premise, window_size=2)
        start = len(flat_chunk_pairs)
        for c in chunks:
            flat_chunk_pairs.append((c, hypo))
        end = len(flat_chunk_pairs)
        pair_chunk_ranges.append((start, end))

    logger.info(f"Computing scores for {len(flat_chunk_pairs)} chunks...")
    dataset = ChunkedPairDataset(flat_chunk_pairs, tokenizer, max_length=192)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_chunk_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_chunk_probs.append(probs)

    all_chunk_probs = np.concatenate(all_chunk_probs, axis=0)  # Shape (N_chunks, 4)

    # Compute aggregation strategies
    max_scores = []
    top2_scores = []
    mean_scores = []
    lse_scores = []

    for i, (start, end) in enumerate(pair_chunk_ranges):
        pair_probs = all_chunk_probs[start:end]  # (k, 4)
        full_p = pair_probs[:, 1]  # P(full_coverage)

        # 1. Max Pooling
        max_scores.append(float(np.max(full_p)))

        # 2. Top-2 Average Pooling
        if len(full_p) >= 2:
            top2_scores.append(float(np.mean(np.sort(full_p)[-2:])))
        else:
            top2_scores.append(float(full_p[0]))

        # 3. Mean Pooling
        mean_scores.append(float(np.mean(full_p)))

        # 4. LogSumExp Pooling (tau=2.0)
        lse_scores.append(logsumexp_pool(full_p, tau=2.0))

    max_scores = np.array(max_scores)
    top2_scores = np.array(top2_scores)
    mean_scores = np.array(mean_scores)
    lse_scores = np.array(lse_scores)

    # Monolithic reference
    mono_scores = df_m["p_v4_full_cov"].to_numpy()

    # Compare at optimal operating thresholds
    print("\n" + "="*100)
    print("COMPARISON OF CHUNK POOLING STRATEGIES AT THRESHOLD theta = 0.35")
    print("="*100)
    print(f"{'Strategy':<24} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'LLM Matches (/188)':<20} | {'Same-Topic FP (/180)':<20}")
    print("-" * 100)

    strategies = {
        "Monolithic V4 (No chunks)": mono_scores,
        "Max Pooling": max_scores,
        "Top-2 Average Pooling": top2_scores,
        "LogSumExp Pooling": lse_scores,
        "Mean Pooling": mean_scores
    }

    report = {"threshold_0_35": {}, "threshold_0_40": {}}

    for name, s_arr in strategies.items():
        m_35 = compute_metrics(s_arr >= 0.35, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask)
        report["threshold_0_35"][name] = m_35
        print(f"{name:<24} | {m_35['precision']*100:>8.2f}% | {m_35['recall']*100:>8.2f}% | {m_35['f1']:>8.4f} | {m_35['llm_188_matches']:>4d}/188 ({m_35['llm_188_matches']/188*100:>5.1f}%)   | {m_35['same_top_fps']:>4d}/180 ({m_35['same_top_fps']/180*100:>5.1f}%)")

    print("\n" + "="*100)
    print("COMPARISON OF CHUNK POOLING STRATEGIES AT THRESHOLD theta = 0.40")
    print("="*100)
    print(f"{'Strategy':<24} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'LLM Matches (/188)':<20} | {'Same-Topic FP (/180)':<20}")
    print("-" * 100)

    for name, s_arr in strategies.items():
        m_40 = compute_metrics(s_arr >= 0.40, is_pos, is_neg, llm_mask, same_pol_mask, same_top_mask)
        report["threshold_0_40"][name] = m_40
        print(f"{name:<24} | {m_40['precision']*100:>8.2f}% | {m_40['recall']*100:>8.2f}% | {m_40['f1']:>8.4f} | {m_40['llm_188_matches']:>4d}/188 ({m_40['llm_188_matches']/188*100:>5.1f}%)   | {m_40['same_top_fps']:>4d}/180 ({m_40['same_top_fps']/180*100:>5.1f}%)")

    print("="*100 + "\n")
    save_json(report, resolve(output_report))
    return report


if __name__ == "__main__":
    run_pooling_comparison()

