"""
Zero-Shot Baseline Chunked Inference (Sentence-Group Chunk Scorer & Aggregation).

Implements the Zero-Shot Chunker Ablation Experiment:
  - Long argument premise (up to 250 words) is split into sentence groups (chunks, window_size=2).
  - Uses an off-the-shelf Zero-Shot NLI Cross-Encoder (default: cross-encoder/nli-deberta-v3-base)
    without task-specific fine-tuning as the chunk scorer.
  - Scores each (chunk, hypothesis) pair for entailment probability P_entail.
  - Aggregates chunk scores per pair across multiple pooling strategies:
      * Max Pooling: max_j P(chunk_j -> hypothesis)
      * Top-2 Average Pooling: mean(top_2(P_chunks))
      * LogSumExp Pooling: tau^-1 * logsumexp(tau * P_chunks)
      * Mean Pooling: mean(P_chunks)
  - Identifies the winning chunk (highest entailment score) and extracts its text.
  - Performs an apples-to-apples comparison between:
      * Monolithic Zero-Shot Baseline (full premise)
      * Chunked Zero-Shot Baseline (window_size=2)
  - Computes threshold-free AUROC, Recall @ 10% FPR, and Category-Specific Separation AUROC
    (LLM Positives vs. Same-Policy Negatives and Same-Topic Negatives).
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.special
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_zeroshot_chunked_inference")


def split_into_sentence_chunks(
    text: str,
    window_size: int = 2,
    step_size: int = 1,
    min_chunk_words: int = 8
) -> List[str]:
    """
    Splits argument into sliding sentence group chunks.
    window_size=1: single sentences
    window_size=2: 2-sentence groups (captures local discourse & reasoning context)
    """
    text = str(text).strip()
    # Split on sentence terminals followed by whitespace or newline
    raw_sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in raw_sents if len(s.strip().split()) >= 3]

    if not sents or len(sents) <= window_size:
        return [text]

    chunks = []
    for i in range(0, len(sents) - window_size + 1, step_size):
        chunk = " ".join(sents[i : i + window_size])
        if len(chunk.split()) >= min_chunk_words:
            chunks.append(chunk)

    last_chunk = " ".join(sents[-window_size:])
    if chunks and chunks[-1] != last_chunk:
        chunks.append(last_chunk)

    return chunks if chunks else [text]


def logsumexp_pool(scores: np.ndarray, tau: float = 2.0) -> float:
    """Computes smooth maximum via temperature-scaled LogSumExp."""
    if len(scores) == 1:
        return float(scores[0])
    return float(scipy.special.logsumexp(tau * scores) / tau)


def compute_recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10) -> Tuple[float, float, float]:
    """
    Computes recall (TPR) at a fixed FPR constraint (default: 10% FPR).
    Returns: (recall_at_target_fpr, actual_fpr, threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0, float(thresholds[0])
    best_i = valid_idx[-1]
    return float(tpr[best_i]), float(fpr[best_i]), float(thresholds[best_i])


def resolve_model_label_indices(model) -> Tuple[int, int, int]:
    """
    Dynamically identifies entailment, contradiction, and neutral label indices
    from HuggingFace SequenceClassification model config.
    """
    id2label = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
    ent_idx = next((i for i, l in id2label.items() if "entail" in l), 1)
    contra_idx = next((i for i, l in id2label.items() if "contra" in l), 0)
    neutral_idx = next((i for i, l in id2label.items() if "neutral" in l), 2)
    return ent_idx, contra_idx, neutral_idx


class ChunkedPairDataset:
    """Lightweight PyTorch dataset for pair classification."""
    def __init__(self, flat_pairs: List[Tuple[str, str]], tokenizer, max_length: int = 192):
        self.flat_pairs = flat_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.flat_pairs)

    def __getitem__(self, idx):
        premise, hypothesis = self.flat_pairs[idx]
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def run_zeroshot_chunked_evaluation(
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    test_csv: str = "data/nli_training_v3/test.csv",
    output_pred_csv: str = "results/metrics/zeroshot_chunked_1118_predictions.csv",
    output_report_json: str = "results/metrics/zeroshot_chunked_evaluation_report.json",
    window_size: int = 2,
    step_size: int = 1,
    batch_size: int = 64,
    device: Optional[str] = None
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    test_path = resolve(test_csv)
    out_csv = resolve(output_pred_csv)
    out_json = resolve(output_report_json)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test benchmark at {test_path}")

    df_test = load_csv(test_path)
    logger.info(f"Loaded {len(df_test)} test pairs from {test_path}...")

    # Determine execution device
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    logger.info(f"Loading Zero-Shot Baseline model '{model_name}' on {device_obj}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device_obj)
    model.eval()

    ent_idx, contra_idx, neutral_idx = resolve_model_label_indices(model)
    logger.info(f"Detected label indices - Entailment: {ent_idx}, Contradiction: {contra_idx}, Neutral: {neutral_idx}")

    # 1. Chunk premises
    flat_chunk_pairs: List[Tuple[str, str]] = []
    pair_chunk_ranges: List[Tuple[int, int]] = []
    pair_chunk_texts: List[List[str]] = []

    for _, row in df_test.iterrows():
        premise = str(row["premise"])
        hypo = str(row["hypothesis"])
        chunks = split_into_sentence_chunks(premise, window_size=window_size, step_size=step_size)
        pair_chunk_texts.append(chunks)
        start = len(flat_chunk_pairs)
        for c in chunks:
            flat_chunk_pairs.append((c, hypo))
        end = len(flat_chunk_pairs)
        pair_chunk_ranges.append((start, end))

    total_chunks = len(flat_chunk_pairs)
    logger.info(f"Generated {total_chunks} total chunks across {len(df_test)} pairs (avg {total_chunks/len(df_test):.1f} chunks/pair).")

    # 2. Batch inference on chunks
    logger.info(f"Scoring {total_chunks} chunks with zero-shot cross-encoder (batch_size={batch_size})...")
    chunk_dataset = ChunkedPairDataset(flat_chunk_pairs, tokenizer, max_length=192)
    chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=False)

    all_chunk_probs = []
    with torch.no_grad():
        for batch in chunk_loader:
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_chunk_probs.append(probs)

    all_chunk_probs = np.concatenate(all_chunk_probs, axis=0)  # Shape (total_chunks, 3)

    # 3. Batch inference on monolithic premises (for exact apples-to-apples comparison)
    logger.info(f"Scoring {len(df_test)} monolithic premises for fair baseline comparison...")
    mono_pairs = [(str(row["premise"]), str(row["hypothesis"])) for _, row in df_test.iterrows()]
    mono_dataset = ChunkedPairDataset(mono_pairs, tokenizer, max_length=256)
    mono_loader = DataLoader(mono_dataset, batch_size=batch_size, shuffle=False)

    mono_probs = []
    with torch.no_grad():
        for batch in mono_loader:
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            mono_probs.append(probs)

    mono_probs = np.concatenate(mono_probs, axis=0)  # Shape (1118, 3)

    # 4. Pooling Aggregations per Pair
    max_scores = []
    top2_scores = []
    mean_scores = []
    lse_scores = []
    best_chunk_idx = []
    best_chunk_text = []
    winning_chunk_probs = []

    for i, (start, end) in enumerate(pair_chunk_ranges):
        pair_probs = all_chunk_probs[start:end]  # Shape (k, 3)
        chunks = pair_chunk_texts[i]
        entail_scores = pair_probs[:, ent_idx]

        # Best chunk index by entailment score
        best_j = int(np.argmax(entail_scores))
        best_chunk_idx.append(best_j)
        best_chunk_text.append(chunks[best_j])
        winning_chunk_probs.append(pair_probs[best_j])

        # Pooling aggregations
        max_scores.append(float(entail_scores[best_j]))
        if len(entail_scores) >= 2:
            top2_scores.append(float(np.mean(np.sort(entail_scores)[-2:])))
        else:
            top2_scores.append(float(entail_scores[0]))
        mean_scores.append(float(np.mean(entail_scores)))
        lse_scores.append(logsumexp_pool(entail_scores, tau=2.0))

    winning_chunk_probs = np.array(winning_chunk_probs)  # Shape (1118, 3)
    max_scores = np.array(max_scores)
    top2_scores = np.array(top2_scores)
    mean_scores = np.array(mean_scores)
    lse_scores = np.array(lse_scores)

    mono_entail = mono_probs[:, ent_idx]
    mono_contra = mono_probs[:, contra_idx]
    mono_neutral = mono_probs[:, neutral_idx]

    winning_entail = winning_chunk_probs[:, ent_idx]
    winning_contra = winning_chunk_probs[:, contra_idx]
    winning_neutral = winning_chunk_probs[:, neutral_idx]

    # Predicted 3-class label:
    id_to_3class = {contra_idx: "contradiction", ent_idx: "entailment", neutral_idx: "neutral"}
    mono_pred_labels = [id_to_3class[i] for i in np.argmax(mono_probs, axis=1)]
    chunk_pred_labels = [id_to_3class[i] for i in np.argmax(winning_chunk_probs, axis=1)]

    # Attach predictions to dataframe
    df_test["num_chunks"] = [len(c) for c in pair_chunk_texts]
    df_test["best_chunk_idx"] = best_chunk_idx
    df_test["best_chunk_text"] = best_chunk_text

    # Monolithic zero-shot columns
    df_test["p_zeroshot_mono_contra"] = mono_contra
    df_test["p_zeroshot_mono_entail"] = mono_entail
    df_test["p_zeroshot_mono_neutral"] = mono_neutral
    df_test["zeroshot_mono_pred"] = mono_pred_labels

    # Chunked zero-shot pooled scores
    df_test["p_zeroshot_chunk_max_entail"] = max_scores
    df_test["p_zeroshot_chunk_top2_entail"] = top2_scores
    df_test["p_zeroshot_chunk_lse_entail"] = lse_scores
    df_test["p_zeroshot_chunk_mean_entail"] = mean_scores

    # Winning chunk 3-class distribution
    df_test["p_zeroshot_winning_contra"] = winning_contra
    df_test["p_zeroshot_winning_entail"] = winning_entail
    df_test["p_zeroshot_winning_neutral"] = winning_neutral
    df_test["zeroshot_chunk_pred_argmax"] = chunk_pred_labels

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_test.to_csv(out_csv, index=False)
    logger.info(f"Saved zero-shot chunked predictions to {out_csv}")

    # Masks
    is_positive = (df_test["label"].str.lower() == "entailment").to_numpy()
    is_negative = ~is_positive
    llm_pos_mask = df_test["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_test["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_test["pair_category"] == "same_topic_diff_concept").to_numpy()
    paraphrase_mask = (df_test["pair_category"] == "within_cluster_paraphrase").to_numpy()

    # -------------------------------------------------------------------------
    # PROTOCOL 1: 3-Class Argmax Evaluation (argmax == 'entailment')
    # -------------------------------------------------------------------------
    def evaluate_argmax(pred_labels: List[str]) -> Dict[str, Any]:
        accept = np.array([p == "entailment" for p in pred_labels])
        tp = int((accept & is_positive).sum())
        fp = int((accept & is_negative).sum())
        fn = int((~accept & is_positive).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / is_positive.sum() if is_positive.sum() > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        llm_m = int((accept & llm_pos_mask).sum())
        para_m = int((accept & paraphrase_mask).sum())
        pol_fp = int((accept & same_pol_mask).sum())
        top_fp = int((accept & same_top_mask).sum())
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "llm_matches": llm_m, "llm_total": int(llm_pos_mask.sum()),
            "para_matches": para_m, "para_total": int(paraphrase_mask.sum()),
            "same_policy_fp": pol_fp, "same_policy_total": int(same_pol_mask.sum()),
            "same_topic_fp": top_fp, "same_topic_total": int(same_top_mask.sum()),
        }

    mono_argmax = evaluate_argmax(mono_pred_labels)
    chunk_argmax = evaluate_argmax(chunk_pred_labels)

    # -------------------------------------------------------------------------
    # PROTOCOL 2: Threshold-Free AUROC & Recall @ 10% FPR
    # -------------------------------------------------------------------------
    pooling_methods = {
        "Monolithic Zero-Shot": mono_entail,
        "Zero-Shot Max Pooling": max_scores,
        "Zero-Shot Top-2 Pooling": top2_scores,
        "Zero-Shot LogSumExp Pooling": lse_scores,
        "Zero-Shot Mean Pooling": mean_scores
    }

    auroc_results = {}
    for name, scores in pooling_methods.items():
        # Overall AUROC
        auc_overall = roc_auc_score(is_positive.astype(int), scores)
        
        # 10% FPR evaluation
        rec_10, act_fpr, th_10 = compute_recall_at_fpr(is_positive.astype(int), scores, target_fpr=0.10)
        
        # Breakdown at th_10
        pos_pass = scores >= th_10
        llm_pass = int((pos_pass & llm_pos_mask).sum())
        para_pass = int((pos_pass & paraphrase_mask).sum())

        # Category-specific AUCs
        # 1. LLM Positives vs Same-Policy Distractors
        pol_subset = llm_pos_mask | same_pol_mask
        y_pol = is_positive[pol_subset].astype(int)
        sc_pol = scores[pol_subset]
        auc_pol = roc_auc_score(y_pol, sc_pol)

        # 2. LLM Positives vs Same-Topic Distractors
        top_subset = llm_pos_mask | same_top_mask
        y_top = is_positive[top_subset].astype(int)
        sc_top = scores[top_subset]
        auc_top = roc_auc_score(y_top, sc_top)

        auroc_results[name] = {
            "overall_auroc": round(float(auc_overall), 4),
            "recall_at_10_fpr": round(float(rec_10), 4),
            "threshold_at_10_fpr": round(float(th_10), 4),
            "llm_rec_at_10_fpr": f"{llm_pass}/{int(llm_pos_mask.sum())} ({llm_pass/int(llm_pos_mask.sum())*100:.1f}%)",
            "para_rec_at_10_fpr": f"{para_pass}/{int(paraphrase_mask.sum())} ({para_pass/int(paraphrase_mask.sum())*100:.1f}%)",
            "auc_llm_vs_same_policy": round(float(auc_pol), 4),
            "auc_llm_vs_same_topic": round(float(auc_top), 4)
        }

    # -------------------------------------------------------------------------
    # PROTOCOL 3: Operating Threshold Sweep (theta in 0.10 to 0.90)
    # -------------------------------------------------------------------------
    sweep_thetas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    sweep_rows = []
    for th in sweep_thetas:
        # Mono
        m_pass = mono_entail >= th
        m_tp = int((m_pass & is_positive).sum())
        m_fp = int((m_pass & is_negative).sum())
        m_prec = m_tp / (m_tp + m_fp) if (m_tp + m_fp) > 0 else 0.0
        m_rec = m_tp / is_positive.sum()
        m_f1 = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0.0
        m_llm = int((m_pass & llm_pos_mask).sum())

        # Chunk Max
        c_pass = max_scores >= th
        c_tp = int((c_pass & is_positive).sum())
        c_fp = int((c_pass & is_negative).sum())
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / is_positive.sum()
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        c_llm = int((c_pass & llm_pos_mask).sum())

        sweep_rows.append({
            "theta": th,
            "mono_prec": round(m_prec, 4), "mono_rec": round(m_rec, 4), "mono_f1": round(m_f1, 4), "mono_llm": f"{m_llm}/188",
            "chunk_prec": round(c_prec, 4), "chunk_rec": round(c_rec, 4), "chunk_f1": round(c_f1, 4), "chunk_llm": f"{c_llm}/188"
        })

    # Assemble report
    report = {
        "model_name": model_name,
        "total_test_pairs": len(df_test),
        "window_size": window_size,
        "step_size": step_size,
        "protocol_1_argmax": {
            "monolithic": mono_argmax,
            "chunked": chunk_argmax
        },
        "protocol_2_auroc_and_recall_at_10_fpr": auroc_results,
        "protocol_3_threshold_sweep": sweep_rows
    }

    save_json(report, out_json)
    logger.info(f"Saved full zero-shot evaluation report to {out_json}")

    # -------------------------------------------------------------------------
    # PRINT FORMATTED TABLES TO CONSOLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"ZERO-SHOT BASELINE EVALUATION ({model_name})")
    print("=" * 95)

    print("\n" + "-" * 95)
    print("PROTOCOL 1: 3-CLASS STRICT ARGMAX EVALUATION (argmax == 'entailment')")
    print("-" * 95)
    print(f"{'Metric':<32} | {'Monolithic Zero-Shot':<22} | {'Chunked Zero-Shot':<20} | {'Delta'}")
    print("-" * 95)
    print(f"{'LLM-to-Human Matches (/188)':<32} | {mono_argmax['llm_matches']:>5} ({mono_argmax['llm_matches']/188*100:>5.2f}%)         | {chunk_argmax['llm_matches']:>5} ({chunk_argmax['llm_matches']/188*100:>5.2f}%)       | {chunk_argmax['llm_matches'] - mono_argmax['llm_matches']:+d}")
    print(f"{'Paraphrase Matches (/352)':<32} | {mono_argmax['para_matches']:>5} ({mono_argmax['para_matches']/352*100:>5.2f}%)         | {chunk_argmax['para_matches']:>5} ({chunk_argmax['para_matches']/352*100:>5.2f}%)       | {chunk_argmax['para_matches'] - mono_argmax['para_matches']:+d}")
    print(f"{'Total True Positives (/540)':<32} | {mono_argmax['tp']:>5} ({mono_argmax['tp']/540*100:>5.2f}%)         | {chunk_argmax['tp']:>5} ({chunk_argmax['tp']/540*100:>5.2f}%)       | {chunk_argmax['tp'] - mono_argmax['tp']:+d}")
    print(f"{'Total False Positives (/578)':<32} | {mono_argmax['fp']:>5}                       | {chunk_argmax['fp']:>5}                     | {chunk_argmax['fp'] - mono_argmax['fp']:+d}")
    print(f"{'  - Same-Policy FP (/240)':<32} | {mono_argmax['same_policy_fp']:>5} ({mono_argmax['same_policy_fp']/240*100:>5.2f}%)         | {chunk_argmax['same_policy_fp']:>5} ({chunk_argmax['same_policy_fp']/240*100:>5.2f}%)       | {chunk_argmax['same_policy_fp'] - mono_argmax['same_policy_fp']:+d}")
    print(f"{'  - Same-Topic FP (/180)':<32} | {mono_argmax['same_topic_fp']:>5} ({mono_argmax['same_topic_fp']/180*100:>5.2f}%)         | {chunk_argmax['same_topic_fp']:>5} ({chunk_argmax['same_topic_fp']/180*100:>5.2f}%)       | {chunk_argmax['same_topic_fp'] - mono_argmax['same_topic_fp']:+d}")
    print(f"{'Overall Binary Precision':<32} | {mono_argmax['precision']*100:>6.2f}%                 | {chunk_argmax['precision']*100:>6.2f}%               | {(chunk_argmax['precision'] - mono_argmax['precision'])*100:+6.2f}%")
    print(f"{'Overall Binary Recall':<32} | {mono_argmax['recall']*100:>6.2f}%                 | {chunk_argmax['recall']*100:>6.2f}%               | {(chunk_argmax['recall'] - mono_argmax['recall'])*100:+6.2f}%")
    print(f"{'Binary F1 Score':<32} | {mono_argmax['f1']:>8.4f}                 | {chunk_argmax['f1']:>8.4f}               | {chunk_argmax['f1'] - mono_argmax['f1']:+8.4f}")

    print("\n" + "-" * 115)
    print("PROTOCOL 2: THRESHOLD-FREE AUROC & RECALL @ 10% FPR")
    print("-" * 115)
    print(f"{'Method':<28} | {'AUROC':<7} | {'Rec @ 10%':<10} | {'LLM Rec @ 10%':<18} | {'AUC (LLM vs Policy)':<19} | {'AUC (LLM vs Topic)'}")
    print("-" * 115)
    for name, res in auroc_results.items():
        print(f"{name:<28} | {res['overall_auroc']:>7.4f} | {res['recall_at_10_fpr']*100:>8.2f}%  | {res['llm_rec_at_10_fpr']:<18} | {res['auc_llm_vs_same_policy']:>19.4f} | {res['auc_llm_vs_same_topic']:>18.4f}")

    print("\n" + "-" * 105)
    print("PROTOCOL 3: IDENTICAL P(entailment) >= theta SWEEP")
    print("-" * 105)
    print(f"{'Theta':<7} | {'Mono Prec':<10} {'Mono Rec':<10} {'Mono F1':<9} {'Mono LLM':<10} | {'Chunk Prec':<10} {'Chunk Rec':<10} {'Chunk F1':<9} {'Chunk LLM'}")
    print("-" * 105)
    for row in sweep_rows:
        print(f"{row['theta']:<7.2f} | {row['mono_prec']*100:>8.2f}% {row['mono_rec']*100:>8.2f}% {row['mono_f1']:>8.4f}  {row['mono_llm']:<10} | {row['chunk_prec']*100:>8.2f}% {row['chunk_rec']*100:>8.2f}% {row['chunk_f1']:>8.4f}  {row['chunk_llm']}")
    print("=" * 105 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot Baseline as Chunk Scorer")
    parser.add_argument("--model_name", type=str, default="cross-encoder/nli-deberta-v3-base",
                        help="HuggingFace cross-encoder model name or path")
    parser.add_argument("--test_csv", type=str, default="data/nli_training_v3/test.csv",
                        help="Held-out 1,118 test benchmark CSV")
    parser.add_argument("--output_csv", type=str, default="results/metrics/zeroshot_chunked_1118_predictions.csv",
                        help="Output path for predictions CSV")
    parser.add_argument("--output_report", type=str, default="results/metrics/zeroshot_chunked_evaluation_report.json",
                        help="Output path for JSON summary report")
    parser.add_argument("--window_size", type=int, default=2,
                        help="Sentence group window size (default: 2)")
    parser.add_argument("--step_size", type=int, default=1,
                        help="Sliding step size across sentences (default: 1)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for cross-encoder inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cuda/cpu)")

    args = parser.parse_args()

    run_zeroshot_chunked_evaluation(
        model_name=args.model_name,
        test_csv=args.test_csv,
        output_pred_csv=args.output_csv,
        output_report_json=args.output_report,
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        device=args.device
    )

