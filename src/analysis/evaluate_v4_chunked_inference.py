"""
Argument-NLI V4 Chunked Inference (Max-Pooling over Sentence Groups).

Implements the Chunker Experiment:
  - Long argument premise (up to 250 words) is split into sentence groups (chunks).
  - Each chunk is paired with the concise human perspective hypothesis.
  - Takes the maximum entailment/coverage score across all chunks:
      P*(full_coverage) = max_j P(chunk_j -> hypothesis)
  - Identifies the specific 'winning chunk' that entailed the perspective.
  - Compares Monolithic V4 vs. Chunked V4 across the 1,118 test pairs.
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_v4_chunked_inference")

ID_TO_LABEL_V4 = {
    0: "contradiction",
    1: "full_coverage",
    2: "no_coverage",
    3: "partial_coverage"
}


def split_into_sentence_chunks(
    text: str,
    window_size: int = 2,
    step_size: int = 1,
    min_chunk_words: int = 8
) -> List[str]:
    """
    Splits argument into sliding sentence group chunks.
    window_size=1: single sentences
    window_size=2: 2-sentence groups (captures local discourse & context)
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


class ChunkedPairDataset(Dataset):
    def __init__(self, flat_pairs: List[Tuple[str, str]], tokenizer, max_length: int = 192):
        self.flat_pairs = flat_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.flat_pairs)

    def __getitem__(self, idx):
        premise_chunk, hypothesis = self.flat_pairs[idx]
        encoding = self.tokenizer(
            premise_chunk,
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


def run_chunked_evaluation(
    model_dir: str = "models/argument_nli_deberta_v3_v4",
    test_csv: str = "data/nli_training_v3/test.csv",
    monolithic_pred_csv: str = "results/metrics/v4_1118_predictions.csv",
    output_pred_csv: str = "results/metrics/v4_chunked_1118_predictions.csv",
    output_report_json: str = "results/metrics/v4_chunked_evaluation_report.json",
    window_size: int = 2,
    batch_size: int = 64
) -> Dict[str, Any]:
    model_path = resolve(model_dir)
    test_path = resolve(test_csv)
    mono_path = resolve(monolithic_pred_csv)
    out_csv = resolve(output_pred_csv)
    out_json = resolve(output_report_json)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"V4 model not found at {model_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test CSV not found at {test_path}")

    logger.info(f"Loading test set from {test_path}...")
    df_test = load_csv(test_path)
    logger.info(f"Loaded {len(df_test)} test pairs. Chunking premises with window_size={window_size}...")

    # Build flat chunks list with tracking indices
    flat_chunk_pairs: List[Tuple[str, str]] = []
    pair_chunk_ranges: List[Tuple[int, int]] = []  # (start_idx, end_idx) in flat_chunk_pairs
    pair_chunk_texts: List[List[str]] = []

    for idx, row in df_test.iterrows():
        premise = str(row["premise"])
        hypo = str(row["hypothesis"])
        chunks = split_into_sentence_chunks(premise, window_size=window_size)
        pair_chunk_texts.append(chunks)

        start = len(flat_chunk_pairs)
        for c in chunks:
            flat_chunk_pairs.append((c, hypo))
        end = len(flat_chunk_pairs)
        pair_chunk_ranges.append((start, end))

    logger.info(f"Generated {len(flat_chunk_pairs)} total chunks across {len(df_test)} pairs (avg {len(flat_chunk_pairs)/len(df_test):.1f} chunks/pair).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on device: {device} | batch_size: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    chunk_dataset = ChunkedPairDataset(flat_chunk_pairs, tokenizer, max_length=192)
    chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=False)

    all_chunk_probs = []
    with torch.no_grad():
        for batch in chunk_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_chunk_probs.append(probs)

    all_chunk_probs = np.concatenate(all_chunk_probs, axis=0)  # Shape (N_chunks, 4)

    # Max-pooling aggregation per pair
    max_full_cov = []
    max_comb_cov = []
    best_chunk_idx = []
    best_chunk_text = []
    num_chunks = []
    pooled_4class_probs = []

    for i, (start, end) in enumerate(pair_chunk_ranges):
        pair_probs = all_chunk_probs[start:end]  # Shape (k, 4)
        chunks = pair_chunk_texts[i]
        k = len(chunks)
        num_chunks.append(k)

        # Full coverage score across chunks
        full_scores = pair_probs[:, 1]
        comb_scores = pair_probs[:, 1] + 0.5 * pair_probs[:, 3]

        # Best chunk index by full coverage score
        best_j = int(np.argmax(full_scores))
        best_chunk_idx.append(best_j)
        best_chunk_text.append(chunks[best_j])

        max_full_cov.append(float(full_scores[best_j]))
        max_comb_cov.append(float(np.max(comb_scores)))
        pooled_4class_probs.append(pair_probs[best_j])

    pooled_4class_probs = np.array(pooled_4class_probs)  # Shape (1118, 4)

    # Attach results to dataframe
    df_test["num_chunks"] = num_chunks
    df_test["best_chunk_idx"] = best_chunk_idx
    df_test["best_chunk_text"] = best_chunk_text

    df_test["p_v4_chunk_contra"] = pooled_4class_probs[:, 0]
    df_test["p_v4_chunk_full_cov"] = max_full_cov
    df_test["p_v4_chunk_no_cov"] = pooled_4class_probs[:, 2]
    df_test["p_v4_chunk_partial_cov"] = pooled_4class_probs[:, 3]
    df_test["p_v4_chunk_combined_cov"] = max_comb_cov

    argmax_ids = np.argmax(pooled_4class_probs, axis=1)
    df_test["v4_chunk_pred_label_4class"] = [ID_TO_LABEL_V4[i] for i in argmax_ids]

    # Ground truth mapping
    is_positive = (df_test["label"].str.lower() == "entailment").to_numpy()
    is_negative = ~is_positive
    llm_pos_mask = df_test["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_test["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_test["pair_category"] == "same_topic_diff_concept").to_numpy()
    paraphrase_mask = (df_test["pair_category"] == "within_cluster_paraphrase").to_numpy()

    # Load monolithic predictions for comparison if available
    has_mono = os.path.exists(mono_path)
    if has_mono:
        df_mono = load_csv(mono_path)
        df_test["v4_mono_pred_label"] = df_mono.get("v4_pred_label_4class", "")
        df_test["p_v4_mono_full_cov"] = df_mono.get("p_v4_full_cov", 0.0)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_test.to_csv(out_csv, index=False)
    logger.info(f"Saved chunked test predictions to {out_csv}")

    # Compute comparison metrics
    # Strict Argmax Policy (full_coverage)
    strict_accept = (argmax_ids == 1)
    tp_s = int((strict_accept & is_positive).sum())
    fp_s = int((strict_accept & is_negative).sum())
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    rec_s = tp_s / is_positive.sum()
    f1_s = (2 * prec_s * rec_s) / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0

    llm_matches = int((strict_accept & llm_pos_mask).sum())
    same_pol_fps = int((strict_accept & same_pol_mask).sum())
    same_top_fps = int((strict_accept & same_top_mask).sum())
    paraphrase_matches = int((strict_accept & paraphrase_mask).sum())

    # Crosstab category by predicted label
    ct = pd.crosstab(df_test["pair_category"], df_test["v4_chunk_pred_label_4class"], margins=True)

    print("\n" + "="*85)
    print("V4 CHUNKED INFERENCE (WINDOW SIZE = {}) - 4-CLASS ARGMAX DISTRIBUTION".format(window_size))
    print("="*85)
    print(df_test["v4_chunk_pred_label_4class"].value_counts())

    print("\n" + "="*85)
    print("V4 CHUNKED PREDICTIONS ACROSS GROUND TRUTH TEST CATEGORIES")
    print("="*85)
    print(ct)

    print("\n" + "="*85)
    print("CHUNKED VS. MONOLITHIC HEAD-TO-HEAD COMPARISON")
    print("="*85)
    print(f"STRICT FULL_COVERAGE METRICS:")
    print(f"  * Overall Precision : {prec_s*100:.2f}%")
    print(f"  * Overall Recall    : {rec_s*100:.2f}% (TP={tp_s}/{is_positive.sum()})")
    print(f"  * Binary F1         : {f1_s:.4f}")
    print(f"  * Total False Pos.  : {fp_s}/{is_negative.sum()}")
    print(f"  * LLM-to-Human Rec. : {llm_matches}/188 ({llm_matches/188.0*100:.2f}%)")
    print(f"  * Same-Policy FP    : {same_pol_fps}/240 ({same_pol_fps/240.0*100:.2f}%)")
    print(f"  * Same-Topic FP     : {same_top_fps}/180 ({same_top_fps/180.0*100:.2f}%)")
    print(f"  * Paraphrase Rec.   : {paraphrase_matches}/352 ({paraphrase_matches/352.0*100:.2f}%)")

    # Probability sweep
    print("\n" + "="*85)
    print("CHUNKED PROBABILITY THRESHOLD SWEEP (P*(full_coverage) >= theta)")
    print("="*85)
    sweep = {}
    for theta in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        acc = (np.array(max_full_cov) >= theta)
        cur_tp = int((acc & is_positive).sum())
        cur_fp = int((acc & is_negative).sum())
        cur_prec = cur_tp / (cur_tp + cur_fp) if (cur_tp + cur_fp) > 0 else 0.0
        cur_rec = cur_tp / is_positive.sum()
        cur_f1 = (2 * cur_prec * cur_rec) / (cur_prec + cur_rec) if (cur_prec + cur_rec) > 0 else 0.0
        cur_llm = int((acc & llm_pos_mask).sum())
        cur_sp_fp = int((acc & same_pol_mask).sum())
        sweep[f"theta_{theta:.2f}"] = {
            "precision": round(cur_prec, 4),
            "recall": round(cur_rec, 4),
            "f1": round(cur_f1, 4),
            "tp": cur_tp,
            "fp": cur_fp,
            "llm_188_matches": cur_llm,
            "same_policy_fp": cur_sp_fp
        }
        print(f"[THETA {theta:.2f}] Prec: {cur_prec*100:.2f}% | Rec: {cur_rec*100:.2f}% | F1: {cur_f1:.4f} | Total FP: {cur_fp:3d} (Same-Pol: {cur_sp_fp:3d}) | LLM Matches: {cur_llm:3d}/188 ({cur_llm/188.0*100:.1f}%)")

    # Sample qualitative showcase: show 3 LLM matches where chunking found the specific sentence
    print("\n" + "="*85)
    print("SAMPLE CHUNKED LOCALIZATION EXAMPLES (LLM Arguments)")
    print("="*85)
    llm_df = df_test[llm_pos_mask & strict_accept].head(3)
    for idx, r in llm_df.iterrows():
        print(f"[Pair {idx}] Target Hypothesis : {r['hypothesis']}")
        print(f"          Premise Length    : {len(str(r['premise']).split())} words (split into {r['num_chunks']} chunks)")
        print(f"          Winning Chunk #{r['best_chunk_idx']}: \"{r['best_chunk_text'][:120]}...\" (P_full={r['p_v4_chunk_full_cov']:.4f})\n")

    report = {
        "experiment": "V4 Chunked Max-Pooling Inference",
        "window_size": window_size,
        "total_pairs": len(df_test),
        "total_chunks": len(flat_chunk_pairs),
        "strict_argmax_metrics": {
            "precision": round(prec_s, 4),
            "recall": round(rec_s, 4),
            "f1": round(f1_s, 4),
            "tp": tp_s,
            "fp": fp_s,
            "llm_188_matches": llm_matches,
            "same_policy_fp": same_pol_fps,
            "same_topic_fp": same_top_fps,
            "paraphrase_matches": paraphrase_matches
        },
        "threshold_sweep": sweep,
        "crosstab": ct.to_dict()
    }
    save_json(report, out_json)
    logger.info(f"Saved chunked report to {out_json}")
    print("="*85 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V4 with Chunked Max-Pooling")
    parser.add_argument("--window_size", type=int, default=2, help="Number of sentences per chunk")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    run_chunked_evaluation(window_size=args.window_size, batch_size=args.batch_size)

