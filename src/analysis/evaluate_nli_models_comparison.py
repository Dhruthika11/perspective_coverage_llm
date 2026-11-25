"""
Evaluation and Comparison of Argument-NLI Model Versions (V1, V2, V3).

Implements the user's experimental protocol:
  Version V1 (Baseline): Existing fine-tuned DeBERTa (inverted direction, random negatives)
  Version V2 (Direction-Corrected): Premise = Generated Argument, Hypothesis = Human Reference
  Version V3 (Coverage-Aware): Direction-Corrected + Length-Stratified Positives + Same-Stance Hard Negatives

Evaluates on the locked, claim-disjoint test partition across:
  1. 3-Class NLI Classification (Accuracy, Macro-F1, Class-wise Precision/Recall)
  2. End-to-End Perspective Matching Pipeline (tau >= 0.75, p_entail >= theta*)
  3. Recall across Length Tiers: Short (20-50), Medium (50-100), Long (100-175), Very Long (175-250)
  4. Negative Rejection Accuracy: Same-Stance Different-Reasoning and Opposite-Stance Conflicts
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json, save_csv
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_nli_models_comparison")

LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}
ID_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


def evaluate_single_nli_model(
    model_path_or_name: str,
    test_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
    input_direction: str = "forward"  # 'forward': P=premise, H=hypothesis; 'reverse': P=hypothesis, H=premise
) -> Dict[str, Any]:
    logger.info(f"Evaluating model: {model_path_or_name} (Direction: {input_direction})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name).to(device)
    model.eval()

    premises = test_df["premise"].fillna("").astype(str).tolist()
    hypotheses = test_df["hypothesis"].fillna("").astype(str).tolist()

    if input_direction == "reverse":
        # Invert inputs to test legacy V1 orientation
        premises, hypotheses = hypotheses, premises

    all_probs = []
    all_preds = []

    for i in range(0, len(premises), batch_size):
        p_batch = premises[i : i + batch_size]
        h_batch = hypotheses[i : i + batch_size]

        inputs = tokenizer(
            p_batch,
            h_batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)

    probs_arr = np.array(all_probs)
    preds_arr = np.array(all_preds)

    # In standard NLI head: index 1 is entailment, 0 is contradiction, 2 is neutral
    # Check id2label to ensure correct column alignment
    id2label = getattr(model.config, "id2label", ID_MAP)
    ent_idx = 1
    for k, v in id2label.items():
        if "entail" in str(v).lower():
            ent_idx = int(k)
            break

    y_true = np.array([LABEL_MAP.get(str(lbl).lower(), 2) for lbl in test_df["label"]])

    # Metrics
    acc = accuracy_score(y_true, preds_arr)
    macro_f1 = f1_score(y_true, preds_arr, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, preds_arr, labels=[0, 1, 2], zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds_arr, labels=[0, 1, 2])

    # End-to-end binary matching performance (Label 1 = Entailment, vs 0/2 = Non-Match)
    # Using production rule: p_entail >= 0.80
    binary_true = (y_true == 1).astype(int)
    binary_pred = (probs_arr[:, ent_idx] >= 0.80).astype(int)

    tp = int(np.sum((binary_pred == 1) & (binary_true == 1)))
    fp = int(np.sum((binary_pred == 1) & (binary_true == 0)))
    fn = int(np.sum((binary_pred == 0) & (binary_true == 1)))
    tn = int(np.sum((binary_pred == 0) & (binary_true == 0)))

    pipeline_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pipeline_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pipeline_f1 = 2 * pipeline_prec * pipeline_rec / (pipeline_prec + pipeline_rec) if (pipeline_prec + pipeline_rec) > 0 else 0.0

    # Recall across length tiers among true positives
    length_breakdown = {}
    if "length_tier" in test_df.columns:
        test_df_copy = test_df.copy()
        test_df_copy["p_entail"] = probs_arr[:, ent_idx]
        test_df_copy["pred_match"] = binary_pred

        pos_subset = test_df_copy[test_df_copy["label"] == "entailment"]
        for tier in ["short", "medium", "long", "very_long"]:
            tier_df = pos_subset[pos_subset["length_tier"] == tier]
            if not tier_df.empty:
                tier_rec = tier_df["pred_match"].mean()
                length_breakdown[tier] = {
                    "count": len(tier_df),
                    "accepted": int(tier_df["pred_match"].sum()),
                    "recall": round(float(tier_rec), 4)
                }

    # Rejection of same-stance different reasoning
    same_stance_df = test_df[test_df["pair_category"] == "same_stance_different_reasoning"]
    same_stance_rejection = 1.0 - (probs_arr[same_stance_df.index, ent_idx] >= 0.80).mean() if not same_stance_df.empty else 1.0

    # Rejection of opposite stance conflicts
    opp_df = test_df[test_df["pair_category"] == "opposite_stance_conflict"]
    opp_rejection = 1.0 - (probs_arr[opp_df.index, ent_idx] >= 0.80).mean() if not opp_df.empty else 1.0

    return {
        "model_path": model_path_or_name,
        "input_direction": input_direction,
        "classification_accuracy": round(float(acc), 4),
        "classification_macro_f1": round(float(macro_f1), 4),
        "class_metrics": {
            "contradiction": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4), "f1": round(float(f1[0]), 4)},
            "entailment": {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4), "f1": round(float(f1[1]), 4)},
            "neutral": {"precision": round(float(prec[2]), 4), "recall": round(float(rec[2]), 4), "f1": round(float(f1[2]), 4)}
        },
        "pipeline_matching_metrics_p80": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(float(pipeline_prec), 4),
            "recall": round(float(pipeline_rec), 4),
            "f1_score": round(float(pipeline_f1), 4)
        },
        "length_tier_recall_entailment": length_breakdown,
        "hard_negative_rejection": {
            "same_stance_different_reasoning_rejection": round(float(same_stance_rejection), 4),
            "opposite_stance_conflict_rejection": round(float(opp_rejection), 4)
        }
    }


def run_models_comparison(
    v3_test_csv: str = "data/nli_training_v3/test.csv",
    v1_model_path: str = "models/argument_nli_deberta_v3",
    v2_model_path: str = "models/argument_nli_deberta_v3_v2",
    v3_model_path: str = "models/argument_nli_deberta_v3_v3",
    base_pretrained: str = "cross-encoder/nli-deberta-v3-small",
    output_dir: str = "results/metrics",
    tables_dir: str = "results/paper_tables"
) -> Dict[str, Any]:
    test_p = resolve(v3_test_csv)
    if not os.path.exists(test_p):
        raise FileNotFoundError(f"Missing locked test set at {test_p}")

    test_df = load_csv(test_p)
    logger.info(f"Loaded locked independent test set: {len(test_df)} pairs from {test_p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on device: {device}")

    models_to_evaluate = []

    # 1. Version 1 (Baseline)
    if os.path.exists(resolve(v1_model_path)):
        models_to_evaluate.append(("V1 — Existing Baseline", resolve(v1_model_path), "reverse"))
    else:
        # Fall back to base pretrained model with reverse input to test baseline behavior
        models_to_evaluate.append(("V1 — Base Pretrained (Legacy Inverted)", base_pretrained, "reverse"))

    # 2. Version 2 (Direction-Corrected)
    if os.path.exists(resolve(v2_model_path)):
        models_to_evaluate.append(("V2 — Direction-Corrected", resolve(v2_model_path), "forward"))
    else:
        models_to_evaluate.append(("V2 — Direction-Corrected (Zero-Shot)", base_pretrained, "forward"))

    # 3. Version 3 (Coverage-Aware)
    if os.path.exists(resolve(v3_model_path)):
        models_to_evaluate.append(("V3 — Coverage-Aware", resolve(v3_model_path), "forward"))

    results = []
    comparison_rows = []

    for name, m_path, direction in models_to_evaluate:
        res = evaluate_single_nli_model(
            model_path_or_name=m_path,
            test_df=test_df,
            device=device,
            input_direction=direction
        )
        res["version_name"] = name
        results.append(res)

        p_stats = res["pipeline_matching_metrics_p80"]
        c_stats = res["class_metrics"]
        neg_stats = res["hard_negative_rejection"]

        comparison_rows.append({
            "Version": name,
            "Macro_F1": res["classification_macro_f1"],
            "Accuracy": res["classification_accuracy"],
            "Pipeline_Precision": p_stats["precision"],
            "Pipeline_Recall": p_stats["recall"],
            "Pipeline_F1": p_stats["f1_score"],
            "Entailment_F1": c_stats["entailment"]["f1"],
            "Same_Stance_Diff_Reason_Rejection": neg_stats["same_stance_different_reasoning_rejection"],
            "Opposite_Stance_Rejection": neg_stats["opposite_stance_conflict_rejection"]
        })

    comp_df = pd.DataFrame(comparison_rows)

    # Save reports
    os.makedirs(resolve(output_dir), exist_ok=True)
    os.makedirs(resolve(tables_dir), exist_ok=True)

    json_out = resolve(os.path.join(output_dir, "nli_v1_v2_v3_comparison_report.json"))
    csv_out = resolve(os.path.join(tables_dir, "table_nli_v1_v2_v3_comparison.csv"))

    final_report = {
        "evaluation_dataset": v3_test_csv,
        "n_test_pairs": len(test_df),
        "comparison_table": comparison_rows,
        "detailed_results": results
    }

    save_json(final_report, json_out)
    save_csv(comp_df, csv_out)
    logger.info(f"Comparison report saved to {json_out} and {csv_out}")

    print("\n" + "=" * 105)
    print("      ARGUMENT-NLI MODEL COMPARISON: V1 (BASELINE) vs V2 (DIRECTION) vs V3 (COVERAGE-AWARE)")
    print("=" * 105)
    print(comp_df.to_string(index=False))
    print("=" * 105 + "\n")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare Argument-NLI model versions (V1, V2, V3).")
    parser.add_argument("--test_csv", type=str, default="data/nli_training_v3/test.csv", help="Path to locked test CSV.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max test samples for quick evaluation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()

    test_p = resolve(args.test_csv)
    test_df = load_csv(test_p)
    if args.max_samples is not None and args.max_samples < len(test_df):
        logger.info(f"Subsampling test set from {len(test_df)} to {args.max_samples} for rapid evaluation.")
        test_df = test_df.head(args.max_samples)
        # Temporarily save or pass df
        temp_test_csv = "data/nli_training_v3/temp_test_eval.csv"
        save_csv(test_df, resolve(temp_test_csv))
        run_models_comparison(v3_test_csv=temp_test_csv)
        if os.path.exists(resolve(temp_test_csv)):
            os.remove(resolve(temp_test_csv))
    else:
        run_models_comparison(v3_test_csv=args.test_csv)

