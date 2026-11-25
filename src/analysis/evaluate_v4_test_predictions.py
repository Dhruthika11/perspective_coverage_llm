"""
Enhanced Comprehensive Evaluation of Argument-NLI V4 on Locked 1,118-Pair Test Set.

Accounts for 4-Class Softmax Dynamics:
  - 4-Class Argmax Predictions (full_coverage, partial_coverage, no_coverage, contradiction)
  - Strict Binary Coverage Policy (full_coverage argmax)
  - Qualified Binary Coverage Policy (full_coverage + partial_coverage)
  - Calibrated Probability Thresholds for 4-class models (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
  - Cross-tabulation by ground truth test category (Paraphrases, LLM Positives, Same-Policy FPs, Same-Topic FPs)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_v4_test_predictions")

ID_TO_LABEL_V4 = {
    0: "contradiction",
    1: "full_coverage",
    2: "no_coverage",
    3: "partial_coverage"
}


class TestEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premises = df["premise"].fillna("").astype(str).tolist()
        self.hypotheses = df["hypothesis"].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        p = self.premises[idx]
        h = self.hypotheses[idx]
        encoding = self.tokenizer(
            p,
            h,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def run_v4_test_evaluation(
    model_dir: str = "models/argument_nli_deberta_v3_v4",
    test_csv: str = "data/nli_training_v3/test.csv",
    output_pred_csv: str = "results/metrics/v4_1118_predictions.csv",
    output_report_json: str = "results/metrics/v4_test_evaluation_report.json",
    batch_size: int = 32
):
    model_path = resolve(model_dir)
    test_path = resolve(test_csv)
    out_csv = resolve(output_pred_csv)
    out_json = resolve(output_report_json)

    df_test = load_csv(test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running V4 evaluation on device: {device} ({len(df_test)} pairs)")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = TestEvalDataset(df_test, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0)

    # 4-class probabilities
    df_test["p_v4_contra"] = probs_arr[:, 0]
    df_test["p_v4_full_cov"] = probs_arr[:, 1]
    df_test["p_v4_no_cov"] = probs_arr[:, 2]
    df_test["p_v4_partial_cov"] = probs_arr[:, 3]

    argmax_ids = np.argmax(probs_arr, axis=1)
    df_test["v4_pred_label_4class"] = [ID_TO_LABEL_V4[i] for i in argmax_ids]

    # Ground truth mapping
    is_positive = (df_test["label"].str.lower() == "entailment").to_numpy()
    is_negative = ~is_positive
    llm_pos_mask = df_test["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_test["pair_category"] == "same_stance_different_reasoning").to_numpy()
    same_top_mask = (df_test["pair_category"] == "same_topic_diff_concept").to_numpy()

    # Link master predictions if available
    master_p = resolve("results/metrics/master_1118_predictions.csv")
    if os.path.exists(master_p):
        df_master = load_csv(master_p)
        df_test["p_forward_base"] = df_master["p_forward_base_entail"]
        df_test["p_v3"] = df_master["p_v3_entail"]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_test.to_csv(out_csv, index=False)

    # 1. 4-Class Argmax Performance Breakdown
    print("\n" + "="*85)
    print("V4 4-CLASS ARGMAX PREDICTION DISTRIBUTION")
    print("="*85)
    print(df_test["v4_pred_label_4class"].value_counts())

    print("\n" + "="*85)
    print("V4 PREDICTIONS ACROSS GROUND TRUTH TEST CATEGORIES")
    print("="*85)
    ct = pd.crosstab(df_test["pair_category"], df_test["v4_pred_label_4class"], margins=True)
    print(ct)

    # 2. Strict Policy (Argmax == full_coverage)
    strict_accept = (argmax_ids == 1)
    tp_s = int((strict_accept & is_positive).sum())
    fp_s = int((strict_accept & is_negative).sum())
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    rec_s = tp_s / is_positive.sum()
    f1_s = (2 * prec_s * rec_s) / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0
    llm_rec_s = int((strict_accept & llm_pos_mask).sum()) / llm_pos_mask.sum()
    same_pol_fp_s = int((strict_accept & same_pol_mask).sum())

    # 3. Qualified Policy (Argmax in {full_coverage, partial_coverage})
    qual_accept = (argmax_ids == 1) | (argmax_ids == 3)
    tp_q = int((qual_accept & is_positive).sum())
    fp_q = int((qual_accept & is_negative).sum())
    prec_q = tp_q / (tp_q + fp_q) if (tp_q + fp_q) > 0 else 0.0
    rec_q = tp_q / is_positive.sum()
    f1_q = (2 * prec_q * rec_q) / (prec_q + rec_q) if (prec_q + rec_q) > 0 else 0.0
    llm_rec_q = int((qual_accept & llm_pos_mask).sum()) / llm_pos_mask.sum()
    same_pol_fp_q = int((qual_accept & same_pol_mask).sum())

    print("\n" + "="*85)
    print("ARGMAX OPERATING POLICIES")
    print("="*85)
    print(f"STRICT POLICY (full_coverage): Prec: {prec_s*100:.2f}% | Rec: {rec_s*100:.2f}% | F1: {f1_s:.4f} | Total FP: {fp_s} (Same-Policy FP: {same_pol_fp_s}) | LLM Recall: {llm_rec_s*100:.2f}% ({int((strict_accept & llm_pos_mask).sum())}/188)")
    print(f"QUALIFIED POLICY (full + part): Prec: {prec_q*100:.2f}% | Rec: {rec_q*100:.2f}% | F1: {f1_q:.4f} | Total FP: {fp_q} (Same-Policy FP: {same_pol_fp_q}) | LLM Recall: {llm_rec_q*100:.2f}% ({int((qual_accept & llm_pos_mask).sum())}/188)")

    # 4. Calibrated Probability Thresholds for 4-Class Model
    # Combined coverage score: P(full) + 0.5 * P(partial)
    comb_score = probs_arr[:, 1] + 0.5 * probs_arr[:, 3]
    df_test["p_v4_combined_coverage"] = comb_score

    print("\n" + "="*85)
    print("CALIBRATED 4-CLASS PROBABILITY SWEEP (P(full) + 0.5 * P(partial))")
    print("="*85)
    sweep_results = {}
    for theta in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        accept = (comb_score >= theta)
        tp = int((accept & is_positive).sum())
        fp = int((accept & is_negative).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / is_positive.sum()
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        llm_m = int((accept & llm_pos_mask).sum())
        same_p_fp = int((accept & same_pol_mask).sum())

        sweep_results[f"theta_{theta:.2f}"] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "same_policy_fp": same_p_fp,
            "llm_188_matches": llm_m,
            "llm_188_recall": round(llm_m / 188.0, 4)
        }
        print(f"[THETA {theta:.2f}] Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1:.4f} | Total FP: {fp:3d} (Same-Pol FP: {same_p_fp:3d}) | LLM Matches: {llm_m:3d}/188 ({llm_m/188.0*100:.1f}%)")

    report = {
        "strict_policy": {"precision": prec_s, "recall": rec_s, "f1": f1_s, "fp": fp_s, "same_policy_fp": same_pol_fp_s, "llm_matches": int((strict_accept & llm_pos_mask).sum())},
        "qualified_policy": {"precision": prec_q, "recall": rec_q, "f1": f1_q, "fp": fp_q, "same_policy_fp": same_pol_fp_q, "llm_matches": int((qual_accept & llm_pos_mask).sum())},
        "calibrated_sweep": sweep_results,
        "crosstab_category_by_pred": ct.to_dict()
    }
    save_json(report, out_json)
    print("="*85 + "\n")
    return report


if __name__ == "__main__":
    run_v4_test_evaluation()
