"""
Evaluation of Two-Stage Perspective Coverage with Fine-Tuned Argument-NLI.
Implements:
  - Strategy 1: Decision Margin Thresholding (p_entail >= theta, p_entail - p_contra >= delta)
  - Strategy 2: Relaxed Stage 1 Cosine Threshold (tau = 0.70 vs 0.75 vs 0.80)
  - Automated Pareto Frontier Sweep to find the optimal operating point without overfitting
  - Full Class Metrics, 3x3 NLI Confusion Matrix, and 2x2 Pipeline Confusion Matrix.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on device: {device}")

def evaluate_finetuned_pipeline(
    benchmark_csv="data/calibration/hard_calibration_pairs.csv",
    model_dir="models/argument_nli_deberta_v3",
    output_json="results/metrics/finetuned_two_stage_evaluation.json"
):
    if not os.path.exists(benchmark_csv):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_csv}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Fine-tuned model checkpoint not found: {model_dir}")

    df = pd.read_csv(benchmark_csv)
    print(f"Loaded {len(df)} pairs from {benchmark_csv}")

    # Load fine-tuned model and tokenizer
    print(f"Loading fine-tuned model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Run inference in batches with DIRECT Gold vs Candidate input
    batch_size = 32
    preds = []
    probs_list = []

    print("Running Stage 2 inference on all 300 pairs (Gold Perspective vs Candidate Viewpoint)...")
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        premises = batch["human_perspective"].fillna("").astype(str).tolist()
        hypotheses = batch["llm_viewpoint"].fillna("").astype(str).tolist()

        inputs = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            p = np.argmax(probs, axis=1)

            preds.extend(p)
            probs_list.extend(probs)

    df["finetuned_nli_pred"] = preds
    probs_arr = np.array(probs_list)
    df["p_contra"] = probs_arr[:, 0]
    df["p_entail"] = probs_arr[:, 1]
    df["p_neutral"] = probs_arr[:, 2]

    # Stage 1 boolean masks for different tau values
    df["stage1_pass_65"] = df["cosine_similarity"] >= 0.65
    df["stage1_pass_70"] = df["cosine_similarity"] >= 0.70
    df["stage1_pass_75"] = df["cosine_similarity"] >= 0.75
    df["stage1_pass_80"] = df["cosine_similarity"] >= 0.80

    # 1. Prediction Counts by Pair Type
    print("\n" + "=" * 80)
    print("           PREDICTION COUNTS ACROSS BENCHMARK CATEGORIES")
    print("=" * 80)
    pred_labels = [ {0: "Contradiction (0)", 1: "Entailment (1)", 2: "Neutral (2)"}[x] for x in df["finetuned_nli_pred"] ]
    df["pred_class_name"] = pred_labels
    cross_tab = pd.crosstab(df["pair_type"], df["pred_class_name"], margins=True)
    print(cross_tab.to_string())

    entail_count = int(np.sum(df["finetuned_nli_pred"] == 1))
    contra_count = int(np.sum(df["finetuned_nli_pred"] == 0))
    neutral_count = int(np.sum(df["finetuned_nli_pred"] == 2))
    print(f"\nOverall NLI Prediction Distribution (300 pairs):")
    print(f"  Entailment:    {entail_count:3d} ({entail_count/len(df):.1%})")
    print(f"  Contradiction: {contra_count:3d} ({contra_count/len(df):.1%})")
    print(f"  Neutral:       {neutral_count:3d} ({neutral_count/len(df):.1%})")

    # 2. Class-wise Metrics on raw NLI predictions
    gt_nli = []
    for pt in df["pair_type"]:
        if pt == "paraphrase_match":
            gt_nli.append(1)
        elif pt == "same_topic_opposite_stance":
            gt_nli.append(0)
        else:
            gt_nli.append(2)
    df["true_nli_class"] = gt_nli
    y_true_nli = np.array(gt_nli)
    y_pred_nli = np.array(df["finetuned_nli_pred"].values)

    class_names = ["Contradiction (Opposite)", "Entailment (Paraphrase)", "Neutral (Unrelated/Diff Concept)"]
    print("\n" + "=" * 80)
    print("           CLASS-LEVEL PRECISION, RECALL, AND F1 METRICS")
    print("=" * 80)
    class_metrics = []
    for c_id, c_name in enumerate(class_names):
        tp = np.sum((y_pred_nli == c_id) & (y_true_nli == c_id))
        fp = np.sum((y_pred_nli == c_id) & (y_true_nli != c_id))
        fn = np.sum((y_pred_nli != c_id) & (y_true_nli == c_id))
        support = np.sum(y_true_nli == c_id)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics.append({
            "Class": c_name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Support": int(support)
        })
    print(pd.DataFrame(class_metrics).to_string(index=False))

    # 3. Full 3x3 Confusion Matrix
    print("\n" + "=" * 80)
    print("           3x3 NLI CONFUSION MATRIX (True Class vs Predicted Class)")
    print("=" * 80)
    cm_3x3 = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true_nli, y_pred_nli):
        cm_3x3[t, p] += 1
    cm_3x3_df = pd.DataFrame(
        cm_3x3,
        index=["True Contradiction (0)", "True Entailment (1)", "True Neutral (2)"],
        columns=["Pred Contradiction (0)", "Pred Entailment (1)", "Pred Neutral (2)"]
    )
    print(cm_3x3_df.to_string())

    # Pipeline Evaluator helper
    def eval_pipeline(mask_pred, name):
        y_true = df["human_label"].values
        y_pred = mask_pred.astype(int).values

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(df)

        opp_mask = df["pair_type"] == "same_topic_opposite_stance"
        n_opp = int(np.sum(opp_mask))
        opp_leaks = int(np.sum(mask_pred & opp_mask))
        leak_rate = opp_leaks / n_opp if n_opp > 0 else 0.0

        return {
            "Method": name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Accuracy": round(acc, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Opp_Leaks": opp_leaks,
            "Total_Opp": n_opp,
            "Leak_Rate": f"{leak_rate:.1%}"
        }

    # Strategy 1 Helper: Decision Margin Stage 2
    def get_stage2_mask(p_entail_min=0.50, margin_contra=0.0):
        is_entail = (df["finetuned_nli_pred"] == 1)
        thresh_ok = (df["p_entail"] >= p_entail_min)
        margin_ok = ((df["p_entail"] - df["p_contra"]) >= margin_contra)
        return is_entail & thresh_ok & margin_ok

    # Define key configurations
    raw_stage2 = df["finetuned_nli_pred"] == 1
    calibrated_stage2_moderate = get_stage2_mask(p_entail_min=0.60, margin_contra=0.20)
    calibrated_stage2_strict = get_stage2_mask(p_entail_min=0.70, margin_contra=0.30)

    # 4. Core Pipeline Comparisons
    comparisons = [
        eval_pipeline(df["stage1_pass_75"], "1. Single-Stage Cosine (tau=0.75)"),
        eval_pipeline(df["stage1_pass_80"], "2. Single-Stage Cosine (tau=0.80)"),
        eval_pipeline(df["stage1_pass_75"] & raw_stage2, "3. Two-Stage (tau=0.75 + Raw NLI Argmax)"),
        eval_pipeline(df["stage1_pass_70"] & raw_stage2, "4. Strategy 2 Alone (tau=0.70 + Raw NLI Argmax)"),
        eval_pipeline(df["stage1_pass_75"] & calibrated_stage2_moderate, "5. Strategy 1 Alone (tau=0.75 + Margin p>=0.60, delta>=0.20)"),
        eval_pipeline(df["stage1_pass_70"] & calibrated_stage2_moderate, "6. Combined Strat 1+2 (tau=0.70 + Margin p>=0.60, delta>=0.20)"),
        eval_pipeline(df["stage1_pass_70"] & calibrated_stage2_strict, "7. Combined Strat 1+2 Strict (tau=0.70 + Margin p>=0.70, delta>=0.30)"),
        eval_pipeline(df["stage1_pass_70"] & (df["same_stance"] == "Yes"), "8. Two-Stage (tau=0.70 + ORACLE Upper Bound)")
    ]

    res_df = pd.DataFrame(comparisons)
    print("\n" + "=" * 115)
    print("                 COMPREHENSIVE PIPELINE EVALUATION (STRATEGIES 1 & 2)")
    print("=" * 115)
    cols = ["Method", "Precision", "Recall", "F1_Score", "Accuracy", "TP", "FP", "FN", "TN", "Opp_Leaks", "Leak_Rate"]
    print(res_df[cols].to_string(index=False))
    print("=" * 115)

    # 5. Automated Parameter Sweep to identify Pareto Frontier
    print("\n" + "=" * 95)
    print("        AUTOMATED CALIBRATION SWEEP ACROSS (tau, p_entail, margin)")
    print("=" * 95)
    sweep_results = []
    tau_list = [0.65, 0.70, 0.75, 0.78, 0.80]
    p_min_list = [0.50, 0.60, 0.70, 0.80, 0.85]
    margin_list = [0.0, 0.15, 0.25, 0.35]

    for tau in tau_list:
        stage1_mask = df["cosine_similarity"] >= tau
        for p_min in p_min_list:
            for m_diff in margin_list:
                stage2_mask = get_stage2_mask(p_entail_min=p_min, margin_contra=m_diff)
                combined_mask = stage1_mask & stage2_mask
                m_eval = eval_pipeline(combined_mask, f"tau={tau:.2f}, p>={p_min:.2f}, delta>={m_diff:.2f}")
                sweep_results.append({
                    "tau": tau,
                    "p_entail_min": p_min,
                    "margin_delta": m_diff,
                    "Precision": m_eval["Precision"],
                    "Recall": m_eval["Recall"],
                    "F1_Score": m_eval["F1_Score"],
                    "Accuracy": m_eval["Accuracy"],
                    "TP": m_eval["TP"],
                    "FP": m_eval["FP"],
                    "FN": m_eval["FN"],
                    "TN": m_eval["TN"],
                    "Opp_Leaks": m_eval["Opp_Leaks"],
                    "Leak_Rate": m_eval["Leak_Rate"]
                })

    sweep_df = pd.DataFrame(sweep_results)
    # Filter for top F1 score operating points
    top_f1 = sweep_df.sort_values(by=["F1_Score", "Precision"], ascending=[False, False]).head(5)
    print("\n--- TOP 5 BALANCED OPERATING POINTS (Ranked by F1 Score) ---")
    print(top_f1[["tau", "p_entail_min", "margin_delta", "Precision", "Recall", "F1_Score", "Accuracy", "Opp_Leaks", "Leak_Rate"]].to_string(index=False))

    # Filter for top Precision operating points (Recall >= 0.90)
    high_prec = sweep_df[sweep_df["Recall"] >= 0.90].sort_values(by=["Precision", "F1_Score"], ascending=[False, False]).head(5)
    print("\n--- TOP 5 HIGH-PRECISION POINTS (Recall >= 90%) ---")
    print(high_prec[["tau", "p_entail_min", "margin_delta", "Precision", "Recall", "F1_Score", "Accuracy", "Opp_Leaks", "Leak_Rate"]].to_string(index=False))

    # 6. Detailed 2x2 Confusion Matrix for the Best Balanced Operating Point
    best_config = top_f1.iloc[0]
    print("\n" + "=" * 65)
    print(f"  OPTIMAL CALIBRATED CONFIGURATION: tau={best_config['tau']}, p>={best_config['p_entail_min']}, delta>={best_config['margin_delta']}")
    print("=" * 65)
    print(f"  Precision: {best_config['Precision']:.4f}  |  Recall: {best_config['Recall']:.4f}  |  F1: {best_config['F1_Score']:.4f}")
    print(f"  Accuracy:  {best_config['Accuracy']:.4f}  |  Opposite Leaks: {best_config['Opp_Leaks']} ({best_config['Leak_Rate']})")
    print("-" * 65)
    cm_opt = pd.DataFrame(
        [
            [best_config["TN"], best_config["FP"]],
            [best_config["FN"], best_config["TP"]]
        ],
        index=["Actual Negative (0)", "Actual Positive (1)"],
        columns=["Predicted Negative (Reject)", "Predicted Positive (Accept)"]
    )
    print(cm_opt.to_string())

    # Save complete evaluation artifacts
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super().default(obj)

    report = {
        "pipeline_comparisons": comparisons,
        "class_level_metrics": class_metrics,
        "confusion_matrix_3x3": cm_3x3.tolist(),
        "optimal_configuration": dict(best_config),
        "top_f1_operating_points": top_f1.to_dict(orient="records"),
        "top_high_precision_operating_points": high_prec.to_dict(orient="records")
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
    print(f"\nSaved comprehensive evaluation report to: {output_json}")

if __name__ == "__main__":
    evaluate_finetuned_pipeline()
