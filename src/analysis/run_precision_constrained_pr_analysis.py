"""
Precision-Constrained Recall and Multi-Objective Operating Point Analysis:
Implements the reviewer-recommended protocol:
1. Primary Objective: Recall at Precision >= 90% (theta* >= 0.90)
2. Targets at Precision >= 95% and >= 98%
3. Secondary Objective: Maximum F1 Operating Point (theta*_max_f1)
4. Full transfer evaluation to the held-out test split (N=147, 66 survivors)
5. Precision-Recall curves on both Calibration and Test sets with labeled operating points.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("precision_constrained_pr_analysis")


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(float(max(0.0, centre - spread)), 4), round(float(min(1.0, centre + spread)), 4)


def infer_v3_probs(model, tokenizer, device, premises, hypotheses, batch_size=32):
    import torch
    probs_list = []
    ent_idx = 1
    for i in range(0, len(premises), batch_size):
        p_batch = [str(p) if pd.notna(p) else "" for p in premises[i : i + batch_size]]
        h_batch = [str(h) if pd.notna(h) else "" for h in hypotheses[i : i + batch_size]]
        inputs = tokenizer(
            p_batch, h_batch,
            truncation=True, max_length=256, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.extend(probs[:, ent_idx])
    return np.array(probs_list)


def evaluate_at_threshold(y_true, scores, theta, dropped_pos=0, filtered_neg=0):
    preds = (scores >= theta).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn_cond = int(np.sum((preds == 0) & (y_true == 1)))
    tn_cond = int(np.sum((preds == 0) & (y_true == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec_cond = tp / (tp + fn_cond) if (tp + fn_cond) > 0 else 0.0
    f1_cond = 2 * prec * rec_cond / (prec + rec_cond) if (prec + rec_cond) > 0 else 0.0

    fn_e2e = fn_cond + dropped_pos
    tn_e2e = tn_cond + filtered_neg
    total_pos = int(np.sum(y_true == 1)) + dropped_pos
    rec_e2e = tp / total_pos if total_pos > 0 else 0.0
    f1_e2e = 2 * prec * rec_e2e / (prec + rec_e2e) if (prec + rec_e2e) > 0 else 0.0

    prec_ci = compute_wilson_ci(tp, tp + fp)
    rec_e2e_ci = compute_wilson_ci(tp, total_pos)

    return {
        "theta": round(float(theta), 4),
        "tp": tp, "fp": fp, "fn_cond": fn_cond, "tn_cond": tn_cond,
        "fn_e2e": fn_e2e, "tn_e2e": tn_e2e,
        "precision": round(float(prec), 4),
        "precision_95_ci": prec_ci,
        "recall_cond": round(float(rec_cond), 4),
        "recall_e2e": round(float(rec_e2e), 4),
        "recall_e2e_95_ci": rec_e2e_ci,
        "f1_cond": round(float(f1_cond), 4),
        "f1_e2e": round(float(f1_e2e), 4)
    }


def find_operating_thresholds_on_pool(y_pool, scores_pool):
    thetas = np.arange(0.01, 1.00, 0.005)
    best_p90 = None
    best_p95 = None
    best_p98 = None
    best_f1 = None

    max_rec_p90 = -1.0
    max_rec_p95 = -1.0
    max_rec_p98 = -1.0
    max_f1_val = -1.0

    for th in thetas:
        res = evaluate_at_threshold(y_pool, scores_pool, th)
        p = res["precision"]
        r = res["recall_cond"]
        f = res["f1_cond"]

        # Precision >= 90%
        if p >= 0.90:
            if r > max_rec_p90:
                max_rec_p90 = r
                best_p90 = res

        # Precision >= 95%
        if p >= 0.95:
            if r > max_rec_p95:
                max_rec_p95 = r
                best_p95 = res

        # Precision >= 98%
        if p >= 0.98:
            if r > max_rec_p98:
                max_rec_p98 = r
                best_p98 = res

        # Max F1
        if f > max_f1_val:
            max_f1_val = f
            best_f1 = res

    return {
        "prec_ge_90": best_p90,
        "prec_ge_95": best_p95,
        "prec_ge_98": best_p98,
        "max_f1": best_f1
    }


def main():
    logger.info("Starting Precision-Constrained Recall and Multi-Objective Analysis...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3_path = resolve("models/argument_nli_deberta_v3_v3")
    logger.info(f"Loading V3 model from {v3_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(v3_path)
    model = AutoModelForSequenceClassification.from_pretrained(v3_path).to(device)
    model.eval()

    # Load Calibration Pool
    cal_p = resolve("data/calibration/expanded_stage1_calibration_pool.csv")
    cal_df = pd.read_csv(cal_p)
    logger.info(f"Loaded calibration pool: {len(cal_df)} rows")
    y_cal = (cal_df["label_3class"] == 2).astype(int).values
    p_base_cal = cal_df["p_verification"].values
    p_v3_cal = infer_v3_probs(model, tokenizer, device, cal_df["premise"], cal_df["hypothesis"])

    # Load Held-Out Test Split
    test_p = resolve("data/calibration/tri_dataset_750_pairs.csv")
    full_test_df = pd.read_csv(test_p)
    test_df = full_test_df[full_test_df["split"] == "test"].copy().reset_index(drop=True)
    logger.info(f"Loaded held-out test split: {len(test_df)} rows")

    p_v3_test_full = infer_v3_probs(model, tokenizer, device, test_df["text_b"], test_df["text_a"])
    test_df["p_v3"] = p_v3_test_full

    survivors_mask = (test_df["cosine_similarity"] >= 0.75).values
    survivors_df = test_df[survivors_mask].copy().reset_index(drop=True)

    y_test_surv = (survivors_df["label_3class"] == 2).astype(int).values
    p_base_surv = survivors_df["p_verification"].values
    p_v3_surv = survivors_df["p_v3"].values

    total_pos = 60
    surv_pos = int(np.sum(y_test_surv == 1))  # 59
    dropped_pos = total_pos - surv_pos       # 1
    filtered_neg = len(test_df) - len(survivors_df) - dropped_pos  # 80

    # 1. Calibration Operating Thresholds
    logger.info("Calculating optimal operating thresholds on Calibration Pool (N=268)...")
    base_pool_opts = find_operating_thresholds_on_pool(y_cal, p_base_cal)
    v3_pool_opts = find_operating_thresholds_on_pool(y_cal, p_v3_cal)

    # 2. Evaluate Operating Thresholds on Test Split
    logger.info("Evaluating Operating Thresholds on Held-Out Test Benchmark (N=147)...")
    objectives = [
        ("Primary: Precision >= 90%", "prec_ge_90"),
        ("High-Precision Target: Precision >= 95%", "prec_ge_95"),
        ("Ultra-High-Precision Target: Precision >= 98%", "prec_ge_98"),
        ("Secondary: Maximum F1", "max_f1")
    ]

    table_rows = []
    report_dict = {
        "calibration_survivors_pool": {
            "n_total": len(cal_df), "n_positives": int(np.sum(y_cal == 1)), "n_negatives": int(np.sum(y_cal == 0))
        },
        "test_benchmark": {
            "n_total": len(test_df), "n_survivors": len(survivors_df), "n_true_positives": total_pos
        },
        "operating_points": {}
    }

    for label, key in objectives:
        base_opt = base_pool_opts[key]
        v3_opt = v3_pool_opts[key]

        th_base = base_opt["theta"] if base_opt else 0.72
        th_v3 = v3_opt["theta"] if v3_opt else 0.95

        # Test evaluation
        base_test_res = evaluate_at_threshold(y_test_surv, p_base_surv, th_base, dropped_pos, filtered_neg)
        v3_test_res = evaluate_at_threshold(y_test_surv, p_v3_surv, th_v3, dropped_pos, filtered_neg)

        table_rows.append({
            "Objective": label,
            "Model": "Baseline DeBERTa",
            "Selected_Theta": th_base,
            "Cal_Precision": base_opt["precision"] if base_opt else None,
            "Cal_Recall": base_opt["recall_cond"] if base_opt else None,
            "Cal_F1": base_opt["f1_cond"] if base_opt else None,
            "Test_TP": base_test_res["tp"],
            "Test_FP": base_test_res["fp"],
            "Test_Cond_Prec": base_test_res["precision"],
            "Test_Cond_Rec": base_test_res["recall_cond"],
            "Test_Pipeline_Rec": base_test_res["recall_e2e"],
            "Test_Pipeline_Rec_95CI": f"{base_test_res['recall_e2e_95_ci'][0]:.4f}–{base_test_res['recall_e2e_95_ci'][1]:.4f}",
            "Test_Pipeline_F1": base_test_res["f1_e2e"]
        })

        table_rows.append({
            "Objective": label,
            "Model": "Version 3 (Coverage-Aware)",
            "Selected_Theta": th_v3,
            "Cal_Precision": v3_opt["precision"] if v3_opt else None,
            "Cal_Recall": v3_opt["recall_cond"] if v3_opt else None,
            "Cal_F1": v3_opt["f1_cond"] if v3_opt else None,
            "Test_TP": v3_test_res["tp"],
            "Test_FP": v3_test_res["fp"],
            "Test_Cond_Prec": v3_test_res["precision"],
            "Test_Cond_Rec": v3_test_res["recall_cond"],
            "Test_Pipeline_Rec": v3_test_res["recall_e2e"],
            "Test_Pipeline_Rec_95CI": f"{v3_test_res['recall_e2e_95_ci'][0]:.4f}–{v3_test_res['recall_e2e_95_ci'][1]:.4f}",
            "Test_Pipeline_F1": v3_test_res["f1_e2e"]
        })

        report_dict["operating_points"][key] = {
            "objective": label,
            "baseline": {"cal": base_opt, "test": base_test_res},
            "v3": {"cal": v3_opt, "test": v3_test_res}
        }

    out_df = pd.DataFrame(table_rows)
    out_csv = resolve("results/paper_tables/table_precision_constrained_operating_points.csv")
    out_df.to_csv(out_csv, index=False)
    out_json = resolve("results/metrics/precision_constrained_operating_points_report.json")
    save_json(report_dict, out_json)
    logger.info(f"Saved operating points table to {out_csv}")
    logger.info(f"Saved operating points JSON to {out_json}")

    # 3. Precision-Recall Curves Plotting
    logger.info("Generating Publication-Grade Precision-Recall Curve Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Calibration Pool (N=268)
    p_b_cal_prec, p_b_cal_rec, _ = precision_recall_curve(y_cal, p_base_cal)
    p_v_cal_prec, p_v_cal_rec, _ = precision_recall_curve(y_cal, p_v3_cal)
    auprc_b_cal = average_precision_score(y_cal, p_base_cal)
    auprc_v_cal = average_precision_score(y_cal, p_v3_cal)

    axes[0].plot(p_b_cal_rec, p_b_cal_prec, label=f"Baseline (AUPRC = {auprc_b_cal:.3f})", color="#1f77b4", lw=2.2)
    axes[0].plot(p_v_cal_rec, p_v_cal_prec, label=f"V3 (AUPRC = {auprc_v_cal:.3f})", color="#d62728", lw=2.2, linestyle="--")

    # Mark points on Cal
    if base_pool_opts["prec_ge_90"]:
        axes[0].scatter([base_pool_opts["prec_ge_90"]["recall_cond"]], [base_pool_opts["prec_ge_90"]["precision"]], color="#1f77b4", s=70, marker="o", zorder=5)
    if v3_pool_opts["prec_ge_90"]:
        axes[0].scatter([v3_pool_opts["prec_ge_90"]["recall_cond"]], [v3_pool_opts["prec_ge_90"]["precision"]], color="#d62728", s=70, marker="s", zorder=5)
    if base_pool_opts["max_f1"]:
        axes[0].scatter([base_pool_opts["max_f1"]["recall_cond"]], [base_pool_opts["max_f1"]["precision"]], color="#1f77b4", s=80, marker="*", zorder=5)
    if v3_pool_opts["max_f1"]:
        axes[0].scatter([v3_pool_opts["max_f1"]["recall_cond"]], [v3_pool_opts["max_f1"]["precision"]], color="#d62728", s=80, marker="*", zorder=5)

    axes[0].axhline(0.90, color="gray", linestyle=":", alpha=0.8, label="90% Precision Constraint")
    axes[0].set_title("A: Precision–Recall Curve (Calibration Pool, N=268)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Conditional Recall", fontsize=10)
    axes[0].set_ylabel("Precision", fontsize=10)
    axes[0].set_ylim(0.70, 1.02)
    axes[0].set_xlim(0.0, 1.02)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=8.5)

    # Panel B: Test Split Survivors (N=66)
    p_b_test_prec, p_b_test_rec, _ = precision_recall_curve(y_test_surv, p_base_surv)
    p_v_test_prec, p_v_test_rec, _ = precision_recall_curve(y_test_surv, p_v3_surv)
    auprc_b_test = average_precision_score(y_test_surv, p_base_surv)
    auprc_v_test = average_precision_score(y_test_surv, p_v3_surv)

    axes[1].plot(p_b_test_rec, p_b_test_prec, label=f"Baseline (AUPRC = {auprc_b_test:.3f})", color="#1f77b4", lw=2.2)
    axes[1].plot(p_v_test_rec, p_v_test_prec, label=f"V3 (AUPRC = {auprc_v_test:.3f})", color="#d62728", lw=2.2, linestyle="--")

    # Mark evaluated test points
    b_test_p90 = report_dict["operating_points"]["prec_ge_90"]["baseline"]["test"]
    v_test_p90 = report_dict["operating_points"]["prec_ge_90"]["v3"]["test"]
    axes[1].scatter([b_test_p90["recall_cond"]], [b_test_p90["precision"]], color="#1f77b4", s=70, marker="o", zorder=5, label=f"Base (theta*={b_test_p90['theta']:.2f})")
    axes[1].scatter([v_test_p90["recall_cond"]], [v_test_p90["precision"]], color="#d62728", s=70, marker="s", zorder=5, label=f"V3 (theta*={v_test_p90['theta']:.2f})")

    axes[1].axhline(0.90, color="gray", linestyle=":", alpha=0.8)
    axes[1].set_title("B: Precision–Recall Curve (Test Survivors, N=66)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Conditional Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_ylim(0.70, 1.02)
    axes[1].set_xlim(0.0, 1.02)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    fig_path = resolve("results/paper_figures/precision_constrained_pr_curves.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall figure to {fig_path}")

    print("\n" + "=" * 115)
    print("      PRECISION-CONSTRAINED RECALL & MULTI-OBJECTIVE OPERATING POINTS (BASELINE vs V3)")
    print("=" * 115)
    print(out_df[["Objective", "Model", "Selected_Theta", "Cal_Precision", "Cal_Recall", "Test_TP", "Test_Cond_Prec", "Test_Pipeline_Rec", "Test_Pipeline_F1"]].to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()

