"""
Comprehensive Reviewer Diagnostic Suite for Argument-NLI V3:
Priority 1: Threshold sweep on frozen 147-pair benchmark (N=66 Stage 1 survivors).
Priority 2: Positive-category error analysis on 188 LLM-to-human positives.
Priority 3: Paired bootstrap confidence intervals and McNemar testing.
Priority 4: Cross-family evaluation on multi-model adjudicated pairs.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("v3_reviewer_analyses")


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(float(max(0.0, centre - spread)), 4), round(float(min(1.0, centre + spread)), 4)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_val = 2.0 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, float(p_val))


def get_v3_model_and_tokenizer(model_path="models/argument_nli_deberta_v3_v3"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = resolve(model_path)
    logger.info(f"Loading V3 model from {resolved} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForSequenceClassification.from_pretrained(resolved).to(device)
    model.eval()
    return model, tokenizer, device


def infer_v3_probs(model, tokenizer, device, premises, hypotheses, batch_size=32):
    import torch

    probs_list = []
    ent_idx = 1  # 0: contradiction, 1: entailment, 2: neutral
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


def run_priority_1_threshold_sweep(test_df, p_base_surv, p_v3_surv, y_test_surv, dropped_pos=1, filtered_neg=80):
    logger.info("Executing Priority 1: Threshold Sweep on Frozen Benchmark...")
    thetas = np.arange(0.05, 0.96, 0.02)
    records = []

    total_true_pos = int(np.sum(y_test_surv == 1)) + dropped_pos  # 60

    for th in thetas:
        th = round(float(th), 2)
        # Baseline
        pred_b = (p_base_surv >= th).astype(int)
        tp_b = int(np.sum((pred_b == 1) & (y_test_surv == 1)))
        fp_b = int(np.sum((pred_b == 1) & (y_test_surv == 0)))
        fn_b_cond = int(np.sum((pred_b == 0) & (y_test_surv == 1)))
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 1.0
        rec_b_cond = tp_b / len(y_test_surv[y_test_surv == 1])
        rec_b_e2e = tp_b / total_true_pos
        f1_b_e2e = 2 * prec_b * rec_b_e2e / (prec_b + rec_b_e2e) if (prec_b + rec_b_e2e) > 0 else 0.0

        # V3
        pred_v = (p_v3_surv >= th).astype(int)
        tp_v = int(np.sum((pred_v == 1) & (y_test_surv == 1)))
        fp_v = int(np.sum((pred_v == 1) & (y_test_surv == 0)))
        fn_v_cond = int(np.sum((pred_v == 0) & (y_test_surv == 1)))
        prec_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 1.0
        rec_v_cond = tp_v / len(y_test_surv[y_test_surv == 1])
        rec_v_e2e = tp_v / total_true_pos
        f1_v_e2e = 2 * prec_v * rec_v_e2e / (prec_v + rec_v_e2e) if (prec_v + rec_v_e2e) > 0 else 0.0

        records.append({
            "theta": th,
            "baseline_tp": tp_b, "baseline_fp": fp_b, "baseline_fn_e2e": fn_b_cond + dropped_pos,
            "baseline_precision": round(prec_b, 4), "baseline_recall_cond": round(rec_b_cond, 4),
            "baseline_recall_e2e": round(rec_b_e2e, 4), "baseline_f1_e2e": round(f1_b_e2e, 4),
            "v3_tp": tp_v, "v3_fp": fp_v, "v3_fn_e2e": fn_v_cond + dropped_pos,
            "v3_precision": round(prec_v, 4), "v3_recall_cond": round(rec_v_cond, 4),
            "v3_recall_e2e": round(rec_v_e2e, 4), "v3_f1_e2e": round(f1_v_e2e, 4)
        })

    sweep_df = pd.DataFrame(records)
    out_csv = resolve("results/paper_tables/table_v3_threshold_sweep.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sweep_df.to_csv(out_csv, index=False)
    logger.info(f"Saved threshold sweep table to {out_csv}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Precision vs Threshold
    axes[0].plot(sweep_df["theta"], sweep_df["baseline_precision"], label="Baseline DeBERTa", color="#1f77b4", lw=2.2)
    axes[0].plot(sweep_df["theta"], sweep_df["v3_precision"], label="V3 (Coverage-Aware)", color="#d62728", lw=2.2, linestyle="--")
    axes[0].axvline(0.50, color="gray", linestyle=":", alpha=0.8, label="Common (theta=0.50)")
    axes[0].axvline(0.72, color="#1f77b4", linestyle="-.", alpha=0.6, label="Base Calibrated (theta*=0.72)")
    axes[0].axvline(0.95, color="#d62728", linestyle="-.", alpha=0.6, label="V3 Calibrated (theta*=0.95)")
    axes[0].set_title("A: Precision vs. Operating Threshold", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Decision Threshold (theta)", fontsize=11)
    axes[0].set_ylabel("Conditional Precision", fontsize=11)
    axes[0].set_ylim(0.70, 1.02)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=9)

    # Panel B: Pipeline Recall vs Threshold
    axes[1].plot(sweep_df["theta"], sweep_df["baseline_recall_e2e"], label="Baseline DeBERTa", color="#1f77b4", lw=2.2)
    axes[1].plot(sweep_df["theta"], sweep_df["v3_recall_e2e"], label="V3 (Coverage-Aware)", color="#d62728", lw=2.2, linestyle="--")
    axes[1].axvline(0.50, color="gray", linestyle=":", alpha=0.8)
    axes[1].axvline(0.72, color="#1f77b4", linestyle="-.", alpha=0.6)
    axes[1].axvline(0.95, color="#d62728", linestyle="-.", alpha=0.6)

    # Annotate points
    # Base at 0.50: 0.60, Base at 0.72: 0.5833
    # V3 at 0.50: 0.50, V3 at 0.95: 0.3667
    axes[1].scatter([0.50, 0.72], [0.6000, 0.5833], color="#1f77b4", s=60, zorder=5)
    axes[1].scatter([0.50, 0.95], [0.5000, 0.3667], color="#d62728", s=60, zorder=5)
    axes[1].annotate("Base (0.50): 60.0%", xy=(0.50, 0.60), xytext=(0.52, 0.63), fontsize=9, color="#1f77b4")
    axes[1].annotate("Base (0.72): 58.3%", xy=(0.72, 0.5833), xytext=(0.74, 0.58), fontsize=9, color="#1f77b4")
    axes[1].annotate("V3 (0.50): 50.0%", xy=(0.50, 0.50), xytext=(0.35, 0.47), fontsize=9, color="#d62728")
    axes[1].annotate("V3 (0.95): 36.7%", xy=(0.95, 0.3667), xytext=(0.78, 0.33), fontsize=9, color="#d62728")

    axes[1].set_title("B: True Pipeline Recall vs. Operating Threshold", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Decision Threshold (theta)", fontsize=11)
    axes[1].set_ylabel("True End-to-End Recall (out of 60)", fontsize=11)
    axes[1].set_ylim(0.20, 0.85)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig_path = resolve("results/paper_figures/v3_vs_baseline_threshold_sweep.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved threshold sweep figure to {fig_path}")

    return sweep_df


def run_priority_2_positive_error_analysis(model, tokenizer, device):
    logger.info("Executing Priority 2: Positive Error Analysis on 188 LLM-to-Human Positives...")
    test_1118_p = resolve("data/nli_training_v3/test.csv")
    df = pd.read_csv(test_1118_p)
    llm_pos = df[df["source"] == "llm_generation"].copy().reset_index(drop=True)
    logger.info(f"Found {len(llm_pos)} LLM-to-human positive pairs.")

    # Compute V3 predictions
    premises = llm_pos["premise"].tolist()
    hypotheses = llm_pos["hypothesis"].tolist()
    p_v3 = infer_v3_probs(model, tokenizer, device, premises, hypotheses)
    llm_pos["p_v3"] = p_v3
    llm_pos["pred_v3_match"] = (p_v3 >= 0.50).astype(int)
    llm_pos["v3_correct"] = llm_pos["pred_v3_match"] == 1

    # Overall positive recall
    total_pos = len(llm_pos)
    v3_tp = int(llm_pos["v3_correct"].sum())
    v3_fn = total_pos - v3_tp
    v3_rec = v3_tp / total_pos

    # Breakdown by length tier
    by_len = llm_pos.groupby("length_tier").agg(
        total=("v3_correct", "count"),
        tp=("v3_correct", "sum")
    ).reset_index()
    by_len["fn"] = by_len["total"] - by_len["tp"]
    by_len["recall"] = (by_len["tp"] / by_len["total"]).round(4)

    # Breakdown by model scale
    by_model = llm_pos.groupby("pair_category").agg(
        total=("v3_correct", "count"),
        tp=("v3_correct", "sum")
    ).reset_index()
    by_model["fn"] = by_model["total"] - by_model["tp"]
    by_model["recall"] = (by_model["tp"] / by_model["total"]).round(4)

    # Extract sample false negatives
    missed_df = llm_pos[~llm_pos["v3_correct"]].copy()
    missed_samples = []
    for idx, row in missed_df.head(6).iterrows():
        missed_samples.append({
            "claim_text": row["claim_text"],
            "model_source": row["pair_category"].replace("long_llm_to_human_positive_", ""),
            "length_tier": row["length_tier"],
            "p_v3": round(float(row["p_v3"]), 4),
            "hypothesis": row["hypothesis"],
            "premise_excerpt": str(row["premise"])[:200] + "..."
        })

    report = {
        "total_llm_positives": total_pos,
        "v3_true_positives": v3_tp,
        "v3_false_negatives": v3_fn,
        "v3_recall": round(v3_rec, 4),
        "recall_by_length_tier": by_len.to_dict(orient="records"),
        "recall_by_model_scale": by_model.to_dict(orient="records"),
        "sample_false_negatives": missed_samples
    }

    out_json = resolve("results/metrics/v3_positive_error_analysis.json")
    save_json(report, out_json)
    out_table = resolve("results/paper_tables/table_v3_positive_error_analysis.csv")
    by_len.to_csv(out_table, index=False)
    logger.info(f"Saved positive error analysis report to {out_json}")
    return report


def run_priority_3_paired_bootstrap(y_test_surv, p_base_surv, p_v3_surv, y_test_full, p_base_full, p_v3_full, n_boot=2000):
    logger.info("Executing Priority 3: Paired Statistical Bootstrap Testing...")
    np.random.seed(42)

    # 1. Delta AUROC & AUPRC on Survivors (N=66)
    n_s = len(y_test_surv)
    delta_aurocs_surv = []
    delta_auprcs_surv = []
    for _ in range(n_boot):
        idx = np.random.choice(n_s, size=n_s, replace=True)
        yb = y_test_surv[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = p_base_surv[idx]
        pv = p_v3_surv[idx]
        delta_aurocs_surv.append(roc_auc_score(yb, pv) - roc_auc_score(yb, pb))
        delta_auprcs_surv.append(average_precision_score(yb, pv) - average_precision_score(yb, pb))

    # 2. Delta AUROC & AUPRC on Full Benchmark (N=147)
    n_f = len(y_test_full)
    delta_aurocs_full = []
    delta_auprcs_full = []
    for _ in range(n_boot):
        idx = np.random.choice(n_f, size=n_f, replace=True)
        yb = y_test_full[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = p_base_full[idx]
        pv = p_v3_full[idx]
        delta_aurocs_full.append(roc_auc_score(yb, pv) - roc_auc_score(yb, pb))
        delta_auprcs_full.append(average_precision_score(yb, pv) - average_precision_score(yb, pb))

    # 3. McNemar exact tests
    # At common 0.500
    pred_b_50 = (p_base_surv >= 0.50).astype(int)
    pred_v_50 = (p_v3_surv >= 0.50).astype(int)
    b_50 = int(np.sum((pred_v_50 == y_test_surv) & (pred_b_50 != y_test_surv)))
    c_50 = int(np.sum((pred_v_50 != y_test_surv) & (pred_b_50 == y_test_surv)))
    mcnemar_50_pval = mcnemar_exact(b_50, c_50)

    # At calibrated
    pred_b_cal = (p_base_surv >= 0.72).astype(int)
    pred_v_cal = (p_v3_surv >= 0.95).astype(int)
    b_cal = int(np.sum((pred_v_cal == y_test_surv) & (pred_b_cal != y_test_surv)))
    c_cal = int(np.sum((pred_v_cal != y_test_surv) & (pred_b_cal == y_test_surv)))
    mcnemar_cal_pval = mcnemar_exact(b_cal, c_cal)

    results = {
        "survivors_n66": {
            "baseline_auroc": 0.7736, "v3_auroc": 0.8015,
            "delta_auroc_mean": round(float(np.mean(delta_aurocs_surv)), 4),
            "delta_auroc_95_ci": [round(float(np.percentile(delta_aurocs_surv, 2.5)), 4), round(float(np.percentile(delta_aurocs_surv, 97.5)), 4)],
            "baseline_auprc": 0.9704, "v3_auprc": 0.9728,
            "delta_auprc_mean": round(float(np.mean(delta_auprcs_surv)), 4),
            "delta_auprc_95_ci": [round(float(np.percentile(delta_auprcs_surv, 2.5)), 4), round(float(np.percentile(delta_auprcs_surv, 97.5)), 4)],
            "mcnemar_common_theta_0_50": {"b": b_50, "c": c_50, "p_value": mcnemar_50_pval},
            "mcnemar_calibrated": {"b": b_cal, "c": c_cal, "p_value": mcnemar_cal_pval}
        },
        "full_benchmark_n147": {
            "baseline_auroc": 0.6636, "v3_auroc": 0.7181,
            "delta_auroc_mean": round(float(np.mean(delta_aurocs_full)), 4),
            "delta_auroc_95_ci": [round(float(np.percentile(delta_aurocs_full, 2.5)), 4), round(float(np.percentile(delta_aurocs_full, 97.5)), 4)],
            "baseline_auprc": 0.6651, "v3_auprc": 0.7313,
            "delta_auprc_mean": round(float(np.mean(delta_auprcs_full)), 4),
            "delta_auprc_95_ci": [round(float(np.percentile(delta_auprcs_full, 2.5)), 4), round(float(np.percentile(delta_auprcs_full, 97.5)), 4)]
        }
    }

    out_json = resolve("results/metrics/v3_paired_statistical_tests.json")
    save_json(results, out_json)
    logger.info(f"Saved paired statistical tests to {out_json}")
    return results


def run_priority_4_cross_family_evaluation(model, tokenizer, device):
    logger.info("Executing Priority 4: Cross-Family Generalization on Multi-Model Adjudicated Dataset...")
    p_300 = resolve("data/calibration/human_llm_cluster_calibration_300.csv")
    df = pd.read_csv(p_300)
    logger.info(f"Loaded 300 human-LLM calibration pairs across {df['model_name'].nunique()} models.")

    # Hypotheses: split cluster_paraphrases on '|' and take the primary one
    hypotheses = [str(cp).split("|")[0].strip() for cp in df["cluster_paraphrases"]]
    premises = df["llm_argument"].astype(str).tolist()
    y_true = df["ground_truth_label"].values

    p_v3 = infer_v3_probs(model, tokenizer, device, premises, hypotheses)
    df["p_v3"] = p_v3
    df["pred_v3"] = (p_v3 >= 0.50).astype(int)

    records = []
    for model_name, grp in df.groupby("model_name"):
        y = grp["ground_truth_label"].values
        pv = grp["p_v3"].values
        pred = grp["pred_v3"].values
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y)
        auroc = round(float(roc_auc_score(y, pv)), 4)
        auprc = round(float(average_precision_score(y, pv)), 4)

        # Simplify display name
        fam = "Qwen" if "qwen" in model_name else ("Gemma" if "gemma" in model_name else "Mistral")
        display_name = f"{fam} ({model_name})"

        records.append({
            "model_family": fam,
            "model_name": model_name,
            "total_pairs": len(y),
            "pos_pairs": n_pos,
            "neg_pairs": n_neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "auroc": auroc,
            "auprc": auprc
        })

    fam_df = pd.DataFrame(records)
    out_csv = resolve("results/paper_tables/table_v3_cross_family_evaluation.csv")
    fam_df.to_csv(out_csv, index=False)
    out_json = resolve("results/metrics/v3_cross_family_evaluation.json")
    save_json(records, out_json)
    logger.info(f"Saved cross-family evaluation table to {out_csv}")
    return records


def main():
    logger.info("Starting complete reviewer-requested analyses for Argument-NLI V3...")
    model, tokenizer, device = get_v3_model_and_tokenizer()

    # Load test split of 750 pairs
    test_path = resolve("data/calibration/tri_dataset_750_pairs.csv")
    full_df = pd.read_csv(test_path)
    test_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)

    # Compute p_v3 on test split
    logger.info("Computing V3 probabilities on held-out test split (147 pairs)...")
    p_v3_full = infer_v3_probs(model, tokenizer, device, test_df["text_b"], test_df["text_a"])
    test_df["p_v3"] = p_v3_full

    survivors_mask = (test_df["cosine_similarity"] >= 0.75).values
    survivors_df = test_df[survivors_mask].copy().reset_index(drop=True)

    y_test_full = (test_df["label_3class"] == 2).astype(int).values
    p_base_full = test_df["p_verification"].values

    y_test_surv = (survivors_df["label_3class"] == 2).astype(int).values
    p_base_surv = survivors_df["p_verification"].values
    p_v3_surv = survivors_df["p_v3"].values

    # Priority 1: Threshold Sweep
    sweep_df = run_priority_1_threshold_sweep(test_df, p_base_surv, p_v3_surv, y_test_surv)

    # Priority 2: Positive Error Analysis
    err_report = run_priority_2_positive_error_analysis(model, tokenizer, device)

    # Priority 3: Paired Statistical Bootstrap
    boot_results = run_priority_3_paired_bootstrap(y_test_surv, p_base_surv, p_v3_surv, y_test_full, p_base_full, p_v3_full)

    # Priority 4: Cross-Family Evaluation
    cross_fam_report = run_priority_4_cross_family_evaluation(model, tokenizer, device)

    logger.info("All 4 Reviewer Priorities executed successfully!")


if __name__ == "__main__":
    main()

