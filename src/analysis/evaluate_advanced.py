import argparse, json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, save_csv, save_json, get_model_tag
from src.utils.logging_utils import get_logger
from src.analysis.similarity_matrix import build_all_similarity_matrices
from run_pipeline import build_config

logger = get_logger("evaluate_advanced")

def compute_claim_metrics(sim_matrix: np.ndarray, gen_sim_matrix: np.ndarray, tau: float, k_sub: int = None) -> dict:
    n_samples, n_persp = sim_matrix.shape
    if n_persp == 0 or n_samples == 0:
        return {"coverage": 0.0, "precision": 0.0, "f1": 0.0}

    if k_sub is not None and k_sub < n_samples:
        sim_matrix = sim_matrix[:k_sub, :]
        gen_sim_matrix = gen_sim_matrix[:k_sub, :k_sub]
        n_samples = k_sub

    best_match_per_persp = sim_matrix.max(axis=0)
    coverage = float((best_match_per_persp >= tau).sum()) / n_persp

    best_match_per_sample = sim_matrix.max(axis=1)

    # Agglomerative Deduplication / Clustering
    clusters = []
    assigned = set()
    for i in range(n_samples):
        if i in assigned: continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, n_samples):
            if j not in assigned and gen_sim_matrix[i, j] >= tau:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    precision_hits = sum(
        1 for cluster in clusters 
        if any(best_match_per_sample[idx] >= tau for idx in cluster)
    )
    n_unique_clusters = len(clusters)

    precision = float(precision_hits) / n_unique_clusters if n_unique_clusters > 0 else 0.0

    f1 = (2 * precision * coverage / (precision + coverage)) if (precision + coverage) > 0 else 0.0
    return {"coverage": coverage, "precision": precision, "f1": f1}

def evaluate_model_at_tau(matrices: dict, tau: float, k_sub: int = None) -> dict:
    coverages, precisions, f1s = [], [], []
    for claim_id, data in matrices.items():
        gen_sim = data.get("gen_sim_matrix", np.eye(data["sim_matrix"].shape[0]))
        m = compute_claim_metrics(data["sim_matrix"], gen_sim, tau=tau, k_sub=k_sub)
        coverages.append(m["coverage"]); precisions.append(m["precision"]); f1s.append(m["f1"])

    return {
        "coverage": float(np.mean(coverages)) if coverages else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_claim_coverage": np.array(coverages),
        "per_claim_precision": np.array(precisions),
        "per_claim_f1": np.array(f1s),
    }

def paired_bootstrap_test(metrics_a: np.ndarray, metrics_b: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(metrics_a)
    if n == 0 or len(metrics_b) != n:
        return {"mean_diff": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0, "p_value": 1.0}

    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(np.mean(metrics_b[idx]) - np.mean(metrics_a[idx]))

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    p_one_tail = np.mean(diffs <= 0) if mean_diff > 0 else np.mean(diffs >= 0)

    return {
        "mean_diff": mean_diff,
        "ci_95_lower": float(np.percentile(diffs, 2.5)),
        "ci_95_upper": float(np.percentile(diffs, 97.5)),
        "p_value": float(min(1.0, 2 * p_one_tail)),
    }

def run_advanced_evaluation_for_model(model_key: str, tau_list: list = None, k_list: list = None) -> dict:
    if tau_list is None: tau_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    if k_list is None: k_list = [1, 2, 3, 4, 5]

    config = build_config(model=model_key)
    matrices = build_all_similarity_matrices(config)
    model_tag = get_model_tag(config["llm"]["model"])

    # 1. Tau Sweep
    tau_rows = [
        {"tau": t, **evaluate_model_at_tau(matrices, tau=t)} for t in tau_list
    ]
    df_tau = pd.DataFrame(tau_rows)

    # 2. Subsampling k
    def_tau = config.get("analysis", {}).get("similarity_threshold", 0.75)
    k_rows = [
        {"k_samples": k, **evaluate_model_at_tau(matrices, tau=def_tau, k_sub=k)} for k in k_list
    ]
    df_k = pd.DataFrame(k_rows)

    metrics_dir = resolve(config["paths"]["metrics_dir"])
    save_csv(df_tau, os.path.join(metrics_dir, "advanced_tau_sweep.csv"))
    save_csv(df_k, os.path.join(metrics_dir, "advanced_k_subsampling.csv"))

    figures_dir = resolve(config["paths"]["figures_dir"])
    os.makedirs(figures_dir, exist_ok=True)

    # Plot Precision/Recall/F1 vs Tau
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_tau["tau"], df_tau["coverage"], marker="o", label="Coverage (Recall)", color="#1f77b4")
    ax.plot(df_tau["tau"], df_tau["precision"], marker="s", label="Precision", color="#ff7f0e")
    ax.plot(df_tau["tau"], df_tau["f1"], marker="^", label="F1 Score", color="#2ca02c")
    ax.set_xlabel("Similarity Threshold (τ)"); ax.set_ylabel("Score"); ax.set_xticks(tau_list)
    ax.set_title(f"Precision, Recall & F1 vs. τ ({config['llm']['model']})", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "precision_recall_f1_tau.png"), dpi=150); plt.close(fig)

    # Plot Coverage vs k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_k["k_samples"], df_k["coverage"], marker="o", label="Coverage (Recall)", color="#1f77b4")
    ax.plot(df_k["k_samples"], df_k["f1"], marker="^", label="F1 Score", color="#2ca02c")
    ax.set_xlabel("Subsampled k"); ax.set_ylabel("Score (τ=0.75)"); ax.set_xticks(k_list)
    ax.set_title(f"Subsampled Coverage vs. k ({config['llm']['model']})", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "coverage_vs_k.png"), dpi=150); plt.close(fig)

    logger.info(f"[{model_tag}] Saved advanced metrics & plots to {metrics_dir}")
    return {"model_key": model_key, "tau_sweep": df_tau, "k_sweep": df_k, "matrices": matrices}

def compare_models_bootstrap(model_a_key: str, model_b_key: str, tau: float = 0.75, n_bootstraps: int = 1000):
    mat_a = build_all_similarity_matrices(build_config(model=model_a_key))
    mat_b = build_all_similarity_matrices(build_config(model=model_b_key))

    claims = sorted(set(mat_a.keys()).intersection(set(mat_b.keys())))
    cov_a = [compute_claim_metrics(mat_a[c]["sim_matrix"], mat_a[c].get("gen_sim_matrix", np.eye(mat_a[c]["sim_matrix"].shape[0])), tau)["coverage"] for c in claims]
    cov_b = [compute_claim_metrics(mat_b[c]["sim_matrix"], mat_b[c].get("gen_sim_matrix", np.eye(mat_b[c]["sim_matrix"].shape[0])), tau)["coverage"] for c in claims]
    f1_a = [compute_claim_metrics(mat_a[c]["sim_matrix"], mat_a[c].get("gen_sim_matrix", np.eye(mat_a[c]["sim_matrix"].shape[0])), tau)["f1"] for c in claims]
    f1_b = [compute_claim_metrics(mat_b[c]["sim_matrix"], mat_b[c].get("gen_sim_matrix", np.eye(mat_b[c]["sim_matrix"].shape[0])), tau)["f1"] for c in claims]

    boot_cov = paired_bootstrap_test(np.array(cov_a), np.array(cov_b), n_bootstraps=n_bootstraps)
    boot_f1 = paired_bootstrap_test(np.array(f1_a), np.array(f1_b), n_bootstraps=n_bootstraps)

    print("\n" + "=" * 75)
    print(f" PAIRED BOOTSTRAP SIGNIFICANCE TEST: {model_a_key.upper()} vs {model_b_key.upper()} (τ={tau})")
    print("=" * 75)
    print(f" Coverage: {boot_cov['mean_diff']:+.3f} [95% CI: {boot_cov['ci_95_lower']:.3f}, {boot_cov['ci_95_upper']:.3f}] (p={boot_cov['p_value']:.4f})")
    print(f" F1 Score: {boot_f1['mean_diff']:+.3f} [95% CI: {boot_f1['ci_95_lower']:.3f}, {boot_f1['ci_95_upper']:.3f}] (p={boot_f1['p_value']:.4f})")
    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["1.5b", "3b", "7b", "12b", "32b"], default="1.5b")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_models_bootstrap(args.compare[0], args.compare[1])
    else:
        models = ["1.5b", "3b", "7b", "12b", "32b"] if args.all else [args.model]
        for m in models: run_advanced_evaluation_for_model(m)

if __name__ == "__main__":
    main()
