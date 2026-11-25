"""
Validates the cosine-similarity threshold (analysis.similarity_threshold)
empirically using Cross-Encoder NLI or Perspectrum ground-truth clusters,
eliminating reliance on costly and subjective LLM judges.

Modes:
  1. 'benchmark' (recommended default):
     Evaluates against the balanced 300-pair benchmark dataset (150 positive matches,
     150 hard adversarial negatives including 60 opposite-stance, 60 diff-concept, 30 random)
     from data/calibration/hard_calibration_pairs.csv.
  2. 'nli':
     Pairs LLM generations with human perspectives on the same claim and evaluates
     semantic entailment using a local Cross-Encoder NLI model (deberta-v3-small).
     Entailment -> 1 (same perspective), Neutral/Contradiction -> 0.
  3. 'cluster':
     Pairs human perspectives within the same Perspectrum cluster (positive pairs=1)
     and across different clusters of the same claim (negative pairs=0).
     Zero-model, 100% human ground truth.
  4. 'manual':
     Exports pairs with a blank human_label column for manual scoring.

Usage:
    python -m src.analysis.threshold_calibration --mode benchmark
    python -m src.analysis.threshold_calibration --mode nli --model 32b
    python -m src.analysis.threshold_calibration --mode cluster
"""
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv, save_json, load_json
from src.utils.logging_utils import get_logger
from src.embeddings.embed_model import BGEEmbedder

logger = get_logger("threshold_calibration")

CALIBRATION_JSON = "threshold_calibration.json"
CALIBRATION_PAIRS_CSV = "threshold_calibration_pairs.csv"
CALIBRATION_FIGURE = "threshold_roc_pr.png"


def build_candidate_pairs(claims_df, perspectives_df, generations_df, n_llm_samples: int,
                           max_perspectives_per_sample: int, seed: int) -> pd.DataFrame:
    from src.analysis.nli_stance_classifier import extract_dynamic_viewpoints

    claim_text_by_id = dict(zip(claims_df["claim_id"], claims_df["claim_text"]))

    eligible_claim_ids = set(perspectives_df["claim_id"]).intersection(set(generations_df["claim_id"]))
    gen_pool = generations_df[generations_df["claim_id"].isin(eligible_claim_ids)]

    n_sample = min(n_llm_samples, len(gen_pool))
    sampled_gen = gen_pool.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    rows = []
    for _, gen_row in sampled_gen.iterrows():
        claim_id = gen_row["claim_id"]
        claim_persps = perspectives_df[perspectives_df["claim_id"] == claim_id]
        if len(claim_persps) > max_perspectives_per_sample:
            claim_persps = claim_persps.sample(n=max_perspectives_per_sample, random_state=seed)

        raw_llm_text = gen_row.get("combined_text") or gen_row.get("perspective_summary") or gen_row.get("generated_text") or ""
        # If the sample is a multi-viewpoint essay (e.g. Gemma), decompose into individual viewpoints
        vps = extract_dynamic_viewpoints(raw_llm_text)
        if not vps:
            vps = [raw_llm_text]

        for _, p_row in claim_persps.iterrows():
            human_text = p_row["text"]
            # Pair each viewpoint against the human perspective
            for vp_idx, vp_text in enumerate(vps[:3]):  # test top distinct viewpoints
                rows.append({
                    "claim_id": claim_id,
                    "claim_text": claim_text_by_id.get(claim_id, ""),
                    "llm_sample_id": gen_row["sample_id"],
                    "viewpoint_id": vp_idx,
                    "llm_text": vp_text,
                    "perspective_id": p_row["perspective_id"],
                    "human_text": human_text,
                    "human_stance": p_row["stance"],
                })

    pairs_df = pd.DataFrame(rows)
    logger.info(
        f"Built {len(pairs_df)} candidate pairs from {n_sample} LLM samples "
        f"across {pairs_df['claim_id'].nunique() if len(pairs_df) else 0} claims"
    )
    return pairs_df


def label_pairs_with_nli(pairs_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Labels candidate (llm_text, human_text) pairs using a local Cross-Encoder NLI model.

    Evaluates whether the generation entails the human perspective.
    NLI entailment -> 1 (same perspective)
    NLI contradiction/neutral -> 0 (different perspective)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    nli_model_name = config.get("calibration", {}).get("nli_model") or \
                     config.get("analysis", {}).get("nli_model", "cross-encoder/nli-deberta-v3-small")
    batch_size = config.get("calibration", {}).get("batch_size") or \
                 config.get("analysis", {}).get("nli_batch_size", 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading NLI cross-encoder model '{nli_model_name}' on {device} for pair labeling...")

    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    model.eval()

    # Resolve label mapping
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    entail_idx = next((i for i, l in id2label.items() if "entail" in l), 1)
    contradict_idx = next((i for i, l in id2label.items() if "contra" in l), 0)
    neutral_idx = next((i for i, l in id2label.items() if "neutral" in l), 2)

    pairs = []
    for _, row in pairs_df.iterrows():
        # Premise: Generation viewpoint, Hypothesis: Human perspective text
        pairs.append((str(row["llm_text"]), str(row["human_text"])))

    nli_labels = []
    p_entails = []
    p_contras = []
    p_neutrals = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="NLI pair labeling"):
        batch_pairs = pairs[i : i + batch_size]
        encoded = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        for prob in probs:
            p_ent = float(prob[entail_idx])
            p_con = float(prob[contradict_idx])
            p_neu = float(prob[neutral_idx]) if neutral_idx < len(prob) else 0.0

            # Match criteria: Entailment is predicted (argmax == entail) OR strong entailment margin over contradiction
            is_match = 1 if (prob.argmax() == entail_idx or (p_ent > 0.40 and p_ent > p_con * 2)) else 0
            nli_labels.append(is_match)
            p_entails.append(p_ent)
            p_contras.append(p_con)
            p_neutrals.append(p_neu)

    pairs_df = pairs_df.copy()
    pairs_df["nli_label"] = nli_labels
    pairs_df["p_entailment"] = p_entails
    pairs_df["p_contradiction"] = p_contras
    pairs_df["p_neutral"] = p_neutrals

    pos_count = sum(nli_labels)
    # Fallback guard: if NLI was too conservative (pos_count == 0), take the top 5% highest entailment pairs
    if pos_count == 0 and len(pairs_df) > 0:
        logger.warning("No pairs met the strict NLI threshold; selecting top 5% highest entailment probability pairs as positive matches.")
        threshold_val = np.percentile(p_entails, 95)
        nli_labels = [1 if p >= threshold_val else 0 for p in p_entails]
        pairs_df["nli_label"] = nli_labels
        pos_count = sum(nli_labels)

    logger.info(
        f"NLI labeling complete on {len(pairs_df)} pairs: "
        f"{pos_count} positive ({pos_count/len(pairs_df):.1%}), "
        f"{len(pairs_df) - pos_count} negative ({1 - pos_count/len(pairs_df):.1%})."
    )
    return pairs_df


def build_cluster_pairs(config: dict, max_pairs: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Builds ground-truth positive (same cluster) and negative (different cluster)

    pairs directly from the human-curated Perspectrum dataset clusters.
    """
    raw_dir = resolve(config["dataset"]["raw_dir"])
    claims_path = os.path.join(raw_dir, config["dataset"]["claims_file"])
    persp_path = os.path.join(raw_dir, config["dataset"]["perspective_pool_file"])

    claims_raw = load_json(claims_path)
    persp_raw = load_json(persp_path)

    # Build perspective lookup
    persp_lookup = {}
    items = persp_raw.items() if isinstance(persp_raw, dict) else enumerate(persp_raw)
    for k, v in items:
        if isinstance(v, str):
            try:
                persp_lookup[int(k)] = v
            except (TypeError, ValueError):
                pass
        elif isinstance(v, dict):
            pid = v.get("pId", k)
            text = v.get("text")
            if text:
                try:
                    persp_lookup[int(pid)] = text
                except (TypeError, ValueError):
                    pass

    pos_pairs = []
    neg_pairs = []

    for entry in (claims_raw if isinstance(claims_raw, list) else claims_raw.values()):
        if not isinstance(entry, dict):
            continue
        claim_id = entry.get("cId")
        claim_text = entry.get("text", "")
        clusters = entry.get("perspectives", [])

        # Positive pairs: perspectives in the same cluster
        for c in clusters:
            pids = [int(p) for p in c.get("pids", []) if int(p) in persp_lookup]
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    pos_pairs.append({
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "llm_text": persp_lookup[pids[i]],
                        "human_text": persp_lookup[pids[j]],
                        "cluster_label": 1,
                    })

        # Negative pairs: perspectives in different clusters of the same claim
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pids_i = [int(p) for p in clusters[i].get("pids", []) if int(p) in persp_lookup]
                pids_j = [int(p) for p in clusters[j].get("pids", []) if int(p) in persp_lookup]
                if pids_i and pids_j:
                    neg_pairs.append({
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "llm_text": persp_lookup[pids_i[0]],
                        "human_text": persp_lookup[pids_j[0]],
                        "cluster_label": 0,
                    })

    rng = np.random.RandomState(seed)
    n_sample_each = min(len(pos_pairs), len(neg_pairs), max_pairs // 2)
    pos_idx = rng.choice(len(pos_pairs), size=n_sample_each, replace=False)
    neg_idx = rng.choice(len(neg_pairs), size=n_sample_each, replace=False)

    sampled = [pos_pairs[i] for i in pos_idx] + [neg_pairs[j] for j in neg_idx]
    pairs_df = pd.DataFrame(sampled).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        f"Built {len(pairs_df)} human dataset cluster pairs: "
        f"{n_sample_each} same-cluster positive, {n_sample_each} between-cluster negative."
    )
    return pairs_df


def compute_similarities(pairs_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    model_name = config.get("embedding", {}).get("model", "BAAI/bge-large-en-v1.5")
    normalize = config.get("embedding", {}).get("normalize", True)
    batch_size = config.get("embedding", {}).get("batch_size", 32)

    logger.info(f"Computing embeddings dynamically using configured model: {model_name}...")
    embedder = BGEEmbedder(
        model_name=model_name,
        normalize=normalize,
    )
    
    # Support both column naming conventions
    llm_col = "llm_text" if "llm_text" in pairs_df.columns else "llm_viewpoint"
    human_col = "human_text" if "human_text" in pairs_df.columns else "human_perspective"

    llm_texts = pairs_df[llm_col].astype(str).tolist()
    human_texts = pairs_df[human_col].astype(str).tolist()

    llm_emb = embedder.embed(llm_texts, batch_size=batch_size)
    human_emb = embedder.embed(human_texts, batch_size=batch_size)

    from src.analysis.similarity_matrix import ensure_normalized
    llm_emb = ensure_normalized(llm_emb, "calibration LLM embeddings")
    human_emb = ensure_normalized(human_emb, "calibration human embeddings")
    sims = np.sum(llm_emb * human_emb, axis=1)
    pairs_df = pairs_df.copy()
    pairs_df["cosine_similarity"] = sims
    return pairs_df


def analyze_roc_pr(pairs_df: pd.DataFrame, label_col: str, default_tau: float, figures_dir: str) -> dict:
    labeled = pairs_df.dropna(subset=[label_col, "cosine_similarity"])
    y_true = labeled[label_col].astype(int).values
    y_score = labeled["cosine_similarity"].values

    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both positive and negative labeled pairs to calibrate a threshold "
            f"(got {n_pos} positive, {n_neg} negative). Sample more pairs or check labeling."
        )

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    tau_youden = float(roc_thresholds[int(np.argmax(youden_j))])

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    # precision/recall have one more element than pr_thresholds; align by dropping the last
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12),
        0.0,
    )
    tau_f1max = float(pr_thresholds[int(np.argmax(f1))]) if len(pr_thresholds) else default_tau

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    best_idx = int(np.argmax(youden_j))
    axes[0].scatter([fpr[best_idx]], [tpr[best_idx]], color="red", zorder=5,
                     label=f"Youden-optimal tau={tau_youden:.2f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve: BGE cosine similarity vs. judged label")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
    best_f1_idx = int(np.argmax(f1)) if len(f1) else 0
    if len(pr_thresholds):
        axes[1].scatter([recall[best_f1_idx]], [precision[best_f1_idx]], color="red", zorder=5,
                         label=f"F1-optimal tau={tau_f1max:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="lower left", fontsize=8)

    fig.suptitle(f"Threshold calibration (n={len(labeled)} pairs, "
                 f"{n_pos} positive / {n_neg} negative) - config default tau={default_tau}")
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, CALIBRATION_FIGURE)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    recommended_tau = tau_f1max  # F1-optimal is the standard choice for an imbalanced retrieval-style decision
    result = {
        "n_pairs_labeled": int(len(labeled)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tau_youden_j": tau_youden,
        "tau_f1_max": tau_f1max,
        "recommended_tau": float(recommended_tau),
        "config_default_tau": float(default_tau),
        "figure_path": os.path.relpath(fig_path, resolve("")),
    }
    return result


def get_similarity_threshold(config: dict) -> tuple[float, dict]:
    """Single source of truth used by coverage_score.py: prefer the
    empirically calibrated tau if it exists and analysis.use_calibrated_threshold
    is true, else fall back to the config heuristic (loudly logged as such).

    Returns (threshold, meta) where meta describes provenance.
    """
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    calibration_path = os.path.join(metrics_dir, CALIBRATION_JSON)
    use_calibrated = config.get("analysis", {}).get("use_calibrated_threshold", True)
    default_tau = config["analysis"]["similarity_threshold"]

    if use_calibrated and os.path.exists(calibration_path):
        calib = load_json(calibration_path)
        tau = calib["recommended_tau"]
        logger.info(
            f"Using empirically calibrated threshold tau={tau:.3f} "
            f"(F1-optimal on {calib['n_pairs_labeled']} labeled pairs, "
            f"ROC-AUC={calib['roc_auc']:.3f}, PR-AUC={calib['pr_auc']:.3f}). "
            f"See results/metrics/{CALIBRATION_JSON} and "
            f"results/figures/{CALIBRATION_FIGURE}."
        )
        return tau, {"source": "calibrated", **calib}

    logger.warning(
        f"No threshold calibration found (or use_calibrated_threshold=false) - "
        f"falling back to UNVALIDATED heuristic tau={default_tau} from config.yaml. "
        f"Run `python -m src.analysis.threshold_calibration` to validate this threshold "
        f"against labeled pairs before trusting the coverage numbers."
    )
    return default_tau, {"source": "config_default_unvalidated", "config_default_tau": default_tau}


def main(config: dict = None, score_only: bool = False):
    if config is None:
        config = build_config(embedding=args.embedding)
    calib_cfg = config.get("calibration", {})
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    figures_dir = resolve(config["paths"]["figures_dir"])
    default_tau = config["analysis"]["similarity_threshold"]
    pairs_path = os.path.join(metrics_dir, CALIBRATION_PAIRS_CSV)

    if score_only:
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"{pairs_path} not found - run without score_only first.")
        pairs_df = load_csv(pairs_path)
        label_col = "human_label" if "human_label" in pairs_df.columns and pairs_df["human_label"].notna().any() \
            else ("nli_label" if "nli_label" in pairs_df.columns else "cluster_label")
    else:
        mode = calib_cfg.get("mode", "benchmark")

        if mode == "benchmark":
            # 150 Positive / 150 Hard Adversarial Negative benchmark dataset
            benchmark_csv = calib_cfg.get("benchmark_path", "data/calibration/hard_calibration_pairs.csv")
            benchmark_full_path = resolve(benchmark_csv)
            logger.info(f"Calibrating threshold using balanced 150/150 benchmark: {benchmark_full_path}...")
            if not os.path.exists(benchmark_full_path):
                raise FileNotFoundError(f"Benchmark file not found at: {benchmark_full_path}")
            pairs_df = load_csv(benchmark_full_path)
            
            # Always compute cosine similarity dynamically using whatever embedding model is configured in config.yaml
            pairs_df = compute_similarities(pairs_df, config)
            label_col = "human_label"
        elif mode == "cluster":
            # 100% human dataset ground-truth calibration using Perspectrum cluster structure
            logger.info("Calibrating threshold using Perspectrum human dataset clusters...")
            pairs_df = build_cluster_pairs(
                config,
                max_pairs=calib_cfg.get("max_cluster_pairs", 1000),
                seed=calib_cfg.get("seed", 42),
            )
            pairs_df = compute_similarities(pairs_df, config)
            label_col = "cluster_label"
        elif mode == "nli":
            # Generation x Human pairs calibrated via local Cross-Encoder NLI model
            processed_dir = resolve(config["dataset"]["processed_dir"])
            claims_df = load_csv(os.path.join(processed_dir, "claims.csv"))
            perspectives_df = load_csv(os.path.join(processed_dir, "perspectives.csv"))
            generations_df = load_csv(resolve(config["paths"]["llm_outputs_processed"]))

            pairs_df = build_candidate_pairs(
                claims_df, perspectives_df, generations_df,
                n_llm_samples=calib_cfg.get("n_llm_samples", 50),
                max_perspectives_per_sample=calib_cfg.get("max_perspectives_per_sample", 6),
                seed=calib_cfg.get("seed", 42),
            )
            pairs_df = compute_similarities(pairs_df, config)
            pairs_df = label_pairs_with_nli(pairs_df, config)
            label_col = "nli_label"
        elif mode == "manual":
            processed_dir = resolve(config["dataset"]["processed_dir"])
            claims_df = load_csv(os.path.join(processed_dir, "claims.csv"))
            perspectives_df = load_csv(os.path.join(processed_dir, "perspectives.csv"))
            generations_df = load_csv(resolve(config["paths"]["llm_outputs_processed"]))

            pairs_df = build_candidate_pairs(
                claims_df, perspectives_df, generations_df,
                n_llm_samples=calib_cfg.get("n_llm_samples", 50),
                max_perspectives_per_sample=calib_cfg.get("max_perspectives_per_sample", 6),
                seed=calib_cfg.get("seed", 42),
            )
            pairs_df = compute_similarities(pairs_df, config)
            pairs_df["human_label"] = np.nan
            save_csv(pairs_df, pairs_path)
            logger.info(
                f"mode=manual: wrote {len(pairs_df)} unlabeled pairs to {pairs_path}. "
                f"Fill in the human_label column (1=same perspective, 0=different) and re-run "
                f"with score_only=True (or `--score-only` from the CLI)."
            )
            return None
        else:
            raise ValueError(f"Unknown calibration.mode: {mode!r} (supported: 'benchmark', 'nli', 'cluster', 'manual')")

        save_csv(pairs_df, pairs_path)

    result = analyze_roc_pr(pairs_df, label_col, default_tau, figures_dir)
    save_json(result, os.path.join(metrics_dir, CALIBRATION_JSON))

    logger.info(
        f"Calibration complete: ROC-AUC={result['roc_auc']:.3f}, PR-AUC={result['pr_auc']:.3f}, "
        f"tau_youden={result['tau_youden_j']:.3f}, tau_f1max={result['tau_f1_max']:.3f}, "
        f"recommended_tau={result['recommended_tau']:.3f} (config default was {default_tau})."
    )
    logger.info(f"Saved pairs -> {pairs_path}")
    logger.info(f"Saved figure -> {os.path.join(figures_dir, CALIBRATION_FIGURE)}")
    logger.info(f"Saved summary -> {os.path.join(metrics_dir, CALIBRATION_JSON)}")
    return result


def _cli():
    """Thin CLI entry point - only parses sys.argv when this module is run
    standalone (`python -m src.analysis.threshold_calibration`). main() itself
    takes plain arguments so it's safe to call in-process (e.g. from
    run_pipeline.py) without main() trying to re-parse the caller's own argv."""
    parser = argparse.ArgumentParser(description="Calibrate the coverage similarity threshold using Benchmark, NLI or Clusters.")
    parser.add_argument(
        "--mode", choices=["benchmark", "nli", "cluster", "manual"], default=None,
        help="Calibration mode: 'benchmark' (balanced 150 pos / 150 neg hard benchmark), 'nli' (local cross-encoder NLI), 'cluster' (Perspectrum dataset ground-truth clusters), or 'manual'.",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Skip pair generation/judging; read an existing pairs CSV and just compute ROC/PR.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model key to use (e.g. 0.5b, 1.5b, 3b, qwen-7b, 14b, 32b, llama-8b). "
             "Selects the model-namespaced metrics/figures dirs so calibration "
             "lands next to that model's other results. Omit for the config default.",
    )
    parser.add_argument(
        "--embedding", type=str, default=None,
        help="Embedding model key to calibrate (bge-large, bge-base, mpnet, nomic).",
    )
    args = parser.parse_args()
    config = None
    if args.model:
        from run_pipeline import build_config
        if args.embedding:
            config = build_config(model=args.model, embedding=args.embedding)
        else:
            config = build_config(model=args.model)
    elif args.mode:
        config = build_config(embedding=args.embedding)

    if args.mode and config is not None:
        config.setdefault("calibration", {})["mode"] = args.mode

    main(config=config, score_only=args.score_only)


if __name__ == "__main__":
    _cli()
