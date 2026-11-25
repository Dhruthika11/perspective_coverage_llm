"""
Builds a Stratified Universal Calibration Dataset across all models and datasets.
Pairs LLM generations (from small, medium, and large models across families)
with human gold perspectives from Perspectrum, Kialo, and IBM Debater.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.embeddings.embed_model import BGEEmbedder

logger = get_logger("build_universal_calibration")


def find_generation_file(dataset_name: str, model_tag: str) -> str:
    """Finds the processed_generations.csv file for a given dataset and model."""
    candidates = [
        resolve(f"data/llm_outputs/{dataset_name}/{model_tag}/processed_generations.csv"),
        resolve(f"data/llm_outputs/{dataset_name}/processed_generations.csv"),
        resolve(f"data/llm_outputs/{model_tag}/processed_generations.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def sample_pairs_for_dataset(
    dataset_name: str,
    config: dict,
    embedder: BGEEmbedder,
    model_keys: list,
    n_pairs_target: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Extracts stratified candidate pairs for one dataset across diverse models."""
    datasets_cfg = config.get("dataset", {}).get("datasets", {})
    ds_info = datasets_cfg.get(dataset_name, {})
    processed_dir = resolve(ds_info.get("processed_dir", f"data/processed/{dataset_name}/"))

    claims_path = os.path.join(processed_dir, "claims.csv")
    persp_path = os.path.join(processed_dir, "perspectives.csv")

    if not os.path.exists(claims_path) or not os.path.exists(persp_path):
        logger.warning(f"Claims or perspectives missing for {dataset_name} at {processed_dir}")
        return pd.DataFrame()

    claims_df = load_csv(claims_path)
    persp_df = load_csv(persp_path)

    # Normalize column names across datasets
    text_col = "perspective_text" if "perspective_text" in persp_df.columns else "text"
    persp_id_col = "perspective_id" if "perspective_id" in persp_df.columns else "pId"

    claim_text_lookup = dict(zip(claims_df["claim_id"], claims_df["claim_text"]))

    collected_pairs = []

    for m_key in model_keys:
        from run_pipeline import build_config
        from src.utils.io_utils import get_model_tag

        sub_cfg = build_config(model=m_key)
        model_name = sub_cfg["llm"]["model"]
        model_tag = get_model_tag(model_name)

        gen_path = find_generation_file(dataset_name, model_tag)
        if not gen_path:
            continue

        gen_df = load_csv(gen_path)
        if gen_df.empty:
            continue

        common_claims = list(set(gen_df["claim_id"]).intersection(set(persp_df["claim_id"])))
        if not common_claims:
            continue

        np.random.seed(seed)
        sampled_claims = np.random.choice(common_claims, size=min(15, len(common_claims)), replace=False)

        for cid in sampled_claims:
            c_text = claim_text_lookup.get(cid, "")
            c_gens = gen_df[gen_df["claim_id"] == cid]
            c_persps = persp_df[persp_df["claim_id"] == cid]

            if c_gens.empty or c_persps.empty:
                continue

            gen_texts = c_gens["combined_text"].dropna().tolist()
            if not gen_texts:
                gen_texts = c_gens.get("perspective_summary", c_gens.get("generated_text")).dropna().tolist()

            p_texts = c_persps[text_col].dropna().tolist()
            p_ids = c_persps[persp_id_col].tolist()

            if not gen_texts or not p_texts:
                continue

            gen_embs = embedder.embed(gen_texts)
            p_embs = embedder.embed(p_texts)
            sim_mat = np.dot(gen_embs, p_embs.T)

            for g_i, g_t in enumerate(gen_texts):
                sims = sim_mat[g_i]
                sorted_idx = np.argsort(sims)[::-1]
                
                # Best match
                best_p = sorted_idx[0]
                collected_pairs.append({
                    "dataset": dataset_name,
                    "model_evaluated": m_key,
                    "claim_id": cid,
                    "claim_text": c_text,
                    "llm_viewpoint": g_t,
                    "perspective_id": p_ids[best_p],
                    "human_perspective": p_texts[best_p],
                    "cosine_similarity": round(float(sims[best_p]), 4),
                    "human_label": ""
                })

                # Borderline candidate
                if len(sorted_idx) > 1:
                    border_p = sorted_idx[min(2, len(sorted_idx) - 1)]
                    collected_pairs.append({
                        "dataset": dataset_name,
                        "model_evaluated": m_key,
                        "claim_id": cid,
                        "claim_text": c_text,
                        "llm_viewpoint": g_t,
                        "perspective_id": p_ids[border_p],
                        "human_perspective": p_texts[border_p],
                        "cosine_similarity": round(float(sims[border_p]), 4),
                        "human_label": ""
                    })

    if not collected_pairs:
        return pd.DataFrame()

    df = pd.DataFrame(collected_pairs)
    df = df.drop_duplicates(subset=["dataset", "claim_id", "llm_viewpoint", "human_perspective"])

    if len(df) > n_pairs_target:
        df["sim_bin"] = pd.qcut(df["cosine_similarity"], q=4, labels=False, duplicates="drop")
        df = df.groupby("sim_bin", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_pairs_target // 4 + 1), random_state=seed)
        ).head(n_pairs_target).drop(columns=["sim_bin"])

    return df


def main():
    parser = argparse.ArgumentParser(description="Build universal calibration pairs across datasets and models.")
    parser.add_argument("--pairs-per-dataset", type=int, default=100, help="Number of candidate pairs per dataset.")
    parser.add_argument("--out-dir", type=str, default="data/calibration", help="Directory to save the pairs.")
    args = parser.parse_args()

    config = load_config()
    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    emb_model = config["embedding"]["model"]
    logger.info(f"Initializing embedder for calibration: {emb_model}")
    embedder = BGEEmbedder(model_name=emb_model, normalize=True)

    # Diverse representative models spanning small, medium, and large
    representative_models = ["3b", "7b", "llama-8b", "32b", "gemma-9b"]

    all_dfs = []
    datasets = ["perspectrum", "kialo", "ibm_debater"]

    for ds in datasets:
        logger.info(f"--- Sampling pairs for {ds} ---")
        ds_df = sample_pairs_for_dataset(
            dataset_name=ds,
            config=config,
            embedder=embedder,
            model_keys=representative_models,
            n_pairs_target=args.pairs_per_dataset,
        )
        if not ds_df.empty:
            logger.info(f"Extracted {len(ds_df)} candidate pairs for {ds}")
            all_dfs.append(ds_df)
        else:
            logger.warning(f"Could not extract pairs for {ds}. Are its generations/processed data available?")

    if not all_dfs:
        logger.error("No pairs could be extracted. Please check your data folders.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by="cosine_similarity", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(out_dir, "universal_calibration_pairs.csv")
    out_json = os.path.join(out_dir, "universal_calibration_pairs.json")

    save_csv(combined_df, out_csv)
    save_json(combined_df.to_dict(orient="records"), out_json)

    logger.info("==================================================================")
    logger.info(f"SUCCESS: Generated {len(combined_df)} total universal calibration pairs!")
    logger.info(f"Saved to: {out_csv}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
