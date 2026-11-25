"""
Multi-Model x Multi-Embedding Pipeline Runner.
Runs:
  1. Embeddings (perspectives + LLM outputs)
  2. Analysis (coverage, diversity, distribution divergence, stance crosscheck)
across all 12 LLMs and 4 Embedding Models.

Usage:
  # Run a single model with a single embedding model:
  python -m src.analysis.run_matrix_embeddings_and_analysis --model 7b --embedding bge-large

  # Run all available models for all 4 embedding models:
  python -m src.analysis.run_matrix_embeddings_and_analysis --all

  # Run only embeddings stage:
  python -m src.analysis.run_matrix_embeddings_and_analysis --all --stage embeddings

  # Run only analysis stage for already-embedded models:
  python -m src.analysis.run_matrix_embeddings_and_analysis --all --stage analysis
"""

import os
import sys
import argparse
import subprocess
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logging_utils import get_logger
from src.utils.io_utils import load_config, resolve

logger = get_logger("matrix_runner")

ALL_EMBEDDING_KEYS = ["bge-large", "bge-base", "mpnet", "nomic"]

# 12-Model Scaling Ladder Keys from configs/config.yaml
ALL_MODEL_KEYS = [
    # Qwen 2.5 Family
    "0.5b", "1.5b", "3b", "qwen-7b", "14b", "32b",
    # LLaMA 3.x Family
    "llama-1b", "llama-3b", "llama-8b",
    # Google Gemma Family
    "gemma-2b", "gemma-9b",
    # SmolLM2 Family (Sub-Billion)
    "smollm2-135m", "smollm2-360m", "smollm2-1.7b",
    # OLMoE Sparse MoE (1B-7B)
    "olmoe-1b-7b",
    # OLMo 2 Alignment Ladder
    "olmo-1b", "olmo2-7b-base", "olmo2-7b-sft", "olmo2-7b-dpo", "olmo2-7b-instruct", "olmo2-13b-instruct"
]


def check_generations_exist(model_key: str, config: dict) -> bool:
    """Check if processed_generations.csv exists for this model."""
    from run_pipeline import build_config
    cfg = build_config(model=model_key)
    processed_path = resolve(cfg["paths"]["llm_outputs_processed"])
    return os.path.exists(processed_path)


def run_command_for_pair(model_key: str, emb_key: str, stage: str):
    """Executes run_pipeline.py for the given model, embedding, and stage."""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "run_pipeline.py"),
        "--model", model_key,
        "--embedding", emb_key,
        "--stage", stage
    ]
    logger.info(f"Running [{stage.upper()}] for Model='{model_key}' with Embedding='{emb_key}'...")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Matrix runner for 12 LLMs x 4 Embedding Models.")
    parser.add_argument("--model", type=str, default=None, help="Specific model key (e.g. 7b, 1.5b, 32b, 12b).")
    parser.add_argument("--embedding", type=str, default=None, help="Specific embedding key (bge-large, bge-base, mpnet, nomic).")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model keys.")
    parser.add_argument("--embeddings", type=str, default=None, help="Comma-separated list of embedding keys.")
    parser.add_argument("--all", action="store_true", help="Run for all 12 models and all 4 embedding models.")
    parser.add_argument("--stage", choices=["embeddings", "analysis", "both"], default="both",
                        help="Stage to execute: embeddings, analysis, or both (default).")

    args = parser.parse_args()

    # Determine target models
    if args.model:
        target_models = [args.model]
    elif args.models:
        target_models = [m.strip() for m in args.models.split(",")]
    elif args.all:
        target_models = ALL_MODEL_KEYS
    else:
        # Default to models that actually have generations on disk
        target_models = []
        base_cfg = load_config()
        for m in ALL_MODEL_KEYS:
            if check_generations_exist(m, base_cfg):
                target_models.append(m)
        if not target_models:
            target_models = ["7b"]
        logger.info(f"Auto-detected models with existing generations: {target_models}")

    # Determine target embeddings
    if args.embedding:
        target_embeddings = [args.embedding]
    elif args.embeddings:
        target_embeddings = [e.strip() for e in args.embeddings.split(",")]
    elif args.all:
        target_embeddings = ALL_EMBEDDING_KEYS
    else:
        target_embeddings = ["bge-large"]

    logger.info("=" * 80)
    logger.info("MATRIX RUNNER PLAN")
    logger.info(f"Target Models ({len(target_models)}): {target_models}")
    logger.info(f"Target Embeddings ({len(target_embeddings)}): {target_embeddings}")
    logger.info(f"Stage: {args.stage}")
    logger.info("=" * 80)

    base_cfg = load_config()
    summary_results = []

    for m_key in target_models:
        has_gen = check_generations_exist(m_key, base_cfg)
        if not has_gen:
            logger.warning(f"[-] Skipping model '{m_key}': processed_generations.csv not found.")
            for e_key in target_embeddings:
                summary_results.append({
                    "model": m_key, "embedding": e_key, "status": "SKIPPED (no generations)"
                })
            continue

        for e_key in target_embeddings:
            logger.info(f"\n>>> Processing Pair: Model=[{m_key}] | Embedding=[{e_key}] <<<")
            
            # Step 1: Embeddings
            if args.stage in ["embeddings", "both"]:
                ok_emb = run_command_for_pair(m_key, e_key, "embeddings")
                if not ok_emb:
                    logger.error(f"Failed embeddings for {m_key} with {e_key}.")
                    summary_results.append({
                        "model": m_key, "embedding": e_key, "status": "FAILED (embeddings)"
                    })
                    continue

            # Step 2: Analysis
            if args.stage in ["analysis", "both"]:
                ok_ana = run_command_for_pair(m_key, e_key, "analysis")
                status = "SUCCESS" if ok_ana else "FAILED (analysis)"
                summary_results.append({
                    "model": m_key, "embedding": e_key, "status": status
                })
            else:
                summary_results.append({
                    "model": m_key, "embedding": e_key, "status": "EMBEDDINGS ONLY (OK)"
                })

    print("\n" + "=" * 80)
    print("                     EXECUTION SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
