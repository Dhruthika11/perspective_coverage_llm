"""
Perspective Generator Engine (Phase 2).

Executes generation under controlled prompt conditions (P1, P2, P3, P4)
with full logging, abstention tracking, and strict provenance preservation.

Output directory structure:
  data/llm_outputs/{model_tag}/{prompt_version}/
    - raw_generations.jsonl      (Full detailed trace of every attempt)
    - processed_generations.csv  (Normalized tabular output ready for embedding)
    - run_manifest.json          (Hyperparameters, seeds, model revision, git hash)
"""
import os
import sys
import time
import json
import random
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv, get_model_tag
from src.utils.logging_utils import get_logger
from src.generation.llm_config import get_llm_client
from src.generation.prompt_templates import (
    STANDARD_ASPECTS,
    build_prompt,
    parse_generation_output,
)

logger = get_logger("perspective_generator")


def generate_perspectives_for_dataset(
    config: Optional[dict] = None,
    prompt_version: str = "p2",
    k_budget: int = 5,
    max_claims: Optional[int] = None,
    dataset_name: str = "perspectrum",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Executes controlled perspective generation for a dataset.
    
    Args:
        config: Loaded config dictionary.
        prompt_version: One of 'p1', 'p2', 'p3', 'p4'.
        k_budget: Number of generation steps per claim (default 5).
        max_claims: If set, limits generation to first N claims (useful for testing/subsets).
        dataset_name: Dataset to generate for ('perspectrum', 'ibm_debater', 'kialo').
        dry_run: If True, builds prompts and validates structure without calling LLM.
    """
    if config is None:
        config = load_config()

    prompt_version = prompt_version.lower().strip()
    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
    model_tag = get_model_tag(model_name)
    temperature = float(llm_cfg.get("temperature", 0.8))
    top_p = float(llm_cfg.get("top_p", 0.95))
    max_new_tokens = int(llm_cfg.get("max_new_tokens", 512))
    seed = int(llm_cfg.get("seed", 42))

    # Dedicated versioned output directory
    out_dir = os.path.join("data", "llm_outputs", model_tag, prompt_version)
    os.makedirs(out_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(out_dir, "raw_generations.jsonl")
    processed_csv_path = os.path.join(out_dir, "processed_generations.csv")
    manifest_path = os.path.join(out_dir, "run_manifest.json")

    # Load claims
    if dataset_name == "perspectrum":
        claims_path = os.path.join("data", "processed", "claims.csv")
    elif dataset_name == "ibm_debater":
        claims_path = os.path.join("data", "processed", "ibm_debater", "claims.csv")
    elif dataset_name == "kialo":
        claims_path = os.path.join("data", "processed", "kialo", "claims.csv")
    else:
        claims_path = os.path.join(resolve(config["dataset"]["processed_dir"]), "claims.csv")

    if not os.path.exists(claims_path):
        raise FileNotFoundError(f"Claims file not found: {claims_path}. Run dataset adapter first.")

    claims_df = load_csv(claims_path)
    if max_claims:
        claims_df = claims_df.iloc[:max_claims]

    logger.info(
        f"Initializing Perspective Generator: Model={model_tag}, Prompt={prompt_version.upper()}, "
        f"K={k_budget}, Claims={len(claims_df)}, DryRun={dry_run}"
    )

    # Save initial run manifest
    manifest = {
        "model_name": model_name,
        "model_tag": model_tag,
        "prompt_version": prompt_version,
        "k_budget": k_budget,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "dataset_name": dataset_name,
        "total_claims": len(claims_df),
        "timestamp_start": time.time(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if dry_run:
        logger.info("Dry-run mode enabled: Verifying prompt construction without calling LLM...")
        sample_claim = str(claims_df.iloc[0]["claim_text"])
        sample_prompt = build_prompt(prompt_version, sample_claim, aspect=STANDARD_ASPECTS[0], history=[])
        logger.info(f"Sample Prompt ({prompt_version.upper()}):\n{sample_prompt[:400]}...\n")
        return pd.DataFrame()

    # Load client
    client = get_llm_client(config)

    # Resume support: check completed claim IDs
    done_claim_ids = set()
    if os.path.exists(raw_jsonl_path):
        with open(raw_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("step") == k_budget - 1:
                        done_claim_ids.add(obj["claim_id"])
                except Exception:
                    pass
        logger.info(f"Resuming: found {len(done_claim_ids)} already completed claims.")

    remaining_df = claims_df[~claims_df["claim_id"].isin(done_claim_ids)]
    if len(remaining_df) == 0:
        logger.info("All claims already generated! Returning existing processed file.")
        return pd.read_csv(processed_csv_path) if os.path.exists(processed_csv_path) else pd.DataFrame()

    # Generation loop
    all_processed_records = []
    
    # Load previously processed records if resuming
    if os.path.exists(processed_csv_path):
        prev_df = pd.read_csv(processed_csv_path)
        all_processed_records.extend(prev_df.to_dict(orient="records"))

    for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc=f"Generating {prompt_version.upper()}"):
        claim_id = int(row["claim_id"])
        claim_text = str(row["claim_text"])
        claim_history = []

        for step in range(k_budget):
            aspect = STANDARD_ASPECTS[step % len(STANDARD_ASPECTS)]
            prompt = build_prompt(prompt_version, claim_text, aspect=aspect, history=claim_history)

            start_t = time.time()
            try:
                samples = client.generate_k_samples(
                    prompt,
                    k=1,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                raw_output = samples[0] if samples else ""
                error_info = None
            except Exception as e:
                raw_output = ""
                error_info = str(e)
                logger.error(f"Error on Claim {claim_id}, Step {step}: {e}")

            latency = round(time.time() - start_t, 3)

            # Parse structured output
            parsed = parse_generation_output(raw_output, prompt_version=prompt_version)

            # Update running history if valid argument was produced
            if parsed["is_valid"] and not parsed["is_abstention"]:
                claim_history.append({
                    "stance": parsed["stance"],
                    "reasoning_criterion": parsed.get("reasoning_criterion", "none"),
                    "argument": parsed["argument"],
                })

            # Record detailed log object (Section 6.5 requirements)
            log_record = {
                "claim_id": claim_id,
                "model_id": model_name,
                "prompt_version": prompt_version,
                "aspect": aspect if prompt_version != "p4" else parsed.get("aspect", "open"),
                "generation_step": step,
                "previous_history": [h["argument"] for h in claim_history[:-1]] if claim_history else [],
                "temperature": temperature,
                "random_seed": seed + step,
                "token_budget": max_new_tokens,
                "raw_model_output": raw_output,
                "parsed_output": parsed,
                "is_valid": parsed["is_valid"],
                "stance": parsed["stance"],
                "reasoning_criterion": parsed.get("reasoning_criterion", "none"),
                "is_abstention": parsed["is_abstention"],
                "generation_time_sec": latency,
                "error_information": error_info,
            }

            # Append to raw jsonl
            with open(raw_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\n")

            # Append to tabular processed outputs (if not abstained and not empty)
            if parsed["is_valid"] and not parsed["is_abstention"]:
                all_processed_records.append({
                    "claim_id": claim_id,
                    "sample_id": step,
                    "viewpoint_id": 0,
                    "aspect": log_record["aspect"],
                    "reasoning_criterion": parsed.get("reasoning_criterion", "none"),
                    "predicted_stance": parsed["stance"],
                    "combined_text": parsed["argument"],
                    "parse_ok": True,
                    "prompt_version": prompt_version,
                    "dataset": dataset_name,
                })

        # Periodically flush processed CSV
        if len(all_processed_records) % 10 == 0:
            pd.DataFrame(all_processed_records).to_csv(processed_csv_path, index=False)

    final_df = pd.DataFrame(all_processed_records)
    final_df.to_csv(processed_csv_path, index=False)
    logger.info(f"Completed generation: Saved {len(final_df)} valid viewpoints to {processed_csv_path}")
    return final_df

