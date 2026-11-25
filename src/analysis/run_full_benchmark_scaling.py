"""
Master Scaling Runner for Remaining 1,100 Benchmark Pairs
Orchestrates the frozen Multi-LLM Jury Committee:
- Judge 1: Google Gemini 3.1 Flash Lite Preview
- Judge 2: OpenAI GPT-OSS (120B primary / 20B fallback)
- Judge 3: Alibaba Qwen 3.8-27B
Checkpoints each batch, merges with Stages A, B, and C into the final 1,700 benchmark,
and exports the targeted human triage queue and stratified unanimous audit.
"""

import os
import sys
import time
import json
import re
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

from src.analysis.run_live_llm_jury import (
    SYSTEM_INSTRUCTION,
    USER_BATCH_TEMPLATE,
    clean_json_response,
    GeminiJudgeClient
)

def evaluate_groq_batch(
    batch_pairs: List[Dict[str, Any]],
    model_name: str,
    api_keys: List[str],
    fallback_model: str = "openai/gpt-oss-20b",
    max_parse_retries: int = 5
) -> Dict[str, Any]:
    """Robust Groq evaluator with TPM backoff and TPD fallback."""
    pairs_text = "\n\n".join([
        f"[Pair ID: {p['pair_id']}]\nDebate Claim: \"{p['claim_text']}\"\nGold Perspective: \"{p['gold_perspective_text']}\"\nGenerated Viewpoint: \"{p['generated_viewpoint_text']}\""
        for p in batch_pairs
    ])
    user_prompt = USER_BATCH_TEMPLATE.format(batch_size=len(batch_pairs), pairs_text=pairs_text)
    expected_pids = {int(p["pair_id"]) for p in batch_pairs}
    
    models_to_try = [model_name]
    if "gpt-oss" in model_name and fallback_model and fallback_model != model_name:
        models_to_try.append(fallback_model)
        
    for cur_model in models_to_try:
        payload = {
            "model": cur_model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 3500
        }
        
        parse_attempts = 0
        rate_limit_attempts = 0
        
        while parse_attempts < max_parse_retries and rate_limit_attempts < 20:
            key = api_keys[(parse_attempts + rate_limit_attempts) % len(api_keys)]
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            t0 = time.time()
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=45
                )
                latency_ms = int((time.time() - t0) * 1000)
                
                if resp.status_code == 200:
                    resp_json = resp.json()
                    raw_text = resp_json["choices"][0]["message"].get("content", "")
                    usage = resp_json.get("usage", {})
                    parsed = clean_json_response(raw_text)
                    
                    returned_pids = {int(item.get("pair_id", -1)) for item in parsed}
                    missing_pids = expected_pids - returned_pids
                    
                    if missing_pids:
                        parse_attempts += 1
                        print(f"    Missing pair IDs {missing_pids} from {cur_model}, retrying (parse attempt {parse_attempts}/{max_parse_retries})...")
                        time.sleep(2)
                        continue
                        
                    return {
                        "success": True,
                        "results": parsed,
                        "latency_ms": latency_ms,
                        "model_used": cur_model,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0)
                    }
                elif resp.status_code == 429:
                    err_msg = resp.text
                    rate_limit_attempts += 1
                    
                    m_wait = re.search(r"try again in (?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)s", err_msg)
                    wait_sec = 15
                    if m_wait:
                        mins = float(m_wait.group(1)) if m_wait.group(1) else 0.0
                        secs = float(m_wait.group(2)) if m_wait.group(2) else 0.0
                        wait_sec = int(mins * 60 + secs) + 3
                        
                    if "tokens per day (TPD)" in err_msg:
                        if wait_sec <= 240:
                            print(f"    Model {cur_model} TPD cooldown is short ({wait_sec}s). Waiting...")
                            time.sleep(wait_sec)
                            continue
                        else:
                            print(f"    Model {cur_model} hit daily TPD quota (cooldown {wait_sec}s). Trying next model...")
                            break
                    else:
                        print(f"    429 TPM on {cur_model}. Waiting {wait_sec}s (retry {rate_limit_attempts}/20)...")
                        time.sleep(wait_sec)
                else:
                    parse_attempts += 1
                    print(f"    HTTP {resp.status_code} on {cur_model}: {resp.text[:120]}")
                    time.sleep(3)
            except Exception as e:
                parse_attempts += 1
                print(f"    Exception calling {cur_model}: {e}")
                time.sleep(3)
                
    raise RuntimeError(f"Failed to evaluate batch with any Groq model: {expected_pids}")

def main():
    load_dotenv()
    gem_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    k1 = os.getenv("GROQ_API_KEY")
    k2 = os.getenv("GROQ_API_KEY_BACKUP")
    groq_keys = [k for k in [k1, k2] if k]
    
    if not gem_key:
        raise ValueError("Missing GEMINI_API_KEY in environment.")
    if not groq_keys:
        raise ValueError("Missing GROQ_API_KEY in environment.")
        
    input_csv = "data/calibration/dataset_b_remaining_1100.csv"
    output_csv = "data/calibration/llm_jury_remaining_1100_annotated.csv"
    
    df_pool = pd.read_csv(input_csv)
    print(f"Loaded {len(df_pool)} remaining unannotated pairs from {input_csv}")
    
    # Check if checkpoint exists
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        completed_pids = set(df_existing["pair_id"])
        print(f"Found existing checkpoint with {len(completed_pids)} completed pairs.")
    else:
        df_existing = pd.DataFrame()
        completed_pids = set()
        
    df_to_run = df_pool[~df_pool["pair_id"].isin(completed_pids)].copy().reset_index(drop=True)
    print(f"Pairs to annotate in this run: {len(df_to_run)}")
    
    if len(df_to_run) == 0:
        print("All 1,100 pairs are already completed!")
    else:
        batch_size = 5
        records = df_to_run.to_dict(orient="records")
        batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
        print(f"Partitioned into {len(batches)} batches of {batch_size} pairs.")
        
        gemini_client = GeminiJudgeClient(api_key=gem_key, model_name="gemini-3.1-flash-lite-preview")
        
        # Load or init completed dict
        completed_dict = {row["pair_id"]: row for row in df_existing.to_dict(orient="records")} if not df_existing.empty else {}
        
        for b_idx, batch in enumerate(batches, 1):
            pids = [p["pair_id"] for p in batch]
            print(f"\n--- Processing Batch {b_idx}/{len(batches)} (Pairs: {pids}) ---")
            
            # Init rows
            for p in batch:
                if p["pair_id"] not in completed_dict:
                    completed_dict[p["pair_id"]] = {
                        "pair_id": p["pair_id"],
                        "claim_id": p["claim_id"],
                        "claim_text": p["claim_text"],
                        "gold_perspective_text": p["gold_perspective_text"],
                        "generated_viewpoint_text": p["generated_viewpoint_text"],
                        "true_heuristic_class": str(p["ground_truth_class"]).lower().replace(" ", "_")
                    }
                    
            # 1. Judge 1: Gemini
            try:
                res_gem = gemini_client.evaluate_batch(batch)
                print(f"  [Judge 1 Gemini] Success | Latency: {res_gem['latency_ms']}ms")
                for item in res_gem["results"]:
                    pid = int(item.get("pair_id", -1))
                    if pid in completed_dict:
                        completed_dict[pid]["Gemini-3-Flash_label"] = str(item.get("label", "ambiguous")).lower()
                        completed_dict[pid]["Gemini-3-Flash_conf"] = float(item.get("confidence", 0.5))
                        completed_dict[pid]["Gemini-3-Flash_explanation"] = item.get("explanation", "")
            except Exception as e:
                print(f"  [Judge 1 Gemini] FAILED: {e}")
                for p in batch:
                    completed_dict[p["pair_id"]]["Gemini-3-Flash_label"] = "ambiguous"
                    completed_dict[p["pair_id"]]["Gemini-3-Flash_conf"] = 0.0
                    completed_dict[p["pair_id"]]["Gemini-3-Flash_explanation"] = str(e)
                    
            # Inter-judge pause
            time.sleep(2)
            
            # 2. Judge 2: GPT-OSS (120B / 20B fallback)
            try:
                res_gpt = evaluate_groq_batch(batch, model_name="openai/gpt-oss-120b", api_keys=groq_keys, fallback_model="openai/gpt-oss-20b")
                print(f"  [Judge 2 GPT-OSS] Success via {res_gpt['model_used']} | Latency: {res_gpt['latency_ms']}ms")
                for item in res_gpt["results"]:
                    pid = int(item.get("pair_id", -1))
                    if pid in completed_dict:
                        completed_dict[pid]["GPT-OSS-120B_label"] = str(item.get("label", "ambiguous")).lower()
                        completed_dict[pid]["GPT-OSS-120B_conf"] = float(item.get("confidence", 0.5))
                        completed_dict[pid]["GPT-OSS-120B_explanation"] = item.get("explanation", "")
            except Exception as e:
                print(f"  [Judge 2 GPT-OSS] FAILED: {e}")
                for p in batch:
                    completed_dict[p["pair_id"]]["GPT-OSS-120B_label"] = "ambiguous"
                    completed_dict[p["pair_id"]]["GPT-OSS-120B_conf"] = 0.0
                    completed_dict[p["pair_id"]]["GPT-OSS-120B_explanation"] = str(e)
                    
            # Inter-judge pause
            time.sleep(2)
            
            # 3. Judge 3: Qwen 3.8-27B
            try:
                res_qwen = evaluate_groq_batch(batch, model_name="qwen/qwen3.8-27b", api_keys=groq_keys, fallback_model="")
                print(f"  [Judge 3 Qwen] Success | Latency: {res_qwen['latency_ms']}ms")
                for item in res_qwen["results"]:
                    pid = int(item.get("pair_id", -1))
                    if pid in completed_dict:
                        completed_dict[pid]["Qwen-3.8-27B_label"] = str(item.get("label", "ambiguous")).lower()
                        completed_dict[pid]["Qwen-3.8-27B_conf"] = float(item.get("confidence", 0.5))
                        completed_dict[pid]["Qwen-3.8-27B_explanation"] = item.get("explanation", "")
            except Exception as e:
                print(f"  [Judge 3 Qwen] FAILED: {e}")
                for p in batch:
                    completed_dict[p["pair_id"]]["Qwen-3.8-27B_label"] = "ambiguous"
                    completed_dict[p["pair_id"]]["Qwen-3.8-27B_conf"] = 0.0
                    completed_dict[p["pair_id"]]["Qwen-3.8-27B_explanation"] = str(e)
                    
            # Compute consensus for completed batch
            for p in batch:
                pid = p["pair_id"]
                r = completed_dict[pid]
                v = [str(r.get("Gemini-3-Flash_label", "ambiguous")).lower(),
                     str(r.get("GPT-OSS-120B_label", "ambiguous")).lower(),
                     str(r.get("Qwen-3.8-27B_label", "ambiguous")).lower()]
                
                if v[0] == v[1] == v[2]:
                    r["consensus_label"] = v[0]
                    r["consensus_type"] = "unanimous_3_of_3"
                    r["human_adjudication_required"] = False if v[0] != "ambiguous" else True
                else:
                    vc = pd.Series(v).value_counts()
                    if vc.iloc[0] == 2:
                        r["consensus_label"] = vc.index[0]
                        r["consensus_type"] = "majority_with_ambiguous" if "ambiguous" in v else "majority_2_of_3"
                        r["human_adjudication_required"] = True
                    else:
                        r["consensus_label"] = "split"
                        r["consensus_type"] = "split_1_1_1"
                        r["human_adjudication_required"] = True
                        
            # Checkpoint immediately to disk
            df_checkpoint = pd.DataFrame(list(completed_dict.values()))
            df_checkpoint.to_csv(output_csv, index=False)
            
            # Inter-batch pause to respect TPM
            time.sleep(5)
            
        print("\nAll remaining batches finished and saved successfully!")

    # Merge full 1,700 benchmark dataset
    print("\n" + "=" * 80)
    print("MERGING FULL 1,700 BENCHMARK DATASET")
    print("=" * 80)
    
    # Load Stage A (50), Stage B (250), Stage C (300), and Remaining (1100)
    df_a = pd.read_csv("data/calibration/llm_jury_50_human_adjudicated.csv")
    df_b = pd.read_csv("data/calibration/llm_jury_stage_b_250_annotated.csv")
    df_c = pd.read_csv("data/calibration/llm_jury_stage_c_300_annotated.csv")
    df_rem = pd.read_csv(output_csv)
    
    # Standardize column structure
    common_cols = [
        "pair_id", "claim_id", "claim_text", "gold_perspective_text", "generated_viewpoint_text",
        "true_heuristic_class", "Gemini-3-Flash_label", "GPT-OSS-120B_label", "Qwen-3.8-27B_label"
    ]
    
    dfs = []
    for d, stage_tag in [(df_a, "Stage_A"), (df_b, "Stage_B"), (df_c, "Stage_C"), (df_rem, "Remaining_Pool")]:
        sub = d.copy()
        sub["stage"] = stage_tag
        if "consensus_label" not in sub.columns and "provisional_label" in sub.columns:
            sub["consensus_label"] = sub["provisional_label"]
        if "human_ground_truth" in sub.columns:
            sub["human_ground_truth_label"] = sub["human_ground_truth"]
        elif "human_label" in sub.columns:
            sub["human_ground_truth_label"] = sub["human_label"]
        else:
            sub["human_ground_truth_label"] = np.nan
        dfs.append(sub)
        
    master_df = pd.concat(dfs, ignore_index=True)
    master_df = master_df.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
    print(f"Total unique pairs in master benchmark: {len(master_df)} (Target: 1,700)")
    
    # Save master benchmark
    master_path = "data/calibration/llm_jury_full_1700_annotated.csv"
    master_df.to_csv(master_path, index=False)
    print(f"Saved Master Benchmark to {master_path}")
    
    # Save Human Triage Queue across entire 1,700 benchmark
    triage_queue = master_df[master_df["human_ground_truth_label"].isna() & (
        (master_df["consensus_type"] != "unanimous_3_of_3") | (master_df["consensus_label"] == "ambiguous")
    )].copy()
    
    triage_path = "data/calibration/llm_jury_full_1700_human_triage_queue.csv"
    triage_queue.to_csv(triage_path, index=False)
    print(f"Saved Master Human Triage Queue ({len(triage_queue)} pairs) to {triage_path}")
    
    # Export Stratified Unanimous Audit (60 pairs)
    unanimous_pool = master_df[master_df["consensus_type"] == "unanimous_3_of_3"].copy()
    audit_sample = unanimous_pool.sample(n=min(60, len(unanimous_pool)), random_state=42)
    audit_path = "data/calibration/llm_jury_full_1700_stratified_unanimous_audit.csv"
    audit_sample.to_csv(audit_path, index=False)
    print(f"Saved Stratified Unanimous Audit ({len(audit_sample)} pairs) to {audit_path}")
    
    print("\nMaster 1,700 Benchmark Scaling Setup Complete!")

if __name__ == "__main__":
    main()
