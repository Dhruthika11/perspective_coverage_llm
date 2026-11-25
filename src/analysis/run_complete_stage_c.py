"""
Complete Stage C Judge 2 (GPT-OSS) Evaluation Script (Smart Cooldown Version)
- Avoids rigid response_format json_object so valid JSON arrays [...] parse smoothly
- max_tokens set to 3500 so reasoning tokens + all 5 JSON objects fit easily
- Dedicated 429 TPM and short TPD (< 90s) sleep backoffs
- Falls back between openai/gpt-oss-120b and openai/gpt-oss-20b
- Persists progress after each batch to disk (checkpointing)
- Recalculates consensus distribution, Cohen's Kappa, and exports adjudication queues
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

from src.analysis.run_live_llm_jury import SYSTEM_INSTRUCTION, USER_BATCH_TEMPLATE, clean_json_response

def evaluate_gpt_batch(batch_pairs: List[Dict[str, Any]], api_keys: List[str], max_parse_retries: int = 5) -> Dict[str, Any]:
    pairs_text = "\n\n".join([
        f"[Pair ID: {p['pair_id']}]\nDebate Claim: \"{p['claim_text']}\"\nGold Perspective: \"{p['gold_perspective_text']}\"\nGenerated Viewpoint: \"{p['generated_viewpoint_text']}\""
        for p in batch_pairs
    ])
    user_prompt = USER_BATCH_TEMPLATE.format(batch_size=len(batch_pairs), pairs_text=pairs_text)
    expected_pids = {int(p["pair_id"]) for p in batch_pairs}
    
    models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    
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
                    
                    # Parse delay
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
                            print(f"    Model {cur_model} hit daily TPD quota (cooldown {wait_sec}s). Trying alternative model...")
                            break  # Fall back to alternative model
                    else:
                        print(f"    429 TPM rate limit on {cur_model}. Waiting {wait_sec}s (retry {rate_limit_attempts}/20)...")
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
    k1 = os.getenv("GROQ_API_KEY")
    k2 = os.getenv("GROQ_API_KEY_BACKUP")
    keys = [k for k in [k1, k2] if k]
    if not keys:
        raise ValueError("No Groq API keys found in environment.")
        
    csv_path = "data/calibration/llm_jury_stage_c_300_annotated.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Identify rows that need evaluation:
    # label is ambiguous AND explanation indicates rate limit, failure, or exception
    is_failed = df["GPT-OSS-120B_explanation"].str.contains("Rate limit|FAILED|Exception|tokens per day", na=False)
    failed_rows = df[is_failed]
    print(f"Found {len(failed_rows)} rows requiring GPT-OSS evaluation.")
    
    if len(failed_rows) == 0:
        print("All GPT labels are already populated! Proceeding to metric calculation...")
    else:
        batch_size = 5
        records_to_process = failed_rows.to_dict(orient="records")
        batches = [records_to_process[i : i + batch_size] for i in range(0, len(records_to_process), batch_size)]
        print(f"Partitioned into {len(batches)} batches of {batch_size} pairs.")
        
        # Build dictionary of all rows
        updated_dict = {row["pair_id"]: row for row in df.to_dict(orient="records")}
        
        for b_idx, batch in enumerate(batches, 1):
            pids = [p["pair_id"] for p in batch]
            print(f"\nProcessing Batch {b_idx}/{len(batches)} (Pairs: {pids})...")
            res = evaluate_gpt_batch(batch, keys)
            
            print(f"  Success via {res['model_used']} | Latency: {res['latency_ms']}ms | Output: {len(res['results'])} pairs")
            
            for item in res["results"]:
                pid = int(item.get("pair_id", -1))
                if pid in updated_dict:
                    updated_dict[pid]["GPT-OSS-120B_label"] = str(item.get("label", "ambiguous")).lower()
                    updated_dict[pid]["GPT-OSS-120B_conf"] = float(item.get("confidence", 0.5))
                    updated_dict[pid]["GPT-OSS-120B_explanation"] = item.get("explanation", "")
                    updated_dict[pid]["GPT-OSS_model_used"] = res["model_used"]
                    
            # Checkpoint immediately to disk
            df_cur = pd.DataFrame(list(updated_dict.values()))
            df_cur.to_csv(csv_path, index=False)
            
            # Inter-batch pause to prevent rapid TPM spikes
            time.sleep(5)
            
        print("\nAll batches completed and saved successfully!")

    # Reload dataset for final metric reconciliation
    df_final = pd.read_csv(csv_path)
    
    print("\nRecalculating consensus labels and classifications...")
    consensus_labels = []
    consensus_types = []
    human_reqs = []
    
    for idx, row in df_final.iterrows():
        votes = [
            str(row["Gemini-3-Flash_label"]).lower(),
            str(row["GPT-OSS-120B_label"]).lower(),
            str(row["Qwen-3.8-27B_label"]).lower()
        ]
        
        # Check unanimity
        if votes[0] == votes[1] == votes[2]:
            c_label = votes[0]
            c_type = "unanimous_3_of_3"
            h_req = False if c_label != "ambiguous" else True
        else:
            vote_counts = pd.Series(votes).value_counts()
            top_label = vote_counts.index[0]
            top_count = vote_counts.iloc[0]
            
            if top_count == 2:
                c_label = top_label
                if "ambiguous" in votes:
                    c_type = "majority_with_ambiguous"
                    h_req = True
                else:
                    c_type = "majority_2_of_3"
                    h_req = True
            else:
                c_label = "split"
                c_type = "split_1_1_1"
                h_req = True
                
        consensus_labels.append(c_label)
        consensus_types.append(c_type)
        human_reqs.append(h_req)
        
    df_final["consensus_label"] = consensus_labels
    df_final["consensus_type"] = consensus_types
    df_final["human_adjudication_required"] = human_reqs
    df_final.to_csv(csv_path, index=False)
    
    # Save human triage queue
    triage_path = "data/calibration/llm_jury_stage_c_300_human_triage.csv"
    df_triage = df_final[df_final["human_adjudication_required"]].copy()
    df_triage.to_csv(triage_path, index=False)
    print(f"Saved human triage queue ({len(df_triage)} pairs) to {triage_path}")
    
    # Create representative blinded adjudication queue:
    # All exceptions + 60 randomly sampled unanimous pairs (reproducible seed=42)
    unanimous_df = df_final[~df_final["human_adjudication_required"]]
    sample_size = min(60, len(unanimous_df))
    sampled_unanimous = unanimous_df.sample(n=sample_size, random_state=42)
    
    blinded_queue = pd.concat([df_triage, sampled_unanimous]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    blinded_export = blinded_queue[[
        "pair_id", "claim_id", "claim_text", "gold_perspective_text", "generated_viewpoint_text", "true_heuristic_class"
    ]].copy()
    blinded_export["human_ground_truth"] = ""
    blinded_export["human_adjudication_rationale"] = ""
    blinded_path = "data/calibration/stage_stage_c_300_human_adjudication_queue.csv"
    blinded_export.to_csv(blinded_path, index=False)
    print(f"Saved blinded human adjudication queue ({len(blinded_export)} pairs) to {blinded_path}")
    
    # Calculate Cohen's Kappa
    y_gem = [str(x) if pd.notna(x) else "ambiguous" for x in df_final["Gemini-3-Flash_label"]]
    y_gpt = [str(x) if pd.notna(x) else "ambiguous" for x in df_final["GPT-OSS-120B_label"]]
    y_qwen = [str(x) if pd.notna(x) else "ambiguous" for x in df_final["Qwen-3.8-27B_label"]]
    
    k_gem_gpt = cohen_kappa_score(y_gem, y_gpt)
    k_gem_qwen = cohen_kappa_score(y_gem, y_qwen)
    k_gpt_qwen = cohen_kappa_score(y_gpt, y_qwen)
    mean_k = np.mean([k_gem_gpt, k_gem_qwen, k_gpt_qwen])
    
    c_counts = df_final["consensus_type"].value_counts().to_dict()
    unanimous_n = c_counts.get("unanimous_3_of_3", 0)
    
    report = {
        "benchmark_sample_size": len(df_final),
        "batch_size": 5,
        "total_batches": len(df_final) // 5,
        "models_evaluated": [
            "Gemini-3-Flash",
            "GPT-OSS (120B / 20B fallback)",
            "Qwen-3.8-27B"
        ],
        "inter_judge_agreement": {
            "cohen_kappa_gemini_vs_gpt": round(float(k_gem_gpt), 4),
            "cohen_kappa_gemini_vs_qwen": round(float(k_gem_qwen), 4),
            "cohen_kappa_gpt_vs_qwen": round(float(k_gpt_qwen), 4),
            "mean_pairwise_kappa": round(float(mean_k), 4)
        },
        "consensus_distribution": {
            "unanimous_3_of_3": unanimous_n,
            "unanimous_pct": round(unanimous_n / len(df_final) * 100, 1),
            "majority_2_of_3": c_counts.get("majority_2_of_3", 0),
            "majority_pct": round(c_counts.get("majority_2_of_3", 0) / len(df_final) * 100, 1),
            "majority_with_ambiguous": c_counts.get("majority_with_ambiguous", 0),
            "split_1_1_1": c_counts.get("split_1_1_1", 0),
            "human_adjudication_required": int(df_final["human_adjudication_required"].sum()),
            "human_adjudication_pct": round(float(df_final["human_adjudication_required"].sum()) / len(df_final) * 100, 1)
        }
    }
    
    report_path = "results/calibration/llm_jury_stage_c_300_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved updated report to {report_path}")
    print("\nSummary Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
