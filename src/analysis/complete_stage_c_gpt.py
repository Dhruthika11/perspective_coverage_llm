"""
Script to re-evaluate failed Judge 2 (GPT-OSS) batches in Stage C (300 pairs).
Primary model: openai/gpt-oss-120b
Fallback model: openai/gpt-oss-20b (if 120b hits daily 200k TPD limit)
"""

import os
import sys
import time
import json
import re

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import requests
from dotenv import load_dotenv

from src.analysis.run_live_llm_jury import SYSTEM_INSTRUCTION, USER_BATCH_TEMPLATE, clean_json_response

def run_gpt_batch(batch_pairs, api_keys, preferred_model="openai/gpt-oss-120b", fallback_model="openai/gpt-oss-20b"):
    pairs_text = "\n\n".join([
        f"[Pair ID: {p['pair_id']}]\nDebate Claim: \"{p['claim_text']}\"\nGold Perspective: \"{p['gold_perspective_text']}\"\nGenerated Viewpoint: \"{p['generated_viewpoint_text']}\""
        for p in batch_pairs
    ])
    user_prompt = USER_BATCH_TEMPLATE.format(batch_size=len(batch_pairs), pairs_text=pairs_text)
    
    models_to_try = [preferred_model, fallback_model]
    
    for cur_model in models_to_try:
        payload = {
            "model": cur_model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        for k_idx, key in enumerate(api_keys):
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            t0 = time.time()
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
                latency_ms = int((time.time() - t0) * 1000)
                
                if resp.status_code == 200:
                    resp_json = resp.json()
                    raw_text = resp_json["choices"][0]["message"].get("content", "")
                    usage = resp_json.get("usage", {})
                    parsed = clean_json_response(raw_text)
                    return {
                        "success": True,
                        "results": parsed,
                        "latency_ms": latency_ms,
                        "model_used": cur_model,
                        "usage": usage
                    }
                elif resp.status_code == 429:
                    err_msg = resp.text
                    if "tokens per day (TPD)" in err_msg:
                        print(f"  [Notice] Model {cur_model} reached TPD limit. Trying next model...")
                        break  # Break key loop to try fallback model
                    else:
                        print(f"  [429 TPM] Waiting 12s on key #{k_idx+1}...")
                        time.sleep(12)
                else:
                    print(f"  [HTTP {resp.status_code}] on {cur_model}: {resp.text[:120]}")
                    time.sleep(2)
            except Exception as e:
                print(f"  [Exception] {e}")
                time.sleep(2)
                
    raise RuntimeError(f"All Groq models and keys failed for batch {[p['pair_id'] for p in batch_pairs]}")

if __name__ == "__main__":
    load_dotenv()
    k1 = os.getenv("GROQ_API_KEY")
    k2 = os.getenv("GROQ_API_KEY_BACKUP")
    keys = [k for k in [k1, k2] if k]
    
    # Test on single batch
    df = pd.read_csv("data/calibration/dataset_b_stage_c_sampled_300.csv")
    batch = df.iloc[15:20].to_dict(orient="records")
    res = run_gpt_batch(batch, keys)
    print("Test run successful! Model used:", res["model_used"])
    print("Parsed results count:", len(res["results"]))
    for r in res["results"]:
        print(f"  Pair {r.get('pair_id')}: {r.get('label')} ({r.get('confidence')})")
