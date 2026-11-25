"""
Tier 1 Recovery Script: High-Yield LLM Jury Resolution (356 Pairs).

Focuses on single-judge recovery pairs where two judges already agreed cleanly:
- Group 1A (265 pairs): Gemini & Qwen agreed cleanly; GPT-OSS-120B hit Groq rate limit.
- Group 1B (91 pairs): Gemini & GPT agreed cleanly; Qwen-3.8-27B hit Groq rate limit.

Key Optimizations:
- GPT-OSS-120B uses batch size = 3 with max_tokens = 2000 to prevent reasoning token truncation.
- Qwen-3.8-27B uses batch size = 5 with max_tokens = 1500.
- Missing-ID resilience: partial batch returns are saved immediately; only missing IDs are retried.
- Continuous auto-save: CSV and checkpoint are updated after every batch.
- Automatic API key rotation and dynamic TPM/TPD backoff.
- Automatic sync to HPC Flair2 upon completion.
"""

import os
import sys
import json
import time
import re
import shutil
import argparse
from typing import Dict, List, Any, Optional
from collections import Counter
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

SYSTEM_INSTRUCTION = """You are an expert deliberative debate judge evaluating argumentative equivalence between a gold human reference perspective and an AI-generated viewpoint.
Your task is to classify each pair into one of five mutually exclusive classes:
- class_2: Same reasoning and same stance (Equivalent Perspective)
- class_1: Same stance, but distinct argumentative reasoning / criteria
- class_0a: Opposite stance on the same claim (Rebuttal / Counter-argument)
- class_0b: Unrelated debate claim / topic
- ambiguous: Genuinely insufficient evidence, vague reasoning, or unresolvable stance

You must analyze all pairs in the batch and output a strictly valid JSON array of evaluation objects.
Do not wrap your output in conversational filler. Output ONLY the JSON array."""

USER_BATCH_TEMPLATE = """Evaluate the following {batch_size} argument pairs:

{pairs_text}

For each pair, return a JSON object with:
{{
  "pair_id": <integer ID>,
  "label": "class_2" | "class_1" | "class_0a" | "class_0b" | "ambiguous",
  "same_reasoning": true/false,
  "same_stance": true/false,
  "explanation": "<1 concise sentence explaining the warrant comparison>",
  "confidence": <float between 0.0 and 1.0>
}}

Return exactly a JSON array containing {batch_size} objects in corresponding order:
[
  {{ ... }},
  ...
]"""

def clean_json_response(raw_text: str) -> List[Dict[str, Any]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return [data]
    except Exception:
        pass
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    objects = []
    for obj_match in re.finditer(r"\{\s*\"pair_id\".*?\}", text, re.DOTALL):
        try:
            objects.append(json.loads(obj_match.group(0)))
        except Exception:
            pass
    if objects:
        return objects
    raise ValueError(f"Failed to parse valid JSON array from LLM output:\n{raw_text[:300]}...")


class GroqJudgeClient:
    def __init__(self, primary_key: str, backup_key: Optional[str], model_name: str, max_tokens: int = 1500):
        self.keys = [k for k in [primary_key, backup_key] if k]
        self.key_idx = 0
        self.model_name = model_name
        self.max_tokens = max_tokens

    def _get_active_key(self) -> str:
        return self.keys[self.key_idx % len(self.keys)]

    def _rotate_key(self):
        if len(self.keys) > 1:
            self.key_idx = (self.key_idx + 1) % len(self.keys)
            print(f"  [API Key Rotated] Switched Groq key to slot #{self.key_idx + 1}")

    def evaluate_batch(self, batch_pairs: List[Dict[str, Any]], max_retries: int = 5) -> Dict[str, Any]:
        pairs_text = "\n\n".join([
            f"[Pair ID: {p['pair_id']}]\n"
            f"Debate Claim: \"{p['claim_text']}\"\n"
            f"Gold Perspective: \"{p['gold_perspective_text']}\"\n"
            f"Generated Viewpoint: \"{p['generated_viewpoint_text']}\""
            for p in batch_pairs
        ])
        
        user_prompt = USER_BATCH_TEMPLATE.format(
            batch_size=len(batch_pairs),
            pairs_text=pairs_text
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": self.max_tokens
        }
        
        last_err = None
        for attempt in range(max_retries):
            active_key = self._get_active_key()
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json"
            }
            
            t0 = time.time()
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=35
                )
                latency_ms = int((time.time() - t0) * 1000)
                
                if resp.status_code == 200:
                    resp_json = resp.json()
                    msg = resp_json["choices"][0]["message"]
                    raw_text = msg.get("content", "")
                    usage = resp_json.get("usage", {})
                    parsed_list = clean_json_response(raw_text)
                    
                    return {
                        "success": True,
                        "results": parsed_list,
                        "latency_ms": latency_ms,
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                elif resp.status_code == 429:
                    wait_sec = 10
                    try:
                        m_wait = re.search(r"try again in (\d+(?:\.\d+)?)s", resp.text)
                        if m_wait:
                            wait_sec = int(float(m_wait.group(1))) + 2
                    except Exception:
                        pass
                    print(f"  [Rate Limit 429] Waiting {wait_sec}s & rotating key...")
                    self._rotate_key()
                    time.sleep(wait_sec)
                    last_err = resp.text
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
                    print(f"  [Error] {last_err}. Retrying in 3s...")
                    time.sleep(3)
            except Exception as e:
                last_err = str(e)
                print(f"  [Exception] {last_err}. Retrying in 3s...")
                time.sleep(3)
                
        raise RuntimeError(f"Groq API ({self.model_name}) failed after {max_retries} attempts: {last_err}")


def compute_consensus(gem_label: str, gpt_label: str, qwen_label: str, has_human: bool):
    votes = [str(gem_label).lower().strip(), str(gpt_label).lower().strip(), str(qwen_label).lower().strip()]
    counts = Counter(votes)
    unique_votes = set(votes)
    
    # 1. Unanimous (3/3)
    if len(unique_votes) == 1:
        v = votes[0]
        if v == "ambiguous":
            return "ambiguous", "unanimous_3_of_3", not has_human
        else:
            return v, "unanimous_3_of_3", False
            
    # 2. Majority (2/3)
    top_label, top_count = counts.most_common(1)[0]
    if top_count == 2:
        dissenting = [v for v in votes if v != top_label][0]
        if dissenting == "ambiguous" or top_label == "ambiguous":
            return top_label, "majority_with_ambiguous", not has_human
        else:
            return top_label, "majority_2_of_3", not has_human
            
    # 3. Three-way split (1/1/1)
    return "split", "split_1_1_1", not has_human


def update_master_records(df: pd.DataFrame, checkpoint: Dict[str, Any], data_path: str, triage_path: str):
    """Merges checkpoint into df, recalculates consensus, and saves both master and triage files."""
    for idx, row in df.iterrows():
        pid_str = str(row["pair_id"])
        updated = False
        
        if pid_str in checkpoint.get("gpt_recovered", {}):
            rec = checkpoint["gpt_recovered"][pid_str]
            df.at[idx, "GPT-OSS-120B_label"] = rec["label"]
            df.at[idx, "GPT-OSS-120B_conf"] = rec["confidence"]
            df.at[idx, "GPT-OSS-120B_explanation"] = rec["explanation"]
            updated = True
            
        if pid_str in checkpoint.get("qwen_recovered", {}):
            rec = checkpoint["qwen_recovered"][pid_str]
            df.at[idx, "Qwen-3.8-27B_label"] = rec["label"]
            df.at[idx, "Qwen-3.8-27B_conf"] = rec["confidence"]
            df.at[idx, "Qwen-3.8-27B_explanation"] = rec["explanation"]
            updated = True
            
        if updated:
            has_h = pd.notna(row["human_ground_truth_label"]) and str(row["human_ground_truth_label"]).strip() != ""
            c_label, c_type, h_req = compute_consensus(
                df.at[idx, "Gemini-3-Flash_label"],
                df.at[idx, "GPT-OSS-120B_label"],
                df.at[idx, "Qwen-3.8-27B_label"],
                has_h
            )
            df.at[idx, "jury_consensus_label"] = c_label
            df.at[idx, "jury_consensus_type"] = c_type
            df.at[idx, "human_adjudication_required"] = h_req
            
    # Save master dataset
    df.to_csv(data_path, index=False)
    # Save triage queue
    triage_df = df[df["human_adjudication_required"] == True].copy()
    triage_df.to_csv(triage_path, index=False)


def run_recovery(gpt_batch_size: int = 3, qwen_batch_size: int = 5, delay_sec: float = 10.0, max_batches: Optional[int] = None):
    data_path = os.path.join(PROJECT_ROOT, "data", "calibration", "llm_jury_full_1700_annotated.csv")
    backup_path = data_path + ".bak_tier1_pre"
    checkpoint_path = os.path.join(PROJECT_ROOT, "data", "calibration", "tier1_recovery_checkpoint.json")
    triage_path = os.path.join(PROJECT_ROOT, "data", "calibration", "llm_jury_full_1700_human_triage_queue.csv")
    
    if not os.path.exists(backup_path):
        print(f"Creating backup of master dataset at: {backup_path}")
        shutil.copyfile(data_path, backup_path)
        
    df = pd.read_csv(data_path)
    print(f"Loaded master dataset: {len(df)} pairs.")
    
    # Load checkpoint
    checkpoint = {"gpt_recovered": {}, "qwen_recovered": {}}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"Loaded existing checkpoint: {len(checkpoint.get('gpt_recovered', {}))} GPT & {len(checkpoint.get('qwen_recovered', {}))} Qwen records.")
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")
            
    def check_fb(exp):
        if not isinstance(exp, str):
            return False
        low = exp.lower()
        return any(e in low for e in ['rate limit', '429', 'failed to evaluate', 'tpd', 'quota exceeded', 'read timed out', 'httpsconnectionpool', 'fallback'])

    gpt_fb = df['GPT-OSS-120B_explanation'].apply(check_fb)
    qwen_fb = df['Qwen-3.8-27B_explanation'].apply(check_fb)
    gemini_fb = df['Gemini-3-Flash_explanation'].apply(check_fb)
    has_human = df['human_ground_truth_label'].notna() & (df['human_ground_truth_label'] != '')
    
    # Tier 1A: GPT needs evaluation
    tier_1a_df = df[gpt_fb & ~qwen_fb & ~gemini_fb & ~has_human].copy()
    # Tier 1B: Qwen needs evaluation
    tier_1b_df = df[qwen_fb & ~gpt_fb & ~gemini_fb & ~has_human].copy()
    
    print(f"Identified Tier 1A (GPT to evaluate): {len(tier_1a_df)} pairs.")
    print(f"Identified Tier 1B (Qwen to evaluate): {len(tier_1b_df)} pairs.")
    print(f"Total Tier 1 target pairs: {len(tier_1a_df) + len(tier_1b_df)} pairs.\n")
    
    k1 = os.environ.get("GROQ_API_KEY", "")
    k2 = os.environ.get("GROQ_API_KEY_BACKUP", "")
    
    client_gpt = GroqJudgeClient(primary_key=k1, backup_key=k2, model_name="openai/gpt-oss-120b", max_tokens=2000)
    client_qwen = GroqJudgeClient(primary_key=k1, backup_key=k2, model_name="qwen/qwen3.8-27b", max_tokens=1500)
    
    # ==========================================
    # Run Tier 1B First: Qwen-3.8-27B (Fast & High Yield)
    # ==========================================
    pids_1b = tier_1b_df["pair_id"].tolist()
    pending_1b = [pid for pid in pids_1b if str(pid) not in checkpoint["qwen_recovered"]]
    print(f"--- Starting Tier 1B: {len(pending_1b)} pending pairs for Qwen-3.8-27B ---")
    
    batches_1b = [pending_1b[i : i + qwen_batch_size] for i in range(0, len(pending_1b), qwen_batch_size)]
    if max_batches is not None:
        batches_1b = batches_1b[:max_batches]
        
    for b_idx, b_pids in enumerate(batches_1b, 1):
        batch_pairs = tier_1b_df[tier_1b_df["pair_id"].isin(b_pids)].to_dict(orient="records")
        t0 = time.time()
        print(f"[Tier 1B - Qwen] Batch {b_idx}/{len(batches_1b)} (Pairs: {b_pids})...", end=" ", flush=True)
        try:
            res = client_qwen.evaluate_batch(batch_pairs)
            new_pids = []
            for item in res["results"]:
                pid = str(item.get("pair_id", -1))
                if pid in [str(p) for p in b_pids]:
                    checkpoint["qwen_recovered"][pid] = {
                        "label": str(item.get("label", "ambiguous")).lower().strip(),
                        "confidence": float(item.get("confidence", 0.5)),
                        "explanation": str(item.get("explanation", ""))
                    }
                    new_pids.append(pid)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
            update_master_records(df, checkpoint, data_path, triage_path)
            print(f"OK ({res['latency_ms']}ms, {res['total_tokens']} tokens, saved {len(new_pids)}). Sleeping {delay_sec}s...")
        except Exception as e:
            print(f"FAILED: {e}")
            break
        if b_idx < len(batches_1b):
            time.sleep(delay_sec)
            
    # ==========================================
    # Run Tier 1A: GPT-OSS-120B (3 Pairs per Batch)
    # ==========================================
    pids_1a = tier_1a_df["pair_id"].tolist()
    pending_1a = [pid for pid in pids_1a if str(pid) not in checkpoint["gpt_recovered"]]
    print(f"\n--- Starting Tier 1A: {len(pending_1a)} pending pairs for GPT-OSS-120B ---")
    
    batches_1a = [pending_1a[i : i + gpt_batch_size] for i in range(0, len(pending_1a), gpt_batch_size)]
    if max_batches is not None:
        batches_1a = batches_1a[:max_batches]
        
    for b_idx, b_pids in enumerate(batches_1a, 1):
        batch_pairs = tier_1a_df[tier_1a_df["pair_id"].isin(b_pids)].to_dict(orient="records")
        t0 = time.time()
        print(f"[Tier 1A - GPT] Batch {b_idx}/{len(batches_1a)} (Pairs: {b_pids})...", end=" ", flush=True)
        try:
            res = client_gpt.evaluate_batch(batch_pairs)
            new_pids = []
            for item in res["results"]:
                pid = str(item.get("pair_id", -1))
                if pid in [str(p) for p in b_pids]:
                    checkpoint["gpt_recovered"][pid] = {
                        "label": str(item.get("label", "ambiguous")).lower().strip(),
                        "confidence": float(item.get("confidence", 0.5)),
                        "explanation": str(item.get("explanation", ""))
                    }
                    new_pids.append(pid)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
            update_master_records(df, checkpoint, data_path, triage_path)
            print(f"OK ({res['latency_ms']}ms, {res['total_tokens']} tokens, saved {len(new_pids)}). Sleeping {delay_sec}s...")
        except Exception as e:
            print(f"FAILED: {e}")
            break
        if b_idx < len(batches_1a):
            time.sleep(delay_sec)

    # Final summary and HPC sync
    print("\n==========================================")
    print("Recovery Run Complete! Master Dataset Updated.")
    print("==========================================")
    df_final = pd.read_csv(data_path)
    print("\nConsensus Type Breakdown across 1,700 Benchmark:")
    print(df_final["jury_consensus_type"].value_counts())
    
    triage_count = (df_final["human_adjudication_required"] == True).sum()
    unanimous_count = (df_final["jury_consensus_type"] == "unanimous_3_of_3").sum()
    print(f"\nTotal Unanimous Decisions (3/3): {unanimous_count}")
    print(f"Total Pending in Human Triage Queue: {triage_count}")
    
    print("\n--- Syncing updated files to HPC Flair2 ---")
    cmd = 'powershell -Command "scp data/calibration/llm_jury_full_1700_annotated.csv data/calibration/llm_jury_full_1700_human_triage_queue.csv flair2:/home/srujan.sai/perspective_coverage/data/calibration/"'
    os.system(cmd)
    print("Sync to Flair2 completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-batch-size", type=int, default=3)
    parser.add_argument("--qwen-batch-size", type=int, default=5)
    parser.add_argument("--delay-sec", type=float, default=10.0)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    run_recovery(
        gpt_batch_size=args.gpt_batch_size,
        qwen_batch_size=args.qwen_batch_size,
        delay_sec=args.delay_sec,
        max_batches=args.max_batches
    )
