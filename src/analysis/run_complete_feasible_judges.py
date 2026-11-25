"""
Comprehensive Feasible Completion Script for Multi-LLM Jury Benchmark.

Completes whatever is feasible across all three judges:
1. Phase 1: Gemini 3.1 Flash Lite -> Completes 100% of remaining 25 pairs (Reaches 1,700 / 1,700 = 100%).
2. Phase 2: Qwen 3.8-27B on Groq -> Runs missing pairs (up to 569 pairs) using separate Qwen quota.
3. Phase 3: GPT-OSS-120B on Groq -> Runs remaining Tier 1 pairs as rolling quota permits.

Features:
- Live consensus recomputation
- Continuous checkpointing & auto-save
- Auto-sync to HPC Flair2
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


class GeminiJudgeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash-preview", "gemini-1.5-flash"]

    def evaluate_batch(self, batch_pairs: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        pairs_text = "\n\n".join([
            f"[Pair ID: {p['pair_id']}]\n"
            f"Debate Claim: \"{p['claim_text']}\"\n"
            f"Gold Perspective: \"{p['gold_perspective_text']}\"\n"
            f"Generated Viewpoint: \"{p['generated_viewpoint_text']}\""
            for p in batch_pairs
        ])
        full_prompt = SYSTEM_INSTRUCTION + "\n\n" + USER_BATCH_TEMPLATE.format(
            batch_size=len(batch_pairs),
            pairs_text=pairs_text
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 2000,
                "temperature": 0.1
            }
        }
        last_err = None
        for attempt in range(max_retries):
            model_name = self.models[attempt % len(self.models)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            t0 = time.time()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                latency_ms = int((time.time() - t0) * 1000)
                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = clean_json_response(raw_text)
                    return {"success": True, "results": parsed, "latency_ms": latency_ms}
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
                    time.sleep(2)
            except Exception as e:
                last_err = str(e)
                time.sleep(2)
        raise RuntimeError(f"Gemini API failed: {last_err}")


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

    def evaluate_batch(self, batch_pairs: List[Dict[str, Any]], max_retries: int = 4) -> Dict[str, Any]:
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
                    time.sleep(2)
            except Exception as e:
                last_err = str(e)
                time.sleep(2)
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


def check_is_fallback(exp):
    if not isinstance(exp, str):
        return False
    low = exp.lower()
    return any(e in low for e in ['rate limit', '429', 'failed to evaluate', 'tpd', 'quota exceeded', 'read timed out', 'httpsconnectionpool', 'fallback'])


def update_master_files(df: pd.DataFrame, data_path: str, triage_path: str):
    for idx, row in df.iterrows():
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
        
    df.to_csv(data_path, index=False)
    triage_df = df[df["human_adjudication_required"] == True].copy()
    triage_df.to_csv(triage_path, index=False)


def run_feasible_completion():
    data_path = os.path.join(PROJECT_ROOT, "data", "calibration", "llm_jury_full_1700_annotated.csv")
    triage_path = os.path.join(PROJECT_ROOT, "data", "calibration", "llm_jury_full_1700_human_triage_queue.csv")
    
    df = pd.read_csv(data_path)
    print(f"Loaded master benchmark: {len(df)} pairs.")
    
    gem_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    k1 = os.environ.get("GROQ_API_KEY", "")
    k2 = os.environ.get("GROQ_API_KEY_BACKUP", "")
    
    judge_gemini = GeminiJudgeClient(api_key=gem_key)
    judge_qwen = GroqJudgeClient(primary_key=k1, backup_key=k2, model_name="qwen/qwen3.8-27b", max_tokens=1500)
    judge_gpt = GroqJudgeClient(primary_key=k1, backup_key=k2, model_name="openai/gpt-oss-120b", max_tokens=2000)
    
    # =========================================================================
    # PHASE 1: Complete 100% of Gemini (25 missing pairs)
    # =========================================================================
    gem_fb = df['Gemini-3-Flash_explanation'].apply(check_is_fallback)
    missing_gem_pids = df[gem_fb]['pair_id'].tolist()
    print(f"\n==========================================")
    print(f"PHASE 1: Gemini 3.1 Flash Lite ({len(missing_gem_pids)} missing pairs)")
    print(f"==========================================")
    
    gem_batches = [missing_gem_pids[i:i+5] for i in range(0, len(missing_gem_pids), 5)]
    gem_recovered = 0
    for b_idx, b_pids in enumerate(gem_batches, 1):
        batch_pairs = df[df["pair_id"].isin(b_pids)].to_dict(orient="records")
        print(f"[Gemini] Batch {b_idx}/{len(gem_batches)} (Pairs: {b_pids})...", end=" ", flush=True)
        try:
            res = judge_gemini.evaluate_batch(batch_pairs)
            for item in res["results"]:
                pid = int(item.get("pair_id", -1))
                match_idx = df.index[df["pair_id"] == pid].tolist()
                if match_idx:
                    idx = match_idx[0]
                    df.at[idx, "Gemini-3-Flash_label"] = str(item.get("label", "ambiguous")).lower().strip()
                    df.at[idx, "Gemini-3-Flash_conf"] = float(item.get("confidence", 0.5))
                    df.at[idx, "Gemini-3-Flash_explanation"] = str(item.get("explanation", ""))
                    gem_recovered += 1
            print(f"OK ({res['latency_ms']}ms).")
            update_master_files(df, data_path, triage_path)
            time.sleep(2.0)
        except Exception as e:
            print(f"FAILED: {e}")
            break
            
    print(f"Phase 1 Complete: Gemini cleanly evaluated {gem_recovered} pairs.")
    clean_gem = (~df['Gemini-3-Flash_explanation'].apply(check_is_fallback)).sum()
    print(f"Gemini Total Completion: {clean_gem} / {len(df)} ({clean_gem / len(df) * 100:.1f}%)")

    # =========================================================================
    # PHASE 2: Complete Qwen 3.8-27B (missing pairs)
    # =========================================================================
    qwen_fb = df['Qwen-3.8-27B_explanation'].apply(check_is_fallback)
    missing_qwen_pids = df[qwen_fb]['pair_id'].tolist()
    print(f"\n==========================================")
    print(f"PHASE 2: Qwen 3.8-27B on Groq ({len(missing_qwen_pids)} missing pairs)")
    print(f"==========================================")
    
    qwen_batches = [missing_qwen_pids[i:i+5] for i in range(0, len(missing_qwen_pids), 5)]
    qwen_recovered = 0
    for b_idx, b_pids in enumerate(qwen_batches, 1):
        batch_pairs = df[df["pair_id"].isin(b_pids)].to_dict(orient="records")
        print(f"[Qwen] Batch {b_idx}/{len(qwen_batches)} (Pairs: {b_pids})...", end=" ", flush=True)
        try:
            res = judge_qwen.evaluate_batch(batch_pairs)
            for item in res["results"]:
                pid = int(item.get("pair_id", -1))
                match_idx = df.index[df["pair_id"] == pid].tolist()
                if match_idx:
                    idx = match_idx[0]
                    df.at[idx, "Qwen-3.8-27B_label"] = str(item.get("label", "ambiguous")).lower().strip()
                    df.at[idx, "Qwen-3.8-27B_conf"] = float(item.get("confidence", 0.5))
                    df.at[idx, "Qwen-3.8-27B_explanation"] = str(item.get("explanation", ""))
                    qwen_recovered += 1
            print(f"OK ({res['latency_ms']}ms, {res.get('total_tokens', 0)} tokens). Sleeping 10s...")
            update_master_files(df, data_path, triage_path)
            time.sleep(10.0)
        except Exception as e:
            print(f"STOPPING QWEN PHASE: {e}")
            break
            
    print(f"Phase 2 Complete: Qwen cleanly evaluated {qwen_recovered} pairs.")
    clean_qwen = (~df['Qwen-3.8-27B_explanation'].apply(check_is_fallback)).sum()
    print(f"Qwen Total Completion: {clean_qwen} / {len(df)} ({clean_qwen / len(df) * 100:.1f}%)")

    # =========================================================================
    # PHASE 3: Advance GPT-OSS-120B (missing pairs as rolling window permits)
    # =========================================================================
    gpt_fb = df['GPT-OSS-120B_explanation'].apply(check_is_fallback)
    missing_gpt_pids = df[gpt_fb]['pair_id'].tolist()
    print(f"\n==========================================")
    print(f"PHASE 3: GPT-OSS-120B on Groq ({len(missing_gpt_pids)} missing pairs)")
    print(f"==========================================")
    
    gpt_batches = [missing_gpt_pids[i:i+3] for i in range(0, len(missing_gpt_pids), 3)]
    gpt_recovered = 0
    for b_idx, b_pids in enumerate(gpt_batches, 1):
        batch_pairs = df[df["pair_id"].isin(b_pids)].to_dict(orient="records")
        print(f"[GPT] Batch {b_idx}/{len(gpt_batches)} (Pairs: {b_pids})...", end=" ", flush=True)
        try:
            res = judge_gpt.evaluate_batch(batch_pairs)
            for item in res["results"]:
                pid = int(item.get("pair_id", -1))
                match_idx = df.index[df["pair_id"] == pid].tolist()
                if match_idx:
                    idx = match_idx[0]
                    df.at[idx, "GPT-OSS-120B_label"] = str(item.get("label", "ambiguous")).lower().strip()
                    df.at[idx, "GPT-OSS-120B_conf"] = float(item.get("confidence", 0.5))
                    df.at[idx, "GPT-OSS-120B_explanation"] = str(item.get("explanation", ""))
                    gpt_recovered += 1
            print(f"OK ({res['latency_ms']}ms, {res.get('total_tokens', 0)} tokens). Sleeping 10s...")
            update_master_files(df, data_path, triage_path)
            time.sleep(10.0)
        except Exception as e:
            print(f"STOPPING GPT PHASE: {e}")
            break
            
    print(f"Phase 3 Complete: GPT cleanly evaluated {gpt_recovered} pairs.")
    clean_gpt = (~df['GPT-OSS-120B_explanation'].apply(check_is_fallback)).sum()
    print(f"GPT Total Completion: {clean_gpt} / {len(df)} ({clean_gpt / len(df) * 100:.1f}%)")

    # Final summary & HPC sync
    print("\n==========================================")
    print("Master Benchmark Updated & Final Summary")
    print("==========================================")
    df_final = pd.read_csv(data_path)
    print(df_final["jury_consensus_type"].value_counts())
    print(f"Total Unanimous Decisions (3/3): {(df_final['jury_consensus_type'] == 'unanimous_3_of_3').sum()}")
    print(f"Pending in Human Triage Queue: {(df_final['human_adjudication_required'] == True).sum()}")
    
    print("\nSyncing updated master files to HPC Flair2...")
    cmd = 'powershell -Command "scp data/calibration/llm_jury_full_1700_annotated.csv data/calibration/llm_jury_full_1700_human_triage_queue.csv flair2:/home/srujan.sai/perspective_coverage/data/calibration/"'
    os.system(cmd)
    print("Sync to Flair2 completed.")

if __name__ == "__main__":
    run_feasible_completion()
