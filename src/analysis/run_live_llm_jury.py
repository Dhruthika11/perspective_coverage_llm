"""
Production Live Multi-LLM Jury Runner (5-Pair Batching, Quota Rotation & HPC Ready).

Evaluates candidate argument pairs using an independent 3-model jury:
  1. Judge 1: Gemini 3 Flash (Google AI Studio)
  2. Judge 2: GPT-OSS-120B (Groq Cloud)
  3. Judge 3: Qwen 3.8-27B (Groq Cloud)

Features:
  - 5-pair batching to amortize system prompt overhead and stay within Groq daily quotas.
  - Automatic Groq API key rotation (Primary <-> Backup) on rate limits (HTTP 429).
  - Transient 503 backoff and fallback logic for Google Generative AI endpoints.
  - Enforced structured JSON parsing with schema validation.
  - Full inter-judge Cohen's kappa and consensus analytics.
  - Human triage priority export for ambiguous/split pairs.
"""

import os
import sys
import json
import time
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("live_llm_jury")


# ===========================================================================
# 1. Environment & API Key Loader
# ===========================================================================
def load_env_keys(env_path: Optional[str] = None) -> Dict[str, str]:
    """Loads API keys from .env file or system environment variables."""
    keys = {}
    
    # Try specified env path, then current dir, then project root
    candidates = []
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        os.path.join(os.getcwd(), ".env"),
        os.path.join(PROJECT_ROOT, ".env"),
        os.path.expanduser("~/.env")
    ])
    
    for c in candidates:
        if os.path.exists(c):
            logger.info(f"Loading environment variables from: {c}")
            with open(c, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        keys[k.strip()] = v.strip().strip("'\"")
            break
            
    # System env fallback
    for env_k in ["GROQ_API_KEY", "GROQ_API_KEY_BACKUP", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        if env_k in os.environ and env_k not in keys:
            keys[env_k] = os.environ[env_k]
            
    return keys


# ===========================================================================
# 2. Prompts for 5-Pair Batching
# ===========================================================================
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
    """Robustly extracts and parses a JSON list from LLM output."""
    text = raw_text.strip()
    
    # Strip markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()
        
    # Attempt direct json load
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If wrapped in a parent dict like {"evaluations": [...]} or {"pairs": [...]}
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return [data]
    except Exception:
        pass
        
    # Regex fallback to find JSON array [ ... ]
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
            
    # Try finding individual JSON objects { ... }
    objects = []
    for obj_match in re.finditer(r"\{\s*\"pair_id\".*?\}", text, re.DOTALL):
        try:
            objects.append(json.loads(obj_match.group(0)))
        except Exception:
            pass
    if objects:
        return objects
        
    raise ValueError(f"Failed to parse valid JSON array from LLM output:\n{raw_text[:300]}...")


# ===========================================================================
# 3. Model API Clients
# ===========================================================================
class GeminiJudgeClient:
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = "Google AI Studio"
        self.family = "Gemini"
        self.fallback_models = ["gemini-3.1-flash-lite", "gemini-3-flash-preview"]

    def evaluate_batch(self, batch_pairs: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        """Calls Gemini API for a 5-pair batch."""
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
            # Rotate model if prior attempt hit 503
            cur_model = self.model_name if attempt == 0 else self.fallback_models[(attempt - 1) % len(self.fallback_models)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent?key={self.api_key}"
            
            t0 = time.time()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                latency_ms = int((time.time() - t0) * 1000)
                
                if resp.status_code == 200:
                    resp_json = resp.json()
                    candidate = resp_json["candidates"][0]
                    raw_text = candidate["content"]["parts"][0]["text"]
                    usage = resp_json.get("usageMetadata", {})
                    
                    parsed_list = clean_json_response(raw_text)
                    
                    # Verify completeness: check all pair IDs are returned
                    expected_pids = {int(p["pair_id"]) for p in batch_pairs}
                    returned_pids = {int(item.get("pair_id", -1)) for item in parsed_list}
                    missing_pids = expected_pids - returned_pids
                    
                    if missing_pids and attempt < max_retries - 1:
                        logger.warning(f"Groq {self.model_name} returned incomplete batch: missing pair IDs {missing_pids}. Retrying batch...")
                        time.sleep(1.5)
                        continue
                        
                    return {
                        "success": True,
                        "results": parsed_list,
                        "latency_ms": latency_ms,
                        "prompt_tokens": usage.get("promptTokenCount", 0),
                        "output_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0),
                        "model_used": cur_model
                    }
                elif resp.status_code in [429, 503]:
                    logger.warning(f"Gemini {cur_model} returned {resp.status_code}. Retrying in {2 ** attempt}s...")
                    time.sleep(2 ** attempt)
                    last_err = resp.text
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
                    time.sleep(1)
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5)
                
        raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {last_err}")


class GroqJudgeClient:
    def __init__(self, primary_key: str, backup_key: Optional[str], model_name: str, family: str):
        self.keys = [k for k in [primary_key, backup_key] if k]
        self.key_idx = 0
        self.model_name = model_name
        self.provider = "Groq Cloud"
        self.family = family

    def _get_active_key(self) -> str:
        return self.keys[self.key_idx % len(self.keys)]

    def _rotate_key(self):
        if len(self.keys) > 1:
            self.key_idx = (self.key_idx + 1) % len(self.keys)
            logger.info(f"Rotated Groq API key to slot #{self.key_idx + 1} for {self.model_name}")

    def evaluate_batch(self, batch_pairs: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        """Calls Groq API for a 5-pair batch."""
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
            "max_tokens": 1500
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
                    timeout=25
                )
                latency_ms = int((time.time() - t0) * 1000)
                
                if resp.status_code == 200:
                    resp_json = resp.json()
                    msg = resp_json["choices"][0]["message"]
                    raw_text = msg.get("content", "")
                    usage = resp_json.get("usage", {})
                    
                    parsed_list = clean_json_response(raw_text)
                    
                    # Verify completeness: check all pair IDs are returned
                    expected_pids = {int(p["pair_id"]) for p in batch_pairs}
                    returned_pids = {int(item.get("pair_id", -1)) for item in parsed_list}
                    missing_pids = expected_pids - returned_pids
                    
                    if missing_pids and attempt < max_retries - 1:
                        logger.warning(f"Groq {self.model_name} returned incomplete batch: missing pair IDs {missing_pids}. Retrying batch...")
                        time.sleep(1.5)
                        continue
                        
                    return {
                        "success": True,
                        "results": parsed_list,
                        "latency_ms": latency_ms,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "model_used": self.model_name
                    }
                elif resp.status_code == 429:
                    # Parse suggested retry delay if present, else default to 12s
                    wait_sec = 12
                    try:
                        m_wait = re.search(r"try again in (\d+(?:\.\d+)?)s", resp.text)
                        if m_wait:
                            wait_sec = int(float(m_wait.group(1))) + 2
                    except Exception:
                        pass
                    logger.warning(f"Groq rate limit (429) hit on key #{self.key_idx + 1}. Waiting {wait_sec}s & rotating key...")
                    self._rotate_key()
                    time.sleep(wait_sec)
                    last_err = resp.text
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
                    time.sleep(1)
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5)
                
        raise RuntimeError(f"Groq API ({self.model_name}) failed after {max_retries} attempts: {last_err}")


# ===========================================================================
# 4. Main Batch Orchestrator
# ===========================================================================
def run_live_jury_batch(
    num_pairs: int = 50,
    batch_size: int = 5,
    seed: int = 42,
    input_csv: str = "data/calibration/human_llm_benchmark_1700.csv",
    output_dir: str = "data/calibration",
    results_dir: str = "results/calibration",
    env_file: Optional[str] = None,
    run_tag: str = "live_50"
) -> Dict[str, Any]:
    
    logger.info("=" * 80)
    logger.info(f"STARTING LIVE MULTI-LLM JURY EVALUATION (N={num_pairs} Pairs, Batch Size={batch_size})")
    logger.info("=" * 80)
    
    # 1. Load keys
    keys = load_env_keys(env_file)
    gem_key = keys.get("GEMINI_API_KEY") or keys.get("GOOGLE_API_KEY")
    groq_primary = keys.get("GROQ_API_KEY")
    groq_backup = keys.get("GROQ_API_KEY_BACKUP")
    
    if not gem_key:
        raise ValueError("Missing GEMINI_API_KEY in environment or .env file.")
    if not groq_primary:
        raise ValueError("Missing GROQ_API_KEY in environment or .env file.")
        
    logger.info(f"API Keys Loaded: Gemini (Active), Groq Primary (Active), Groq Backup ({'Active' if groq_backup else 'None'})")
    
    # 2. Initialize Judges
    judge_gemini = GeminiJudgeClient(api_key=gem_key, model_name="gemini-3.1-flash-lite-preview")
    judge_gpt = GroqJudgeClient(primary_key=groq_primary, backup_key=groq_backup, model_name="openai/gpt-oss-120b", family="Meta/OpenSource")
    judge_qwen = GroqJudgeClient(primary_key=groq_primary, backup_key=groq_backup, model_name="qwen/qwen3.8-27b", family="Alibaba/Qwen")
    
    jury = [judge_gemini, judge_gpt, judge_qwen]
    
    # 3. Load dataset
    resolved_input = resolve(input_csv)
    df_all = load_csv(resolved_input)
    logger.info(f"Loaded {len(df_all)} pairs from {resolved_input}")
    
    if len(df_all) <= num_pairs:
        df_sample = df_all.copy()
        logger.info(f"Using all {len(df_sample)} pre-sampled pairs directly.")
    elif num_pairs == 50:
        # Stratified balance across the 4 classes for 50 pairs
        strata_counts = {"Class 2": 18, "Class 1": 12, "Class 0A": 12, "Class 0B": 8}
        sampled_dfs = []
        for cls_name, count in strata_counts.items():
            cls_sub = df_all[df_all["ground_truth_class"] == cls_name]
            sampled_dfs.append(cls_sub.sample(n=min(count, len(cls_sub)), random_state=seed))
        df_sample = pd.concat(sampled_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        logger.info(f"Selected {len(df_sample)} stratified pairs for live benchmark.")
    else:
        df_sample = df_all.sample(n=num_pairs, random_state=seed).reset_index(drop=True)
        logger.info(f"Sampled {len(df_sample)} pairs randomly from {len(df_all)} total.")
    
    # 4. Partition into batches
    batches = []
    pairs_list = df_sample.to_dict(orient="records")
    for i in range(0, len(pairs_list), batch_size):
        batches.append(pairs_list[i : i + batch_size])
        
    logger.info(f"Created {len(batches)} batches of {batch_size} pairs each ({len(batches) * 3} total LLM calls).")
    
    # 5. Execute Live Jury Calls
    annotated_rows: Dict[int, Dict[str, Any]] = {p["pair_id"]: {
        "pair_id": p["pair_id"],
        "claim_id": p["claim_id"],
        "claim_text": p["claim_text"],
        "gold_perspective_text": p["gold_perspective_text"],
        "generated_viewpoint_text": p["generated_viewpoint_text"],
        "true_heuristic_class": str(p["ground_truth_class"]).lower().replace(" ", "_"),
        "stance_gold": p.get("stance_gold", ""),
        "stance_generated": p.get("stance_generated", "")
    } for p in pairs_list}
    
    telemetry = {
        "Gemini-3-Flash": {"latencies": [], "prompt_tokens": 0, "output_tokens": 0},
        "GPT-OSS-120B": {"latencies": [], "prompt_tokens": 0, "output_tokens": 0},
        "Qwen-3.8-27B": {"latencies": [], "prompt_tokens": 0, "output_tokens": 0}
    }
    
    for batch_idx, batch in enumerate(batches, 1):
        logger.info(f"\n--- Processing Batch {batch_idx}/{len(batches)} (Pairs: {[p['pair_id'] for p in batch]}) ---")
        
        # Call Judge 1: Gemini
        try:
            res_gem = judge_gemini.evaluate_batch(batch)
            telemetry["Gemini-3-Flash"]["latencies"].append(res_gem["latency_ms"])
            telemetry["Gemini-3-Flash"]["prompt_tokens"] += res_gem["prompt_tokens"]
            telemetry["Gemini-3-Flash"]["output_tokens"] += res_gem["output_tokens"]
            logger.info(f"  [Judge 1 Gemini] Latency: {res_gem['latency_ms']}ms | Output: {len(res_gem['results'])} pairs")
            for item in res_gem["results"]:
                pid = int(item.get("pair_id", -1))
                if pid in annotated_rows:
                    annotated_rows[pid]["Gemini-3-Flash_label"] = str(item.get("label", "ambiguous")).lower()
                    annotated_rows[pid]["Gemini-3-Flash_conf"] = float(item.get("confidence", 0.5))
                    annotated_rows[pid]["Gemini-3-Flash_explanation"] = item.get("explanation", "")
        except Exception as e:
            logger.error(f"  [Judge 1 Gemini] FAILED batch {batch_idx}: {e}")
            for p in batch:
                annotated_rows[p["pair_id"]]["Gemini-3-Flash_label"] = "ambiguous"
                annotated_rows[p["pair_id"]]["Gemini-3-Flash_conf"] = 0.0
                annotated_rows[p["pair_id"]]["Gemini-3-Flash_explanation"] = str(e)
                
        # Call Judge 2: GPT-OSS-120B
        try:
            res_gpt = judge_gpt.evaluate_batch(batch)
            telemetry["GPT-OSS-120B"]["latencies"].append(res_gpt["latency_ms"])
            telemetry["GPT-OSS-120B"]["prompt_tokens"] += res_gpt["prompt_tokens"]
            telemetry["GPT-OSS-120B"]["output_tokens"] += res_gpt["output_tokens"]
            logger.info(f"  [Judge 2 GPT-OSS] Latency: {res_gpt['latency_ms']}ms | Output: {len(res_gpt['results'])} pairs")
            for item in res_gpt["results"]:
                pid = int(item.get("pair_id", -1))
                if pid in annotated_rows:
                    annotated_rows[pid]["GPT-OSS-120B_label"] = str(item.get("label", "ambiguous")).lower()
                    annotated_rows[pid]["GPT-OSS-120B_conf"] = float(item.get("confidence", 0.5))
                    annotated_rows[pid]["GPT-OSS-120B_explanation"] = item.get("explanation", "")
        except Exception as e:
            logger.error(f"  [Judge 2 GPT-OSS] FAILED batch {batch_idx}: {e}")
            for p in batch:
                annotated_rows[p["pair_id"]]["GPT-OSS-120B_label"] = "ambiguous"
                annotated_rows[p["pair_id"]]["GPT-OSS-120B_conf"] = 0.0
                annotated_rows[p["pair_id"]]["GPT-OSS-120B_explanation"] = str(e)
                
        # Call Judge 3: Qwen 3.8-27B
        try:
            res_qwen = judge_qwen.evaluate_batch(batch)
            telemetry["Qwen-3.8-27B"]["latencies"].append(res_qwen["latency_ms"])
            telemetry["Qwen-3.8-27B"]["prompt_tokens"] += res_qwen["prompt_tokens"]
            telemetry["Qwen-3.8-27B"]["output_tokens"] += res_qwen["output_tokens"]
            logger.info(f"  [Judge 3 Qwen] Latency: {res_qwen['latency_ms']}ms | Output: {len(res_qwen['results'])} pairs")
            for item in res_qwen["results"]:
                pid = int(item.get("pair_id", -1))
                if pid in annotated_rows:
                    annotated_rows[pid]["Qwen-3.8-27B_label"] = str(item.get("label", "ambiguous")).lower()
                    annotated_rows[pid]["Qwen-3.8-27B_conf"] = float(item.get("confidence", 0.5))
                    annotated_rows[pid]["Qwen-3.8-27B_explanation"] = item.get("explanation", "")
        except Exception as e:
            logger.error(f"  [Judge 3 Qwen] FAILED batch {batch_idx}: {e}")
            for p in batch:
                annotated_rows[p["pair_id"]]["Qwen-3.8-27B_label"] = "ambiguous"
                annotated_rows[p["pair_id"]]["Qwen-3.8-27B_conf"] = 0.0
                annotated_rows[p["pair_id"]]["Qwen-3.8-27B_explanation"] = str(e)
                
        # Pause between batches to respect Groq free tier 1000 OTPM limit
        time.sleep(6.0)
        
    # 6. Consensus & Statistical Agreement Engine
    df_out = pd.DataFrame(list(annotated_rows.values()))
    
    consensus_list = []
    provisional_labels = []
    action_codes = []
    
    valid_classes = {"class_2", "class_1", "class_0a", "class_0b"}
    
    for _, row in df_out.iterrows():
        v1 = row.get("Gemini-3-Flash_label", "ambiguous")
        v2 = row.get("GPT-OSS-120B_label", "ambiguous")
        v3 = row.get("Qwen-3.8-27B_label", "ambiguous")
        votes = [v1, v2, v3]
        
        if "ambiguous" in votes:
            consensus_list.append("AMBIGUOUS (Requires Human Review)")
            provisional_labels.append("ambiguous")
            action_codes.append("HUMAN_ADJUDICATION_URGENT")
        elif len(set(votes)) == 1:
            consensus_list.append("3/3 UNANIMOUS")
            provisional_labels.append(votes[0])
            action_codes.append("HIGH_CONFIDENCE_AUTO")
        elif len(set(votes)) == 2:
            maj = max(set(votes), key=votes.count)
            consensus_list.append("2/3 MAJORITY")
            provisional_labels.append(maj)
            action_codes.append("PROVISIONAL_MAJORITY")
        else:
            consensus_list.append("UNRESOLVED_SPLIT (1/1/1)")
            provisional_labels.append("split")
            action_codes.append("HUMAN_ADJUDICATION_REQUIRED")
            
    df_out["consensus_status"] = consensus_list
    df_out["provisional_label"] = provisional_labels
    df_out["action_code"] = action_codes
    
    # 7. Compute Cohen's Kappa Metrics
    v_gem = [str(x) if pd.notna(x) else "ambiguous" for x in df_out.get("Gemini-3-Flash_label", [])]
    v_gpt = [str(x) if pd.notna(x) else "ambiguous" for x in df_out.get("GPT-OSS-120B_label", [])]
    v_qwen = [str(x) if pd.notna(x) else "ambiguous" for x in df_out.get("Qwen-3.8-27B_label", [])]
    
    # Ensure all list elements are valid strings
    labels_all = sorted(list(set(v_gem + v_gpt + v_qwen)))
    
    kappa_gem_gpt = cohen_kappa_score(v_gem, v_gpt, labels=labels_all) if len(labels_all) > 1 else 1.0
    kappa_gem_qwen = cohen_kappa_score(v_gem, v_qwen, labels=labels_all) if len(labels_all) > 1 else 1.0
    kappa_gpt_qwen = cohen_kappa_score(v_gpt, v_qwen, labels=labels_all) if len(labels_all) > 1 else 1.0
    mean_kappa = np.mean([kappa_gem_gpt, kappa_gem_qwen, kappa_gpt_qwen])
    
    unanimous_n = (df_out["consensus_status"] == "3/3 UNANIMOUS").sum()
    majority_n = (df_out["consensus_status"] == "2/3 MAJORITY").sum()
    human_n = len(df_out) - unanimous_n
    
    # Human triage subset
    df_human_triage = df_out[df_out["action_code"].str.startswith("HUMAN")].copy()
    
    # 8. Save Outputs
    os.makedirs(resolve(output_dir), exist_ok=True)
    os.makedirs(resolve(results_dir), exist_ok=True)
    
    csv_annotated_path = os.path.join(resolve(output_dir), f"llm_jury_{run_tag}_annotated.csv")
    csv_triage_path = os.path.join(resolve(output_dir), f"llm_jury_{run_tag}_human_triage.csv")
    csv_adjudication_sample_path = os.path.join(resolve(output_dir), f"stage_{run_tag}_human_adjudication_queue.csv")
    json_report_path = os.path.join(resolve(results_dir), f"llm_jury_{run_tag}_report.json")
    
    # Mentor Requirement: Sample 60 random unanimous pairs + all exceptions for human validation
    df_exceptions = df_out[df_out["consensus_status"] != "3/3 UNANIMOUS"].copy()
    df_unanimous = df_out[df_out["consensus_status"] == "3/3 UNANIMOUS"].copy()
    
    n_sample_unanimous = min(60, len(df_unanimous))
    df_unanimous_sample = df_unanimous.sample(n=n_sample_unanimous, random_state=seed) if len(df_unanimous) > 0 else pd.DataFrame()
    
    # Combined adjudication queue (blinded: only raw texts, claim, pair_id)
    df_adjudication_queue = pd.concat([df_exceptions, df_unanimous_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_adjudication_blinded = df_adjudication_queue[[
        "pair_id", "claim_id", "claim_text", "gold_perspective_text", "generated_viewpoint_text"
    ]].copy()
    df_adjudication_blinded["human_label"] = ""
    df_adjudication_blinded["human_rationale"] = ""
    
    save_csv(df_out, csv_annotated_path)
    save_csv(df_human_triage, csv_triage_path)
    save_csv(df_adjudication_blinded, csv_adjudication_sample_path)
    
    report = {
        "benchmark_sample_size": len(df_out),
        "batch_size": batch_size,
        "total_batches": len(batches),
        "models_evaluated": ["Gemini-3-Flash", "GPT-OSS-120B", "Qwen-3.8-27B"],
        "inter_judge_agreement": {
            "cohen_kappa_gemini_vs_gpt": round(float(kappa_gem_gpt), 4),
            "cohen_kappa_gemini_vs_qwen": round(float(kappa_gem_qwen), 4),
            "cohen_kappa_gpt_vs_qwen": round(float(kappa_gpt_qwen), 4),
            "mean_pairwise_kappa": round(float(mean_kappa), 4)
        },
        "consensus_distribution": {
            "unanimous_3_of_3": int(unanimous_n),
            "unanimous_pct": round(float(unanimous_n / len(df_out) * 100), 1),
            "majority_2_of_3": int(majority_n),
            "majority_pct": round(float(majority_n / len(df_out) * 100), 1),
            "human_adjudication_required": int(human_n),
            "human_adjudication_pct": round(float(human_n / len(df_out) * 100), 1)
        },
        "telemetry_and_latencies": {
            model: {
                "mean_latency_ms": round(float(np.mean(vals["latencies"])), 1) if vals["latencies"] else 0,
                "median_latency_ms": round(float(np.median(vals["latencies"])), 1) if vals["latencies"] else 0,
                "p95_latency_ms": round(float(np.percentile(vals["latencies"], 95)), 1) if vals["latencies"] else 0,
                "total_prompt_tokens": vals["prompt_tokens"],
                "total_output_tokens": vals["output_tokens"]
            }
            for model, vals in telemetry.items()
        }
    }
    
    save_json(report, json_report_path)
    
    logger.info("=" * 80)
    logger.info("LIVE JURY BATCH RUN COMPLETE!")
    logger.info(f"Mean Pairwise Cohen's Kappa: {mean_kappa:.4f}")
    logger.info(f"Unanimous Consensus Rate: {unanimous_n}/{len(df_out)} ({unanimous_n/len(df_out)*100:.1f}%)")
    logger.info(f"Human Triage Cases: {human_n}/{len(df_out)} ({human_n/len(df_out)*100:.1f}%)")
    logger.info(f"Saved Annotated CSV to: {csv_annotated_path}")
    logger.info(f"Saved Human Triage Queue to: {csv_triage_path}")
    logger.info(f"Saved Comprehensive Report to: {json_report_path}")
    logger.info("=" * 80)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live Multi-LLM Jury Batch")
    parser.add_argument("--num_pairs", type=int, default=50, help="Number of pairs to evaluate")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of pairs per API call")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--input_csv", type=str, default="data/calibration/human_llm_benchmark_1700.csv")
    parser.add_argument("--output_dir", type=str, default="data/calibration")
    parser.add_argument("--results_dir", type=str, default="results/calibration")
    parser.add_argument("--env_file", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default="stage_b_250")
    
    args = parser.parse_args()
    run_live_jury_batch(
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        seed=args.seed,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        env_file=args.env_file,
        run_tag=args.run_tag
    )
