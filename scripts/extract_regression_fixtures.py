#!/usr/bin/env python3
"""
Extract Real-World Regression Test Fixtures from Raw Model Generations on HPC.

Collects 10 representative failure cases discovered during the forensic audit:
  1. multi_json.txt          - Multiple concatenated JSON objects
  2. nested_json.txt         - Nested JSON structure with reasoning/aspect dictionaries
  3. markdown_viewpoints.txt - Markdown viewpoint headings (### Perspective 1:, **Stance:**)
  4. numbered_viewpoints.txt - Numbered viewpoint lists (1. ... 2. ...)
  5. missing_stance.txt      - Generation without literal 'supportive/critical viewpoint' prefix
  6. duplicate_viewpoints.txt- Repetitive loops from small models (e.g. SmolLM2)
  7. browser_error.txt       - Corrupted execution / system error dumps
  8. unrelated_topic.txt     - Exam question / off-topic leakage
  9. merged_viewpoints.txt   - Multiple distinct perspectives collapsed in one paragraph
 10. truncated_generation.txt- Trailing mid-token generation cutoff

Saves all fixtures to tests/fixtures/ on HPC and writes a manifest of fixture origins.
"""

import os
import re
import json
import pandas as pd

BASE_DIR = os.path.expanduser("~/perspective_coverage")
RAW_DIR = os.path.join(BASE_DIR, "data", "llm_outputs", "raw_generations")
FIXTURES_DIR = os.path.join(BASE_DIR, "tests", "fixtures")

os.makedirs(FIXTURES_DIR, exist_ok=True)

print("=" * 80)
print(f"EXTRACTING REGRESSION TEST FIXTURES: {FIXTURES_DIR}")
print("=" * 80)

fixtures_found = {}

# Helper to load raw csv for a model
def load_model_raw(model_key):
    csv_path = os.path.join(RAW_DIR, model_key, "raw_generations.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# 1. Corrupted execution / browser error (olmo_2_0425_1b, cId 710, sId 3)
df_olmo_1b = load_model_raw("olmo_2_0425_1b")
if df_olmo_1b is not None:
    match = df_olmo_1b[(df_olmo_1b["claim_id"] == 710) & (df_olmo_1b["sample_id"] == 3)]
    if len(match) > 0:
        txt = str(match["generated_text"].values[0])
        fixtures_found["browser_error"] = {
            "model_key": "olmo_2_0425_1b",
            "claim_id": 710,
            "sample_id": 3,
            "text": txt
        }

# 2. Numbered Viewpoints (qwen2_5_32b_instruct, claim_id 1000, sample_id 0)
df_qwen32b = load_model_raw("qwen2_5_32b_instruct")
if df_qwen32b is not None:
    for _, r in df_qwen32b.iterrows():
        t = str(r["generated_text"])
        if re.search(r'(?:\A|\n)1[\.\)]\s+', t) and re.search(r'\n2[\.\)]\s+', t) and len(t.split()) > 100:
            fixtures_found["numbered_viewpoints"] = {
                "model_key": "qwen2_5_32b_instruct",
                "claim_id": int(r["claim_id"]),
                "sample_id": int(r["sample_id"]),
                "text": t
            }
            break

# 3. Markdown Viewpoints (qwen2_5_1_5b_instruct or qwen2_5_32b_instruct)
if df_qwen32b is not None:
    for _, r in df_qwen32b.iterrows():
        t = str(r["generated_text"])
        if "###" in t and ("Perspective" in t or "Viewpoint" in t):
            fixtures_found["markdown_viewpoints"] = {
                "model_key": "qwen2_5_32b_instruct",
                "claim_id": int(r["claim_id"]),
                "sample_id": int(r["sample_id"]),
                "text": t
            }
            break

# 4. Multi-JSON (concatenated JSON blocks)
# Check llama_3_1_8b_instruct or gemma_2_9b_it or olmo
for m in ["llama_3_1_8b_instruct", "llama_3_2_3b_instruct", "gemma_2_9b_it", "olmo_2_1124_13b"]:
    df_m = load_model_raw(m)
    if df_m is not None:
        for _, r in df_m.iterrows():
            t = str(r["generated_text"])
            # Find multiple '{' and '}' pairs
            if t.count("{") >= 2 and t.count("}") >= 2 and ("}\s*{" in t or "}\n{" in t or "```\n```" in t or t.count("\"stance\"") >= 2):
                fixtures_found["multi_json"] = {
                    "model_key": m,
                    "claim_id": int(r["claim_id"]),
                    "sample_id": int(r["sample_id"]),
                    "text": t
                }
                break
    if "multi_json" in fixtures_found:
        break

# 5. Nested JSON (reasoning/aspect sub-dictionaries)
for m in ["gemma_2_27b_it", "gemma_2_9b_it", "llama_3_1_8b_instruct"]:
    df_m = load_model_raw(m)
    if df_m is not None:
        for _, r in df_m.iterrows():
            t = str(r["generated_text"])
            if "{" in t and ("\"aspect\"" in t or "\"reasoning\"" in t or "\"justification\"" in t) and t.count("{") >= 2:
                fixtures_found["nested_json"] = {
                    "model_key": m,
                    "claim_id": int(r["claim_id"]),
                    "sample_id": int(r["sample_id"]),
                    "text": t
                }
                break
    if "nested_json" in fixtures_found:
        break

# If nested_json not found above, look for any JSON with nested braces
if "nested_json" not in fixtures_found:
    for m in ["llama_3_1_8b_instruct", "olmo_2_0325_32b"]:
        df_m = load_model_raw(m)
        if df_m is not None:
            for _, r in df_m.iterrows():
                t = str(r["generated_text"])
                if re.search(r'\{\s*"[a-zA-Z_]+":\s*\{', t):
                    fixtures_found["nested_json"] = {
                        "model_key": m,
                        "claim_id": int(r["claim_id"]),
                        "sample_id": int(r["sample_id"]),
                        "text": t
                    }
                    break
        if "nested_json" in fixtures_found:
            break

# 6. Missing Stance (valid viewpoint where legacy regex s[:35] missed stance because it lacked literal 'supportive/critical viewpoint')
for m in ["llama_3_2_1b_instruct", "olmo_2_1124_7b", "olmo_2_0325_32b", "gemma_2_2b_it"]:
    df_m = load_model_raw(m)
    if df_m is not None:
        for _, r in df_m.iterrows():
            t = str(r["generated_text"]).strip()
            # Stance is present semantically or as heading or in body, but s[:35] doesn't start with "supportive viewpoint"
            if len(t.split()) > 20 and not t.lower().startswith("supportive viewpoint") and not t.lower().startswith("critical viewpoint"):
                if "support" in t.lower() or "undermine" in t.lower() or "oppose" in t.lower() or "agree" in t.lower():
                    fixtures_found["missing_stance"] = {
                        "model_key": m,
                        "claim_id": int(r["claim_id"]),
                        "sample_id": int(r["sample_id"]),
                        "text": t
                    }
                    break
        if "missing_stance" in fixtures_found:
            break

# 7. Duplicate Viewpoints (SmolLM2-135M loop repetition)
df_smol = load_model_raw("smollm2_135m_instruct")
if df_smol is not None:
    for _, r in df_smol.iterrows():
        t = str(r["generated_text"])
        # Check repeated sentence / phrase
        sentences = [s.strip() for s in t.split(".") if len(s.strip()) > 15]
        if len(sentences) > len(set(sentences)) + 2:
            fixtures_found["duplicate_viewpoints"] = {
                "model_key": "smollm2_135m_instruct",
                "claim_id": int(r["claim_id"]),
                "sample_id": int(r["sample_id"]),
                "text": t
            }
            break

# 8. Unrelated Topic / Exam answer leakage (olmo_2_0425_1b)
if df_olmo_1b is not None:
    for _, r in df_olmo_1b.iterrows():
        t = str(r["generated_text"])
        if "Question:" in t or "Answer:" in t or "Solution:" in t or "Exam" in t or "multiple choice" in t.lower():
            fixtures_found["unrelated_topic"] = {
                "model_key": "olmo_2_0425_1b",
                "claim_id": int(r["claim_id"]),
                "sample_id": int(r["sample_id"]),
                "text": t
            }
            break

# 9. Merged Viewpoints (Multiple viewpoints merged in monolithic prose)
df_olmo_32b = load_model_raw("olmo_2_0325_32b")
if df_olmo_32b is not None:
    for _, r in df_olmo_32b.iterrows():
        t = str(r["generated_text"])
        # Prose with multiple contrastive conjunctions/arguments
        if ("On the one hand" in t or "However" in t or "Conversely" in t or "Furthermore" in t) and len(t.split()) > 80 and "{" not in t:
            fixtures_found["merged_viewpoints"] = {
                "model_key": "olmo_2_0325_32b",
                "claim_id": int(r["claim_id"]),
                "sample_id": int(r["sample_id"]),
                "text": t
            }
            break

# 10. Truncated Generation (cut off mid-token or mid-json)
for m in ["llama_3_2_1b_instruct", "olmo_2_0425_1b", "qwen2_5_0_5b_instruct", "smollm2_360m_instruct"]:
    df_m = load_model_raw(m)
    if df_m is not None:
        for _, r in df_m.iterrows():
            t = str(r["generated_text"]).strip()
            # Ends without period, brace, or closing quote
            if len(t) > 100 and not t.endswith((".", "\"", "}", ">", ")", "```")):
                fixtures_found["truncated_generation"] = {
                    "model_key": m,
                    "claim_id": int(r["claim_id"]),
                    "sample_id": int(r["sample_id"]),
                    "text": t
                }
                break
    if "truncated_generation" in fixtures_found:
        break

# Write fixtures to files
fixture_metadata = []
for name, data in fixtures_found.items():
    file_path = os.path.join(FIXTURES_DIR, f"{name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data["text"])
    fixture_metadata.append({
        "fixture_name": name,
        "filename": f"{name}.txt",
        "model_key": data["model_key"],
        "claim_id": data["claim_id"],
        "sample_id": data["sample_id"],
        "text_length": len(data["text"]),
        "words": len(data["text"].split())
    })
    print(f"  [SAVED] {name:<24} from {data['model_key']} (cId: {data['claim_id']}, sId: {data['sample_id']}, {len(data['text'])} chars)")

meta_df = pd.DataFrame(fixture_metadata)
meta_path = os.path.join(FIXTURES_DIR, "fixtures_manifest.csv")
meta_df.to_csv(meta_path, index=False)
print("=" * 80)
print(f"Total fixtures saved: {len(fixture_metadata)}/10")
print(f"Fixtures manifest saved to: {meta_path}")
print("=" * 80)
