"""
Audit Generation Formats Across All Models on HPC.

Classifies each model's raw/processed outputs into:
  1. JSON Format (e.g. {"stance": ..., "perspective_summary": ...})
  2. Multi-Viewpoint Format (Numbered lists '1. ... 2. ...', Markdown headers '### ')
  3. Plain Text / Monolithic Prose

Usage on HPC:
    python scripts/audit_generation_formats_on_hpc.py
"""

import os
import glob
import re
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.expanduser("~/perspective_coverage/data/llm_outputs")
if not os.path.exists(BASE_DIR):
    BASE_DIR = "data/llm_outputs"

print("=" * 110)
print(f"AUDITING MODEL GENERATION FORMATS: {BASE_DIR}")
print("=" * 110)

subdirs = sorted([
    d for d in glob.glob(os.path.join(BASE_DIR, "*"))
    if os.path.isdir(d) and os.path.basename(d) not in ["raw_generations", "ibm_debater", "kialo"]
])

if not subdirs:
    print(f"No model directories found in {BASE_DIR}!")
    exit(1)

results = []

for sdir in subdirs:
    m_name = os.path.basename(sdir)
    csv_path = os.path.join(sdir, "processed_generations.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(sdir, "raw_generations.csv")
        
    if not os.path.exists(csv_path):
        continue
        
    df = pd.read_csv(csv_path)
    text_col = "generated_text" if "generated_text" in df.columns else ("argument" if "argument" in df.columns else "combined_text")
    
    total_samples = len(df)
    unique_claims = df["claim_id"].nunique() if "claim_id" in df.columns else "N/A"
    
    n_json = 0
    n_multi_vp = 0
    n_plain = 0
    word_counts = []
    
    for t in df[text_col].dropna():
        t_str = str(t).strip()
        words = len(t_str.split())
        word_counts.append(words)
        
        # Check 1: JSON
        is_json = False
        t_clean = re.sub(r'^```(?:json)?\s*', '', t_str, flags=re.IGNORECASE)
        t_clean = re.sub(r'\s*```$', '', t_clean.strip())
        if (t_clean.startswith("{") and ("perspective_summary" in t_clean or "stance" in t_clean or "summary" in t_clean)):
            is_json = True
        elif "{" in t_clean and "}" in t_clean and ("\"stance\"" in t_clean or "\"perspective_summary\"" in t_clean):
            is_json = True
            
        if is_json:
            n_json += 1
            continue
            
        # Check 2: Multi-Viewpoint (Numbered items or Markdown section headers)
        has_numbered_list = bool(re.search(r'(?:\A|\n+)\s*(?:1[\.\)]|\bViewpoint\s+1|\bPerspective\s+1)\s+', t_str, flags=re.IGNORECASE) and
                                re.search(r'\n+\s*(?:2[\.\)]|\bViewpoint\s+2|\bPerspective\s+2)\s+', t_str, flags=re.IGNORECASE))
        has_aspect_headers = bool(re.search(r'###\s+[A-Za-z]+', t_str) or re.search(r'\*\*[A-Za-z\s]+Perspective:?\*\*', t_str))
        
        if has_numbered_list or has_aspect_headers:
            n_multi_vp += 1
        else:
            n_plain += 1
            
    pct_json = (n_json / total_samples) * 100 if total_samples > 0 else 0
    pct_multi = (n_multi_vp / total_samples) * 100 if total_samples > 0 else 0
    pct_plain = (n_plain / total_samples) * 100 if total_samples > 0 else 0
    median_words = np.median(word_counts) if word_counts else 0
    
    # Primary classification
    if pct_json >= 70:
        primary_fmt = "Structured JSON"
    elif pct_multi >= 70:
        primary_fmt = "Multi-Viewpoint List"
    elif pct_json > pct_multi:
        primary_fmt = f"Mixed (JSON {pct_json:.0f}%)"
    else:
        primary_fmt = f"Mixed (Multi-VP {pct_multi:.0f}%)"
        
    results.append({
        "Model": m_name,
        "Claims": unique_claims,
        "Total Rows": total_samples,
        "Median Words": f"{median_words:.0f}w",
        "JSON %": f"{pct_json:.1f}%",
        "Multi-VP %": f"{pct_multi:.1f}%",
        "Plain %": f"{pct_plain:.1f}%",
        "Format Class": primary_fmt
    })

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
print("=" * 110)
print("\nSUMMARY:")
print("  * 'Structured JSON': Generates 1 atomic JSON object per sample (summary + reasoning).")
print("  * 'Multi-Viewpoint List': Generates 2-6 numbered viewpoints with Markdown headers per sample.")
print("=" * 110)
