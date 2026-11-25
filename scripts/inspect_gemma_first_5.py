"""
Inspect the First 5 Rows of Gemma-2-9B Processed Generations.

Usage on HPC:
    python scripts/inspect_gemma_first_5.py
"""

import os
import pandas as pd

CSV_PATH = "data/llm_outputs/gemma_2_9b_it/processed_generations.csv"

if not os.path.exists(CSV_PATH):
    # Try absolute path on HPC
    CSV_PATH = os.path.expanduser("~/perspective_coverage/data/llm_outputs/gemma_2_9b_it/processed_generations.csv")

if not os.path.exists(CSV_PATH):
    print(f"Error: File not found at {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH, nrows=5)

print("=" * 100)
print(f"GEMMA-2-9B PROCESSED GENERATIONS: FIRST 5 ROWS")
print(f"Source: {CSV_PATH}")
print(f"Columns: {list(df.columns)}")
print("=" * 100)

for idx, row in df.iterrows():
    print(f"\n--- [ROW {idx + 1} / 5] ---")
    print(f"Claim ID:     {row.get('claim_id')}")
    print(f"Sample ID:    {row.get('sample_id')}")
    if "llm_reported_stance" in row and pd.notna(row["llm_reported_stance"]):
        print(f"Stance:       {row['llm_reported_stance']}")
    
    if "perspective_summary" in row and pd.notna(row["perspective_summary"]):
        p_sum = str(row["perspective_summary"]).strip()
        print(f"Summary ({len(p_sum.split())}w):   {p_sum}")
        
    if "core_reasoning" in row and pd.notna(row["core_reasoning"]):
        c_reas = str(row["core_reasoning"]).strip()
        print(f"Reasoning ({len(c_reas.split())}w): {c_reas}")
        
    gen_text = str(row.get("generated_text", row.get("argument", row.get("combined_text", "")))).strip()
    w_cnt = len(gen_text.split())
    print(f"Raw Text ({w_cnt}w):")
    print(gen_text[:400] + ("..." if len(gen_text) > 400 else ""))
    print("-" * 100)
