"""
Merge script for Tri-Dataset Evaluation: Perspectrum + IBM Debater + Kialo.
Runs directly on the HPC server (or locally if paths exist).

Finds claims, perspectives, and model generations across all 3 datasets,
assigns clean globally unique IDs, and outputs unified master files:
  data/processed/combined_tri_dataset/claims.csv
  data/processed/combined_tri_dataset/perspectives.csv
  data/llm_outputs/{model}/combined_tri_generations.csv
"""
import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = "."
PROCESSED_OUT = "data/processed/combined_tri_dataset"
os.makedirs(PROCESSED_OUT, exist_ok=True)

print("=" * 60)
print("TRI-DATASET MERGER & INSPECTION")
print("=" * 60)

# 1. Search for Claims & Perspectives across datasets
dataset_locations = {
    "perspectrum": {
        "claims": "data/processed/claims.csv",
        "perspectives": "data/processed/perspectives.csv",
        "gen_base": "data/llm_outputs"
    },
    "ibm_debater": {
        "claims": ["data/processed/ibm_debater/claims.csv", "data/ibm_debater/claims.csv"],
        "perspectives": ["data/processed/ibm_debater/perspectives.csv", "data/ibm_debater/perspectives.csv"],
        "gen_base": "data/llm_outputs/ibm_debater"
    },
    "kialo": {
        "claims": ["data/processed/kialo/claims.csv", "data/kialo/claims.csv"],
        "perspectives": ["data/processed/kialo/perspectives.csv", "data/kialo/perspectives.csv"],
        "gen_base": "data/llm_outputs/kialo"
    }
}

# Inspect what exists
found_files = {}
for dname, paths in dataset_locations.items():
    print(f"\nChecking dataset: {dname.upper()}")
    c_path = paths["claims"] if isinstance(paths["claims"], str) else next((p for p in paths["claims"] if os.path.exists(p)), None)
    p_path = paths["perspectives"] if isinstance(paths["perspectives"], str) else next((p for p in paths["perspectives"] if os.path.exists(p)), None)
    
    print(f"  Claims file: {c_path} (Exists: {os.path.exists(str(c_path))})")
    print(f"  Perspectives file: {p_path} (Exists: {os.path.exists(str(p_path))})")
    print(f"  Generation dir: {paths['gen_base']} (Exists: {os.path.exists(paths['gen_base'])})")
    
    if os.path.exists(paths['gen_base']):
        gen_files = glob.glob(os.path.join(paths['gen_base'], "**/*.csv"), recursive=True)
        print(f"  Found {len(gen_files)} generation CSV files in {paths['gen_base']}")
        for gf in gen_files[:5]:
            print(f"    - {gf} ({os.path.getsize(gf)} bytes)")
        if len(gen_files) > 5:
            print(f"    ... and {len(gen_files)-5} more")

print("\n" + "=" * 60)
