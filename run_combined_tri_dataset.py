"""
Full Pipeline Script to Combine All Three Datasets and Compute Coverage.
1. Checks for IBM Debater and Kialo claims/perspectives (or parses them from Hugging Face if missing).
2. Merges Perspectrum, IBM Debater, and Kialo into master unified claims and perspectives CSVs:
     data/processed/combined_tri_dataset/claims.csv
     data/processed/combined_tri_dataset/perspectives.csv
3. Merges model generations across datasets for each model.
4. Computes calibrated two-stage coverage across the combined master benchmark.
5. Produces cross-dataset breakdown tables.
"""
import os
import glob
import json
import pandas as pd
import numpy as np

BASE_DIR = "."
OUT_PROCESSED = "data/processed/combined_tri_dataset"
OUT_RESULTS = "results/combined_tri_dataset"
os.makedirs(OUT_PROCESSED, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

print("=" * 70)
print("TRI-DATASET UNIFIED PIPELINE (PERSPECTRUM + IBM DEBATER + KIALO)")
print("=" * 70)

# Check datasets
datasets = ["perspectrum", "ibm_debater", "kialo"]
print("Inspecting available generations across models on HPC:")

# Models to look for
models = [
    "qwen2_5_0_5b_instruct", "qwen2_5_1_5b_instruct", "qwen2_5_3b_instruct",
    "qwen2_5_7b_instruct", "qwen2_5_14b_instruct", "qwen2_5_32b_instruct",
    "llama_3_2_1b_instruct", "llama_3_2_3b_instruct",
    "gemma_2_2b_it", "gemma_3_12b_it", "mistral_7b_instruct_v0_3"
]

for m in models:
    p_gen = f"data/llm_outputs/{m}/processed_generations.csv"
    ibm_gen = f"data/llm_outputs/ibm_debater/{m}/processed_generations.csv"
    if not os.path.exists(ibm_gen):
        ibm_gen = f"data/llm_outputs/ibm_debater/{m}.csv"
    kialo_gen = f"data/llm_outputs/kialo/{m}/processed_generations.csv"
    if not os.path.exists(kialo_gen):
        kialo_gen = f"data/llm_outputs/kialo/{m}.csv"
    
    print(f"\nModel: {m}")
    print(f"  Perspectrum: {os.path.exists(p_gen)}")
    print(f"  IBM Debater: {os.path.exists(ibm_gen)} ({ibm_gen})")
    print(f"  Kialo:       {os.path.exists(kialo_gen)} ({kialo_gen})")

print("\n" + "=" * 70)
