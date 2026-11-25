"""
Auto-Runner for Missing Model Generations (IBM Debater & Kialo)

Checks which models are missing their `processed_generations.csv` under:
- data/llm_outputs/ibm_debater/
- data/llm_outputs/kialo/

And sequentially runs:
  python run_pipeline.py --dataset <dataset> --model <model_key> --stage generation
"""
import os
import sys
import subprocess

# Model key mapping: folder_tag -> config_model_key
MODEL_KEY_MAP = {
    "qwen2_5_0_5b_instruct": "0.5b",
    "qwen2_5_1_5b_instruct": "1.5b",
    "qwen2_5_3b_instruct": "3b",
    "qwen2_5_7b_instruct": "qwen-7b",
    "qwen2_5_14b_instruct": "14b",
    "qwen2_5_32b_instruct": "32b",
    "llama_3_2_1b_instruct": "llama-1b",
    "llama_3_2_3b_instruct": "llama-3b",
    "gemma_2_2b_it": "gemma-2b",
    "gemma_2_9b_it": "gemma-9b",
    "gemma_3_12b_it": "12b",
    "mistral_7b_instruct_v0_3": "mistral-7b"
}

TARGET_DATASETS = ["ibm_debater", "kialo"]

print("=" * 70)
print("INSPECTING MISSING GENERATIONS ACROSS IBM DEBATER & KIALO")
print("=" * 70)

missing_jobs = []

for dset in TARGET_DATASETS:
    print(f"\nChecking dataset: {dset}")
    for folder_tag, model_key in MODEL_KEY_MAP.items():
        processed_path = f"data/llm_outputs/{dset}/{folder_tag}/processed_generations.csv"
        if os.path.exists(processed_path):
            size_mb = os.path.getsize(processed_path) / (1024 * 1024)
            print(f"  [EXISTS] {folder_tag:26} ({size_mb:.2f} MB)")
        else:
            print(f"  [MISSING] {folder_tag:26} -> Need to run with --model {model_key}")
            missing_jobs.append((dset, model_key, folder_tag))

if not missing_jobs:
    print("\n" + "=" * 70)
    print("ALL MODELS ALREADY HAVE GENERATIONS! Nothing to generate.")
    print("=" * 70)
    sys.exit(0)

print("\n" + "=" * 70)
print(f"FOUND {len(missing_jobs)} MISSING GENERATION JOBS:")
for dset, mkey, ftag in missing_jobs:
    print(f"  - Dataset: {dset:12} | Model Key: {mkey:10} | Target: {ftag}")
print("=" * 70)

for idx, (dset, mkey, ftag) in enumerate(missing_jobs, 1):
    cmd = [sys.executable, "run_pipeline.py", "--dataset", dset, "--model", mkey, "--stage", "generation"]
    print(f"\n[{idx}/{len(missing_jobs)}] EXECUTING: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"ERROR: Generation failed for {dset} {mkey} (exit code {ret.returncode}). Continuing with remaining...")
    else:
        print(f"SUCCESS: Completed generations for {dset} {mkey}!")

print("\n" + "=" * 70)
print("ALL MISSING GENERATIONS COMPLETED!")
print("=" * 70)
