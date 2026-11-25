"""
Run this single script on Flair2 to extract text arguments from Gemma, Llama-3, OLMo, and Qwen
into one lightweight CSV file (approx. 1.5 - 2 MB):
    python extract_flair2_multimodel.py
"""

import os
import json
import pandas as pd

raw_dir = os.path.expanduser("~/perspective_coverage/data/llm_outputs/raw_generations")
output_csv = os.path.expanduser("~/perspective_coverage/data/llm_outputs/multimodel_extracted_arguments.csv")

target_models = [
    "gemma_2_9b_it",
    "llama_3_1_8b_instruct",
    "olmo_2_1124_7b_instruct",
    "qwen2_5_7b_instruct"
]

records = []
print(f"Scanning directories in {raw_dir}...")

for model_name in target_models:
    model_path = os.path.join(raw_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Directory not found: {model_path}")
        continue
    
    files = [f for f in os.listdir(model_path) if f.endswith(".json")]
    print(f"Found {len(files)} files in {model_name}...")
    
    for fname in files:
        cid = fname.split("_")[0]
        fpath = os.path.join(model_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            claim_text = data.get("claim_text", "")
            samples = data.get("samples", [])
            for s_idx, sample in enumerate(samples[:2]):  # Take top 2 viewpoints
                paragraphs = [p.strip() for p in str(sample).split("\n\n") if len(p.strip().split()) >= 20]
                for p in paragraphs:
                    records.append({
                        "model": model_name,
                        "claim_id": cid,
                        "claim_text": claim_text,
                        "argument_text": p
                    })
        except Exception as e:
            pass

df = pd.DataFrame(records)
print(f"\nTotal extracted arguments: {len(df)}")
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"Saved lightweight CSV to: {output_csv} ({os.path.getsize(output_csv) / (1024*1024):.2f} MB)")
print("Done! You can leave this file on the server or transfer this tiny file.")

