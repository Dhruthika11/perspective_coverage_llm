import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

print("Loading test.csv...", flush=True)
df = pd.read_csv("data/nli_training_v3/test.csv")
llm_pos = df[df["pair_category"].str.startswith("long_llm_to_human_positive")].reset_index(drop=True)
print(f"Loaded {len(llm_pos)} long LLM-to-human positive pairs.", flush=True)

model_name = "BAAI/bge-large-en-v1.5"
print(f"Loading {model_name}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on device: {device}", flush=True)

def embed_list(texts, batch_size=16, label=""):
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=192, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            cls_v = out.last_hidden_state[:, 0]
            cls_v = torch.nn.functional.normalize(cls_v, p=2, dim=1)
            embs.append(cls_v.cpu().numpy())
        print(f"  {label}: {min(i+batch_size, len(texts))}/{len(texts)} done", flush=True)
    return np.concatenate(embs, axis=0)

print("Embedding premises...", flush=True)
p_embs = embed_list(llm_pos["premise"].tolist(), batch_size=16, label="Premises")

print("Embedding hypotheses...", flush=True)
h_embs = embed_list(llm_pos["hypothesis"].tolist(), batch_size=32, label="Hypotheses")

sims = np.sum(p_embs * h_embs, axis=1)

print("\n" + "=" * 80, flush=True)
print("STAGE 1 RETRIEVAL RECALL ON 188 LONG LLM POSITIVES (BGE-Large)", flush=True)
print("=" * 80, flush=True)
print(f"Mean Cosine:   {np.mean(sims):.4f}", flush=True)
print(f"Median Cosine: {np.median(sims):.4f}", flush=True)
print(f"Min Cosine:    {np.min(sims):.4f}", flush=True)
print(f"Max Cosine:    {np.max(sims):.4f}", flush=True)
print(f"Std Cosine:    {np.std(sims):.4f}", flush=True)
print("-" * 80, flush=True)
for t in [0.65, 0.70, 0.72, 0.75, 0.78, 0.80]:
    passed = int(np.sum(sims >= t))
    pct = (passed / len(sims)) * 100
    print(f"tau >= {t:.2f}: {passed:>3d} / {len(sims)} ({pct:>5.2f}%) pass  |  Dropped (FN): {len(sims)-passed:>3d} ({100-pct:>5.2f}%)", flush=True)
print("=" * 80, flush=True)

