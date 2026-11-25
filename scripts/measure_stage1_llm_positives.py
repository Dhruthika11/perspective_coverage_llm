import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv("data/nli_training_v3/test.csv")
llm_pos = df[df["pair_category"].str.startswith("long_llm_to_human_positive")].reset_index(drop=True)
print(f"Loaded {len(llm_pos)} long LLM-to-human positive pairs.")

model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on device: {device}")

def embed_texts(texts, batch_size=16, desc=""):
    all_embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        b_texts = texts[i : i + batch_size]
        enc = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
            all_embs.append(cls_emb.cpu().numpy())
        print(f"  [{desc}] Processed {min(i+batch_size, total)}/{total}...")
    return np.concatenate(all_embs, axis=0)

print("Embedding 188 premises...")
p_embs = embed_texts(llm_pos["premise"].tolist(), batch_size=16, desc="Premises")
print("Embedding 188 hypotheses...")
h_embs = embed_texts(llm_pos["hypothesis"].tolist(), batch_size=32, desc="Hypotheses")

sims = np.sum(p_embs * h_embs, axis=1)

print("\n" + "=" * 80)
print("STAGE 1 RETRIEVAL ON 188 LONG LLM POSITIVES (BAAI/bge-large-en-v1.5)")
print("=" * 80)
print(f"Mean Cosine:   {np.mean(sims):.4f}")
print(f"Median Cosine: {np.median(sims):.4f}")
print(f"Min Cosine:    {np.min(sims):.4f}")
print(f"Max Cosine:    {np.max(sims):.4f}")
print(f"Std Cosine:    {np.std(sims):.4f}")
print("-" * 80)
for t in [0.65, 0.70, 0.72, 0.75, 0.78, 0.80]:
    passed = int(np.sum(sims >= t))
    pct = (passed / len(sims)) * 100
    print(f"tau >= {t:.2f}: {passed:>3d} / {len(sims)} ({pct:>5.2f}%) pass  |  Dropped: {len(sims)-passed:>3d} ({100-pct:>5.2f}%)")
print("=" * 80)

