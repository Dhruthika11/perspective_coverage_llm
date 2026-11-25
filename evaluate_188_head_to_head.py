import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test dataset
test_csv = "data/nli_training_v3/test.csv"
df = pd.read_csv(test_csv)
print(f"Loaded {len(df)} total rows from {test_csv}")

# 1. Load V3 model
v3_path = "models/argument_nli_deberta_v3_v3"
print(f"Loading V3 model from {v3_path}...")
tokenizer_v3 = AutoTokenizer.from_pretrained(v3_path)
model_v3 = AutoModelForSequenceClassification.from_pretrained(v3_path).to(device)
model_v3.eval()

# 2. Load Baseline model (cached cross-encoder/nli-deberta-v3-small)
base_name = "cross-encoder/nli-deberta-v3-small"
print(f"Loading Baseline model from {base_name}...")
tokenizer_base = AutoTokenizer.from_pretrained(base_name)
model_base = AutoModelForSequenceClassification.from_pretrained(base_name).to(device)
model_base.eval()

def infer_probs(model, tokenizer, premises, hypotheses, batch_size=32, ent_idx=1):
    probs_list = []
    logits_list = []
    for i in range(0, len(premises), batch_size):
        p_b = premises[i : i + batch_size]
        h_b = hypotheses[i : i + batch_size]
        inputs = tokenizer(
            p_b, h_b,
            truncation=True, max_length=256, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.extend(probs)
            logits_list.extend(logits.cpu().numpy())
    return np.array(probs_list), np.array(logits_list)

# Determine entailment index for both models
id2label_v3 = getattr(model_v3.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
print("V3 id2label:", id2label_v3)
ent_idx_v3 = 1
for k, v in id2label_v3.items():
    if "entail" in str(v).lower():
        ent_idx_v3 = int(k)
        break

id2label_base = getattr(model_base.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
print("Baseline id2label:", id2label_base)
ent_idx_base = 1
for k, v in id2label_base.items():
    if "entail" in str(v).lower():
        ent_idx_base = int(k)
        break

# Run on 188 LLM positive pairs
llm_pos = df[df["source"] == "llm_generation"].copy().reset_index(drop=True)
print(f"\nEvaluating on {len(llm_pos)} LLM-to-human positive pairs...")

# V3: Premise = LLM (premise col), Hypothesis = Human (hypothesis col)
v3_probs_188, _ = infer_probs(model_v3, tokenizer_v3, llm_pos["premise"].tolist(), llm_pos["hypothesis"].tolist(), ent_idx=ent_idx_v3)
p_v3_entail_188 = v3_probs_188[:, ent_idx_v3]

# Baseline Legacy Inverted: Premise = Hypothesis, Hypothesis = Premise
base_inv_probs_188, _ = infer_probs(model_base, tokenizer_base, llm_pos["hypothesis"].tolist(), llm_pos["premise"].tolist(), ent_idx=ent_idx_base)
p_base_inv_entail_188 = base_inv_probs_188[:, ent_idx_base]

# Baseline Direction-Corrected: Premise = Premise, Hypothesis = Hypothesis
base_fwd_probs_188, _ = infer_probs(model_base, tokenizer_base, llm_pos["premise"].tolist(), llm_pos["hypothesis"].tolist(), ent_idx=ent_idx_base)
p_base_fwd_entail_188 = base_fwd_probs_188[:, ent_idx_base]

llm_pos["p_v3"] = p_v3_entail_188
llm_pos["p_base_inv"] = p_base_inv_entail_188
llm_pos["p_base_fwd"] = p_base_fwd_entail_188

for theta in [0.50, 0.72, 0.80, 0.90]:
    tp_v3 = (p_v3_entail_188 >= theta).sum()
    tp_base_inv = (p_base_inv_entail_188 >= theta).sum()
    tp_base_fwd = (p_base_fwd_entail_188 >= theta).sum()
    print(f"\n--- At Theta = {theta:.2f} (Total = 188) ---")
    print(f"V3 True Positives: {tp_v3} ({tp_v3/188*100:.2f}%)")
    print(f"Baseline (Legacy Inverted) TP: {tp_base_inv} ({tp_base_inv/188*100:.2f}%)")
    print(f"Baseline (Forward / Zero-Shot) TP: {tp_base_fwd} ({tp_base_fwd/188*100:.2f}%)")

# Detailed breakdown at theta = 0.80 and theta = 0.50 by length tier
for theta in [0.50, 0.80]:
    print(f"\n================ Breakdown by Length Tier at Theta = {theta:.2f} ================")
    for tier in ["short", "medium", "long", "very_long"]:
        tier_df = llm_pos[llm_pos["length_tier"] == tier]
        n_tier = len(tier_df)
        if n_tier == 0: continue
        rec_v3 = (tier_df["p_v3"] >= theta).sum() / n_tier
        rec_base_inv = (tier_df["p_base_inv"] >= theta).sum() / n_tier
        rec_base_fwd = (tier_df["p_base_fwd"] >= theta).sum() / n_tier
        print(f"Tier: {tier:10s} (N={n_tier:2d}) | V3: {rec_v3*100:5.1f}% | Base (Inverted): {rec_base_inv*100:5.1f}% | Base (Fwd): {rec_base_fwd*100:5.1f}%")

# Save detailed results
llm_pos.to_csv("results/paper_tables/llm_188_head_to_head.csv", index=False)
print("\nSaved results/paper_tables/llm_188_head_to_head.csv")

