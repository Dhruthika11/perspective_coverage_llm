import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

print("Starting unified 1,118-pair evaluation...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

test_csv = "data/nli_training_v3/test.csv"
df = pd.read_csv(test_csv)
print(f"Loaded {len(df)} rows from {test_csv}", flush=True)

# 1. Load V3
from transformers import AutoTokenizer, AutoModelForSequenceClassification

v3_path = "models/argument_nli_deberta_v3_v3"
print(f"Loading V3 from {v3_path}...", flush=True)
tokenizer_v3 = AutoTokenizer.from_pretrained(v3_path)
model_v3 = AutoModelForSequenceClassification.from_pretrained(v3_path).to(device)
model_v3.eval()

# 2. Load Baseline
base_path = "cross-encoder/nli-deberta-v3-small"
print(f"Loading Baseline from {base_path}...", flush=True)
tokenizer_base = AutoTokenizer.from_pretrained(base_path)
model_base = AutoModelForSequenceClassification.from_pretrained(base_path).to(device)
model_base.eval()

def infer_all_probs(model, tokenizer, premises, hypotheses, batch_size=64):
    probs_list = []
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
    return np.array(probs_list)

premises = df["premise"].fillna("").astype(str).tolist()
hypotheses = df["hypothesis"].fillna("").astype(str).tolist()

# V3 Forward
print("Running V3 forward inference on 1,118 rows...", flush=True)
v3_probs = infer_all_probs(model_v3, tokenizer_v3, premises, hypotheses)

# Baseline Forward (Zero-Shot)
print("Running Baseline forward (zero-shot) inference on 1,118 rows...", flush=True)
base_fwd_probs = infer_all_probs(model_base, tokenizer_base, premises, hypotheses)

# Baseline Inverted (Legacy)
print("Running Baseline inverted (legacy) inference on 1,118 rows...", flush=True)
base_inv_probs = infer_all_probs(model_base, tokenizer_base, hypotheses, premises)

# Label alignment:
# In V3 & DeBERTa NLI: index 0 = contradiction, index 1 = entailment, index 2 = neutral
# Let's verify config
print("V3 id2label:", model_v3.config.id2label, flush=True)
print("Base id2label:", model_base.config.id2label, flush=True)

df["v3_p_contra"] = v3_probs[:, 0]
df["v3_p_entail"] = v3_probs[:, 1]
df["v3_p_neutral"] = v3_probs[:, 2]

df["base_fwd_p_contra"] = base_fwd_probs[:, 0]
df["base_fwd_p_entail"] = base_fwd_probs[:, 1]
df["base_fwd_p_neutral"] = base_fwd_probs[:, 2]

df["base_inv_p_contra"] = base_inv_probs[:, 0]
df["base_inv_p_entail"] = base_inv_probs[:, 1]
df["base_inv_p_neutral"] = base_inv_probs[:, 2]

# Save master predictions CSV
master_csv = "results/metrics/master_1118_predictions.csv"
os.makedirs(os.path.dirname(master_csv), exist_ok=True)
df.to_csv(master_csv, index=False)
print(f"Saved master predictions to {master_csv}", flush=True)

