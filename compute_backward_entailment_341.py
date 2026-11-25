import time
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 80)
print("COMPUTING BACKWARD ENTAILMENT (Gold -> Gen) FOR ALL 341 BENCHMARK PAIRS")
print("=" * 80)

# Load existing 341 predictions
in_path = "results/calibration/stage2_cross_encoder_predictions_341.csv"
df = pd.read_csv(in_path)
print(f"Loaded {len(df)} pairs from {in_path}")

# Model path
model_path = "models/argument_nli_deberta_v3_v3"
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print(f"Loaded model in {time.time() - t0:.2f}s")

# In forward: Premise = Gen, Hypothesis = Gold (already in p_entail)
# In backward: Premise = Gold, Hypothesis = Gen
premises_bwd = df['gold_perspective_text'].tolist()
hypotheses_bwd = df['generated_viewpoint_text'].tolist()

batch_size = 32
bwd_contra = []
bwd_entail = []
bwd_neutral = []

t1 = time.time()
for i in range(0, len(premises_bwd), batch_size):
    p_b = premises_bwd[i : i + batch_size]
    h_b = hypotheses_bwd[i : i + batch_size]
    inputs = tokenizer(p_b, h_b, truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()
    bwd_contra.extend(probs[:, 0])
    bwd_entail.extend(probs[:, 1])
    bwd_neutral.extend(probs[:, 2])
    print(f"Processed {min(i + batch_size, len(premises_bwd))}/{len(premises_bwd)} pairs...")

print(f"Inference completed in {time.time() - t1:.2f}s")

df['p_contra_bwd'] = [round(float(x), 6) for x in bwd_contra]
df['p_entail_bwd'] = [round(float(x), 6) for x in bwd_entail]
df['p_neutral_bwd'] = [round(float(x), 6) for x in bwd_neutral]

# Forward entailment
# Note: df['p_entail'] already has forward entailment
df['p_entail_fwd'] = df['p_entail']
df['bidirectional_min_entail'] = np.minimum(df['p_entail_fwd'], df['p_entail_bwd'])
df['bidirectional_asymmetry'] = np.abs(df['p_entail_fwd'] - df['p_entail_bwd'])

out_path = "results/calibration/stage2_bidirectional_predictions_341.csv"
df.to_csv(out_path, index=False)
print(f"Saved bidirectional predictions to {out_path}")
