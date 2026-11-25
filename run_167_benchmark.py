"""
Self-contained script to build and run the 167 matched quadruplet benchmark:
  - 167 Arguments
  - 4 Perspective Types (True Positive, Opposite-Stance, Same-Stance Diff-Reason, Unrelated)
  - 2 Length Regimes: Full Discourse (180w) vs. Concise (58w)
  - Total: 1,336 evaluated pairs
"""

import os
import json
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Ensure matched quadruplets exist with verified stances
quad_path = "data/audit/stance_verified_matched_quadruplets.csv"
legacy_quad_path = "data/audit/matched_quadruplets_167.csv"

if not os.path.exists(quad_path):
    print("Building stance-verified matched quadruplets...")
    from src.analysis.build_stance_verified_matched_quadruplets import build_stance_verified_quadruplets
    df = build_stance_verified_quadruplets(
        input_quad_path=legacy_quad_path if os.path.exists(legacy_quad_path) else "data/audit/matched_quadruplets_167.csv",
        output_path=quad_path,
        sync_legacy_csv=True
    )
else:
    df = pd.read_csv(quad_path)

print("=" * 105)
print("EVALUATING 167 MATCHED QUADRUPLETS (1,336 PAIRS) ACROSS 4 TYPES & 2 LENGTH REGIMES")
print("Total Quadruplets:", len(df), "| Device:", device)
print("=" * 105)

# 2. Load V5 Verifier
cand_paths = ["models/ensemble_data_only/seed_42", "models/ensemble_data_only", "models/argument_nli_deberta_v3_v3"]
v5_dir = next((d for d in cand_paths if os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin"))), cand_paths[0])
print("Loading V5 Model from:", v5_dir)

tokenizer = AutoTokenizer.from_pretrained(v5_dir)
model = AutoModelForSequenceClassification.from_pretrained(v5_dir).to(device).eval()

# 3. Build Evaluation Records
types = [
    ("True Positive", "hyp_true_positive"),
    ("Opposite-Stance", "hyp_opposite_stance"),
    ("Same-Stance Diff-Reason", "hyp_same_stance_diff_reason"),
    ("Unrelated", "hyp_unrelated")
]

records = []
for length_regime, col_prem in [("Full (180w)", "full_premise_180w"), ("Concise (58w)", "concise_premise_58w")]:
    for ptype, col_hyp in types:
        for idx, r in df.iterrows():
            records.append({
                "length_regime": length_regime,
                "perspective_type": ptype,
                "premise": str(r[col_prem]).strip(),
                "hypothesis": str(r[col_hyp]).strip()
            })

eval_df = pd.DataFrame(records)

# 4. Batched Scoring
batch_size = 64
all_probs = []
for i in range(0, len(eval_df), batch_size):
    batch = eval_df.iloc[i : i + batch_size]
    enc = tokenizer(batch["premise"].tolist(), batch["hypothesis"].tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits / 1.9104
        probs = F.softmax(logits, dim=-1)
        entail_idx = 1 if logits.shape[-1] >= 2 else 0
        all_probs.extend(probs[:, entail_idx].cpu().numpy().tolist())

eval_df["p_entail"] = [round(float(p), 4) for p in all_probs]
eval_df["accepted_f1"] = eval_df["p_entail"] >= 0.52
eval_df["accepted_prec90"] = eval_df["p_entail"] >= 0.83

# Save detailed results
os.makedirs("results", exist_ok=True)
eval_df.to_csv("results/length_controlled_matched_benchmark.csv", index=False)

# 5. Summary Table
print("\n" + "=" * 105)
print("MATCHED QUADRUPLET FALSE ACCEPTANCE AUDIT TABLE (167 ARGS x 4 TYPES x 2 LENGTHS)")
print("=" * 105)
print("Length Regime   | Perspective Type          | Pairs | Mean P | Accept Rate (F1) | Rejection Rate | Accept (Prec90)")
print("-" * 105)

for length_regime in ["Full (180w)", "Concise (58w)"]:
    for ptype, _ in types:
        sub = eval_df[(eval_df["length_regime"] == length_regime) & (eval_df["perspective_type"] == ptype)]
        n = len(sub)
        mean_p = sub["p_entail"].mean()
        acc_f1 = sub["accepted_f1"].mean() * 100
        rej_f1 = 100.0 - acc_f1
        acc_p90 = sub["accepted_prec90"].mean() * 100
        print(f"{length_regime:<15} | {ptype:<25} | {n:<5} | {mean_p:<6.4f} | {acc_f1:>15.2f}% | {rej_f1:>13.2f}% | {acc_p90:>13.2f}%")
print("=" * 105 + "\n")
