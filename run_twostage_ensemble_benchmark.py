"""
Production Two-Stage Benchmark Runner for Matched Quadruplets.

Evaluates the complete cascade:
  Stage 1: BGE-Large Dense Embedding Cosine Retrieval (tau* = 0.75)
  Stage 2: 3-Seed Calibrated V5 DeBERTa-v3 Ensemble Verifier (T* = 1.9104, theta* = 0.52 / 0.83)
  Combined: Two-Stage System Pass = (Stage 1 Pass) AND (Stage 2 Pass)

Audits 154 Stance-Verified Quadruplets (1,232 total pairs) across:
  - 4 Perspective Types (True Positive, Opposite-Stance, Same-Stance Diff-Reason, Unrelated)
  - 2 Length Regimes (Full 180w vs. Concise 58w)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 110)
print("PRODUCTION TWO-STAGE (BGE-LARGE + V5 ENSEMBLE) BENCHMARK")
print(f"Device: {device}")
print("=" * 110)

# ---------------------------------------------------------------------------
# 1. Load Stance-Verified Dataset
# ---------------------------------------------------------------------------
quad_cand = [
    "data/audit/stance_verified_matched_quadruplets.csv",
    "data/audit/matched_quadruplets_167.csv"
]
quad_path = next((p for p in quad_cand if os.path.exists(p)), quad_cand[0])
df_quads = pd.read_csv(quad_path)
print(f"Loaded {len(df_quads)} quadruplets from: {quad_path}")

types = [
    ("True Positive", "hyp_true_positive"),
    ("Opposite-Stance", "hyp_opposite_stance"),
    ("Same-Stance Diff-Reason", "hyp_same_stance_diff_reason"),
    ("Unrelated", "hyp_unrelated")
]

records = []
for length_regime, col_prem in [("Full (180w)", "full_premise_180w"), ("Concise (58w)", "concise_premise_58w")]:
    for ptype, col_hyp in types:
        for idx, r in df_quads.iterrows():
            records.append({
                "quad_id": r["quad_id"],
                "claim_id": r["claim_id"],
                "length_regime": length_regime,
                "perspective_type": ptype,
                "premise": str(r[col_prem]).strip(),
                "hypothesis": str(r[col_hyp]).strip()
            })

eval_df = pd.DataFrame(records)
print(f"Total pairs to evaluate: {len(eval_df)}")

# ---------------------------------------------------------------------------
# 2. Stage 1: BGE-Large Dense Embedding Cosine Retrieval
# ---------------------------------------------------------------------------
bge_model_name = "BAAI/bge-large-en-v1.5"
print("-" * 80)
print(f"[Stage 1] Loading BGE Embedder: {bge_model_name}...")
bge_tok = AutoTokenizer.from_pretrained(bge_model_name)
bge_mod = AutoModel.from_pretrained(bge_model_name).to(device).eval()

def embed_texts(texts, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        b = texts[i : i + batch_size]
        inp = bge_tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = bge_mod(**inp)
            cls_embs = out.last_hidden_state[:, 0]
            norm_embs = F.normalize(cls_embs, p=2, dim=1)
            all_embs.append(norm_embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, 1024), dtype=np.float32)

# Deduplicate embeddings for speed
unique_prems = eval_df["premise"].unique().tolist()
unique_hyps = eval_df["hypothesis"].unique().tolist()
print(f"Embedding {len(unique_prems)} unique premises and {len(unique_hyps)} unique hypotheses...")

prem_embs = embed_texts(unique_prems, batch_size=32 if device == "cpu" else 64)
hyp_embs = embed_texts(unique_hyps, batch_size=32 if device == "cpu" else 64)

prem_map = dict(zip(unique_prems, prem_embs))
hyp_map = dict(zip(unique_hyps, hyp_embs))

# Compute cosine similarity for each pair
cos_sims = []
for _, r in eval_df.iterrows():
    ep = prem_map[r["premise"]]
    eh = hyp_map[r["hypothesis"]]
    sim = float(np.dot(ep, eh))
    cos_sims.append(round(sim, 4))

eval_df["stage1_cosine"] = cos_sims
eval_df["stage1_pass_tau075"] = eval_df["stage1_cosine"] >= 0.75
eval_df["stage1_pass_tau070"] = eval_df["stage1_cosine"] >= 0.70
print("[Stage 1] Dense cosine retrieval complete.")

# ---------------------------------------------------------------------------
# 3. Stage 2: V5 DeBERTa Ensemble Verifier
# ---------------------------------------------------------------------------
print("-" * 80)
print("[Stage 2] Loading V5 Verifier...")
v5_base_dir = "models/ensemble_data_only"
seed_dirs = [
    os.path.join(v5_base_dir, f"seed_{s}")
    for s in [42, 123, 999]
    if os.path.exists(os.path.join(v5_base_dir, f"seed_{s}")) and (
        os.path.exists(os.path.join(v5_base_dir, f"seed_{s}", "model.safetensors")) or
        os.path.exists(os.path.join(v5_base_dir, f"seed_{s}", "pytorch_model.bin"))
    )
]

if not seed_dirs:
    cand_fallbacks = [
        "models/argument_nli_deberta_v3_v3",
        "models/ensemble_data_only",
        "cross-encoder/nli-deberta-v3-base"
    ]
    fb = next((d for d in cand_fallbacks if os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin")) or d.startswith("cross-encoder/")), cand_fallbacks[0])
    seed_dirs = [fb]

print(f"Loaded {len(seed_dirs)} checkpoint(s): {seed_dirs}")
v5_tok = AutoTokenizer.from_pretrained(seed_dirs[0])
v5_models = [AutoModelForSequenceClassification.from_pretrained(sd).to(device).eval() for sd in seed_dirs]

def predict_stage2_ensemble(premises, hypotheses, batch_size=32 if device == "cpu" else 64, temperature=1.9104):
    all_probs = []
    for i in range(0, len(premises), batch_size):
        b_p = premises[i : i + batch_size]
        b_h = hypotheses[i : i + batch_size]
        enc = v5_tok(b_p, b_h, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            per_model_probs = []
            for m in v5_models:
                logits = m(**enc).logits / max(temperature, 1e-4)
                probs = F.softmax(logits, dim=-1)
                per_model_probs.append(probs)
            avg_probs = torch.stack(per_model_probs, dim=0).mean(dim=0)
            entail_idx = 1 if avg_probs.shape[-1] >= 2 else 0
            all_probs.extend(avg_probs[:, entail_idx].cpu().numpy().tolist())
    return all_probs

print("Scoring pairs with V5 Verifier ensemble...")
stage2_probs = predict_stage2_ensemble(eval_df["premise"].tolist(), eval_df["hypothesis"].tolist())
eval_df["p_entail"] = [round(float(p), 4) for p in stage2_probs]
eval_df["stage2_pass_f1"] = eval_df["p_entail"] >= 0.52
eval_df["stage2_pass_prec90"] = eval_df["p_entail"] >= 0.83

# ---------------------------------------------------------------------------
# 4. Two-Stage Cascade Decisions
# ---------------------------------------------------------------------------
# Two-Stage Pass: Stage 1 Pass AND Stage 2 Pass
eval_df["twostage_pass_f1"] = eval_df["stage1_pass_tau075"] & eval_df["stage2_pass_f1"]
eval_df["twostage_pass_prec90"] = eval_df["stage1_pass_tau075"] & eval_df["stage2_pass_prec90"]
eval_df["twostage_pass_tau070_f1"] = eval_df["stage1_pass_tau070"] & eval_df["stage2_pass_f1"]

os.makedirs("results", exist_ok=True)
eval_df.to_csv("results/twostage_matched_benchmark_predictions.csv", index=False)
print("Saved predictions to results/twostage_matched_benchmark_predictions.csv")

# ---------------------------------------------------------------------------
# 5. Master Publication Audit Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 125)
print("             TWO-STAGE PERSPECTIVE COVERAGE AUDIT TABLE (154 ARGS x 4 TYPES x 2 LENGTHS)")
print("                 Stage 1: BGE Cosine (tau*=0.75)  |  Stage 2: V5 Ensemble (theta*=0.52 / 0.83)")
print("=" * 125)
hdr = f"{'Length Regime':<15} | {'Perspective Type':<25} | {'Pairs':<5} | {'S1 Cos':<6} | {'S1 Pass':<8} | {'S2 P':<6} | {'S2 Alone':<9} | {'2-Stage F1':<10} | {'2-Stage Rej':<11} | {'2-Stage P90':<11}"
print(hdr)
print("-" * 125)

summary_rows = []
for length_regime in ["Full (180w)", "Concise (58w)"]:
    for ptype, _ in types:
        sub = eval_df[(eval_df["length_regime"] == length_regime) & (eval_df["perspective_type"] == ptype)]
        n = len(sub)
        s1_cos = sub["stage1_cosine"].mean()
        s1_pass = sub["stage1_pass_tau075"].mean() * 100
        s2_p = sub["p_entail"].mean()
        s2_alone = sub["stage2_pass_f1"].mean() * 100
        two_stage_f1 = sub["twostage_pass_f1"].mean() * 100
        two_stage_rej = 100.0 - two_stage_f1
        two_stage_p90 = sub["twostage_pass_prec90"].mean() * 100
        
        row_str = f"{length_regime:<15} | {ptype:<25} | {n:<5} | {s1_cos:.4f} | {s1_pass:>7.2f}% | {s2_p:.4f} | {s2_alone:>8.2f}% | {two_stage_f1:>9.2f}% | {two_stage_rej:>10.2f}% | {two_stage_p90:>10.2f}%"
        print(row_str)
        summary_rows.append({
            "Length Regime": length_regime,
            "Perspective Type": ptype,
            "Pairs": n,
            "Stage 1 Mean Cosine": s1_cos,
            "Stage 1 Pass Rate (tau=0.75)": s1_pass,
            "Stage 2 Mean Prob": s2_p,
            "Stage 2 Alone Accept (theta=0.52)": s2_alone,
            "Two-Stage Accept (tau=0.75, theta=0.52)": two_stage_f1,
            "Two-Stage Rejection Rate": two_stage_rej,
            "Two-Stage Accept (tau=0.75, theta=0.83)": two_stage_p90
        })

print("=" * 125)

# Save summary
pd.DataFrame(summary_rows).to_csv("results/twostage_matched_benchmark_summary.csv", index=False)

# Compute precision comparisons
for regime in ["Full (180w)", "Concise (58w)"]:
    sub = eval_df[eval_df["length_regime"] == regime]
    tp_s2 = sub[sub["perspective_type"] == "True Positive"]["stage2_pass_f1"].sum()
    all_s2 = sub["stage2_pass_f1"].sum()
    prec_s2 = tp_s2 / max(all_s2, 1) * 100

    tp_2st = sub[sub["perspective_type"] == "True Positive"]["twostage_pass_f1"].sum()
    all_2st = sub["twostage_pass_f1"].sum()
    prec_2st = tp_2st / max(all_2st, 1) * 100

    print(f"\n[{regime}] System Precision Comparison:")
    print(f"  - Stage 2 Alone (Standalone Verifier): Precision = {prec_s2:.2f}% ({tp_s2}/{all_s2} accepted are True Positives)")
    print(f"  - Two-Stage System (BGE + V5 Verifier): Precision = {prec_2st:.2f}% ({tp_2st}/{all_2st} accepted are True Positives)")
print("=" * 125 + "\n")
