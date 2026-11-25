"""
Claim-Split Calibration & Out-of-Sample Validation for Point-Sized Arguments.

1. Calibrates Stage 1 Threshold tau on Point-Sized Concise Arguments (~40-58 words)
   using the Stance-Verified Benchmark (154 quadruplets across 32 claims).
2. Implements Rigorous Out-of-Sample Claim Splitting:
   - 70% Calibration Claims (22 claims) -> TUNE tau* here
   - 30% Evaluation Claims  (10 claims) -> VALIDATE held-out performance
   - Also runs 5-Fold Claim Cross-Validation across all 32 claims.
3. Evaluates Two Operating Objectives:
   - Objective 1: Method A (85% Recall Target on True Positives)
   - Objective 2: High-F1 Optimization (Maximizing Strict F1 on Calibration)
4. Reports Held-Out Validation:
   - Held-Out Recall
   - Rebuttal Rejection Rate
   - Strict & Stance-Consistent Precision
   - Held-Out F1-Score
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 115)
print("CLAIM-SPLIT CALIBRATION & HELD-OUT VALIDATION (POINT-SIZED ARGUMENTS)")
print(f"Device: {device}")
print("=" * 115)

# 1. Load Stance-Verified Benchmark
quad_path = "data/audit/stance_verified_matched_quadruplets.csv"
if not os.path.exists(quad_path):
    quad_path = "data/audit/matched_quadruplets_167.csv"

df_quads = pd.read_csv(quad_path)
print(f"Loaded {len(df_quads)} quadruplets across {df_quads['claim_id'].nunique()} unique claims from: {quad_path}")

types = [
    ("True Positive", "hyp_true_positive"),
    ("Opposite-Stance", "hyp_opposite_stance"),
    ("Same-Stance Diff-Reason", "hyp_same_stance_diff_reason"),
    ("Unrelated", "hyp_unrelated")
]

records = []
for ptype, col_hyp in types:
    for idx, r in df_quads.iterrows():
        records.append({
            "quad_id": int(r["quad_id"]),
            "claim_id": int(r["claim_id"]),
            "claim_text": str(r.get("claim_text", "")).strip(),
            "perspective_type": ptype,
            "concise_premise_58w": str(r["concise_premise_58w"]).strip(),
            "hypothesis": str(r[col_hyp]).strip(),
            "premise_stance": str(r.get("premise_stance", "support")).strip().lower(),
            "hyp_stance": str(r.get("tp_stance" if ptype=="True Positive" else ("opp_stance" if ptype=="Opposite-Stance" else "diff_stance"), "support")).strip().lower()
        })

eval_df = pd.DataFrame(records)
print(f"Constructed {len(eval_df)} point-sized evaluation pairs (4 per quadruplet).")

# 2. Compute or Load BGE Embeddings
pred_cache = "results/claim_split_calibration_predictions.csv"
if os.path.exists(pred_cache):
    print(f"Loading cached similarities from {pred_cache}...")
    cached_df = pd.read_csv(pred_cache)
    eval_df["stage1_cosine"] = cached_df["stage1_cosine"]
else:
    from transformers import AutoTokenizer, AutoModel
    bge_name = "BAAI/bge-large-en-v1.5"
    print(f"Loading {bge_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(bge_name)
    model = AutoModel.from_pretrained(bge_name).to(device).eval()

    def embed_texts(texts, batch_size=64 if device=="cuda" else 32):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            b = texts[i : i + batch_size]
            enc = tokenizer(b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
                cls_e = F.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
                all_embs.append(cls_e.cpu().numpy())
        return np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, 1024), dtype=np.float32)

    unique_prems = eval_df["concise_premise_58w"].unique().tolist()
    unique_hyps = eval_df["hypothesis"].unique().tolist()
    print(f"Embedding {len(unique_prems)} unique premises and {len(unique_hyps)} unique hypotheses...")
    
    prem_map = dict(zip(unique_prems, embed_texts(unique_prems)))
    hyp_map = dict(zip(unique_hyps, embed_texts(unique_hyps)))

    cos_sims = []
    for _, r in eval_df.iterrows():
        ep = prem_map[r["concise_premise_58w"]]
        eh = hyp_map[r["hypothesis"]]
        cos_sims.append(round(float(np.dot(ep, eh)), 4))
    eval_df["stage1_cosine"] = cos_sims
    os.makedirs("results", exist_ok=True)
    eval_df.to_csv(pred_cache, index=False)
    print(f"Saved predictions to: {pred_cache}")

# ---------------------------------------------------------------------------
# 3. Helper Functions for Tuning and Validation
# ---------------------------------------------------------------------------
def compute_metrics_at_tau(df_sub, tau):
    tp_sub = df_sub[df_sub["perspective_type"] == "True Positive"]
    opp_sub = df_sub[df_sub["perspective_type"] == "Opposite-Stance"]
    ssdr_sub = df_sub[df_sub["perspective_type"] == "Same-Stance Diff-Reason"]
    unrel_sub = df_sub[df_sub["perspective_type"] == "Unrelated"]
    
    tp_acc = int((tp_sub["stage1_cosine"] >= tau).sum())
    opp_acc = int((opp_sub["stage1_cosine"] >= tau).sum())
    ssdr_acc = int((ssdr_sub["stage1_cosine"] >= tau).sum())
    unrel_acc = int((unrel_sub["stage1_cosine"] >= tau).sum())
    total_acc = tp_acc + opp_acc + ssdr_acc + unrel_acc
    
    n_tp = len(tp_sub)
    n_opp = len(opp_sub)
    
    recall = (tp_acc / max(n_tp, 1)) * 100
    strict_prec = (tp_acc / max(total_acc, 1)) * 100
    stance_prec = ((tp_acc + ssdr_acc) / max(total_acc, 1)) * 100
    f1 = (2 * strict_prec * recall) / max(strict_prec + recall, 1e-4)
    opp_rej = ((n_opp - opp_acc) / max(n_opp, 1)) * 100
    
    return {
        "tau": round(tau, 4),
        "recall": recall,
        "strict_prec": strict_prec,
        "stance_prec": stance_prec,
        "f1": f1,
        "opp_rej": opp_rej,
        "tp_acc": tp_acc,
        "total_acc": total_acc,
        "opp_acc": opp_acc
    }

def find_optimal_f1_tau(df_train):
    best_f1 = -1.0
    best_tau = 0.60
    for tau in np.arange(0.50, 0.72, 0.005):
        m = compute_metrics_at_tau(df_train, tau)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_tau = tau
    return round(float(best_tau), 4)

# ---------------------------------------------------------------------------
# 4. Single 70/30 Claim-Disjoint Split Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 115)
print("1. SINGLE 70/30 CLAIM-DISJOINT SPLIT EVALUATION")
print("=" * 115)

unique_claims = np.array(sorted(eval_df["claim_id"].unique()))
np.random.seed(42)
shuffled_claims = np.random.permutation(unique_claims)

n_train_claims = int(0.70 * len(unique_claims))
calib_claims = set(shuffled_claims[:n_train_claims])
eval_claims  = set(shuffled_claims[n_train_claims:])

df_calib = eval_df[eval_df["claim_id"].isin(calib_claims)]
df_eval  = eval_df[eval_df["claim_id"].isin(eval_claims)]

print(f"Calibration Claims: {len(calib_claims)} claims ({len(df_calib)//4} quadruplets, {len(df_calib)} pairs)")
print(f"Validation Claims:  {len(eval_claims)} claims ({len(df_eval)//4} quadruplets, {len(df_eval)} pairs)")

# Tune Method A (85% Recall Target on Calibration)
tp_calib = df_calib[df_calib["perspective_type"] == "True Positive"]["stage1_cosine"].values
tau_85_tuned = float(np.percentile(tp_calib, 15.0))

# Tune Max-F1 on Calibration
tau_f1_tuned = find_optimal_f1_tau(df_calib)

print(f"\nTuned Thresholds on 70% Calibration Claims:")
print(f"  * Method A (85% Target Recall): tau* = {tau_85_tuned:.4f}")
print(f"  * High-F1 Objective:            tau* = {tau_f1_tuned:.4f}")

# Evaluate In-Sample vs. Out-of-Sample Held-Out
print("\n" + "-" * 115)
print(f"{'Objective':<22} | {'Tuned tau*':<10} | {'Split':<12} | {'TP Recall':<10} | {'Strict Prec':<12} | {'Stance Prec':<12} | {'Rebut Rej':<10} | {'Strict F1':<10}")
print("-" * 115)

for label, tau_val in [("Method A (85% Recall)", tau_85_tuned), ("High-F1 Objective", tau_f1_tuned)]:
    m_in = compute_metrics_at_tau(df_calib, tau_val)
    m_out = compute_metrics_at_tau(df_eval, tau_val)
    
    print(f"{label:<22} | {tau_val:<10.4f} | {'In-Sample':<12} | {m_in['recall']:>8.2f}% | {m_in['strict_prec']:>10.2f}% | {m_in['stance_prec']:>10.2f}% | {m_in['opp_rej']:>8.2f}% | {m_in['f1']:>8.2f}%")
    print(f"{'':<22} | {'':<10} | {'HELD-OUT':<12} | {m_out['recall']:>8.2f}% | {m_out['strict_prec']:>10.2f}% | {m_out['stance_prec']:>10.2f}% | {m_out['opp_rej']:>8.2f}% | {m_out['f1']:>8.2f}%")
    print("-" * 115)

# ---------------------------------------------------------------------------
# 5. 5-Fold Claim-Split Cross-Validation (Full Generalization Audit)
# ---------------------------------------------------------------------------
print("\n" + "=" * 115)
print("2. 5-FOLD CLAIM-SPLIT CROSS-VALIDATION (UNBIASED GENERALIZATION ACROSS ALL 32 CLAIMS)")
print("=" * 115)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results_85 = []
cv_results_f1 = []

fold_idx = 1
for train_c_idx, val_c_idx in kf.split(unique_claims):
    c_train = set(unique_claims[train_c_idx])
    c_val   = set(unique_claims[val_c_idx])
    
    df_f_train = eval_df[eval_df["claim_id"].isin(c_train)]
    df_f_val   = eval_df[eval_df["claim_id"].isin(c_val)]
    
    # 1. Method A (85% Recall)
    tp_f_train = df_f_train[df_f_train["perspective_type"] == "True Positive"]["stage1_cosine"].values
    t_85 = float(np.percentile(tp_f_train, 15.0))
    val_m_85 = compute_metrics_at_tau(df_f_val, t_85)
    cv_results_85.append(val_m_85)
    
    # 2. High-F1
    t_f1 = find_optimal_f1_tau(df_f_train)
    val_m_f1 = compute_metrics_at_tau(df_f_val, t_f1)
    cv_results_f1.append(val_m_f1)
    
    print(f"Fold {fold_idx}: Train {len(c_train)} claims, Val {len(c_val)} claims | Method A tau*={t_85:.4f} (Val Rec={val_m_85['recall']:.1f}%, F1={val_m_85['f1']:.1f}%) | Max-F1 tau*={t_f1:.4f} (Val F1={val_m_f1['f1']:.1f}%)")
    fold_idx += 1

print("-" * 115)
# Compute Mean & Std across folds
def print_cv_summary(label, res_list):
    df_res = pd.DataFrame(res_list)
    print(f"5-Fold CV Held-Out Summary [{label}]:")
    print(f"  * Calibrated tau*:      {df_res['tau'].mean():.4f} +/- {df_res['tau'].std():.4f}")
    print(f"  * Held-Out Recall:      {df_res['recall'].mean():.2f}% +/- {df_res['recall'].std():.2f}%")
    print(f"  * Strict Precision:     {df_res['strict_prec'].mean():.2f}% +/- {df_res['strict_prec'].std():.2f}%")
    print(f"  * Stance-Consistent:    {df_res['stance_prec'].mean():.2f}% +/- {df_res['stance_prec'].std():.2f}%")
    print(f"  * Rebuttal Rejection:   {df_res['opp_rej'].mean():.2f}% +/- {df_res['opp_rej'].std():.2f}%")
    print(f"  * Held-Out Strict F1:   {df_res['f1'].mean():.2f}% +/- {df_res['f1'].std():.2f}%")

print_cv_summary("Method A (85% Target Recall)", cv_results_85)
print("")
print_cv_summary("High-F1 Objective", cv_results_f1)

print("\n" + "=" * 115)
print("CALIBRATION & VALIDATION AUDIT COMPLETE.")
print("=" * 115)
