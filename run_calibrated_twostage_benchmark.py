"""
Calibrated Two-Stage Perspective Coverage Benchmark Runner.

Calibrates Stage 1 (BGE-Large Dense Retrieval) at the user-specified 85% and 90%
Recall operating thresholds, and evaluates both length regimes:
  - Full (180w)
  - Concise (58w)

Across all 4 perspective types:
  - True Positive (Ground truth match)
  - Opposite-Stance (Rebuttal trap)
  - Same-Stance Diff-Reason (Alternative cluster trap)
  - Unrelated (Off-topic negative control)

Outputs complete audit tables comparing:
  - Baseline Strict Threshold (tau* = 0.75)
  - Calibrated 85% Recall Threshold (tau* = tau_85)
  - Calibrated 90% Recall Threshold (tau* = tau_90)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 115)
print("CALIBRATED TWO-STAGE BENCHMARK: 85% & 90% RECALL THRESHOLD EVALUATION")
print(f"Device: {device}")
print("=" * 115)

# ---------------------------------------------------------------------------
# 1. Load or Generate Predictions
# ---------------------------------------------------------------------------
pred_path = "results/twostage_matched_benchmark_predictions.csv"

if os.path.exists(pred_path):
    print(f"Loading existing predictions from: {pred_path}")
    eval_df = pd.read_csv(pred_path)
else:
    print(f"{pred_path} not found. Running inference to generate predictions...")
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

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

    # Stage 1: BGE Embeddings
    bge_model_name = "BAAI/bge-large-en-v1.5"
    print(f"[Stage 1] Loading BGE Embedder: {bge_model_name}...")
    bge_tok = AutoTokenizer.from_pretrained(bge_model_name)
    bge_mod = AutoModel.from_pretrained(bge_model_name).to(device).eval()

    def embed_texts(texts, batch_size=32 if device == "cpu" else 64):
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

    unique_prems = eval_df["premise"].unique().tolist()
    unique_hyps = eval_df["hypothesis"].unique().tolist()
    prem_map = dict(zip(unique_prems, embed_texts(unique_prems)))
    hyp_map = dict(zip(unique_hyps, embed_texts(unique_hyps)))

    cos_sims = []
    for _, r in eval_df.iterrows():
        ep = prem_map[r["premise"]]
        eh = hyp_map[r["hypothesis"]]
        cos_sims.append(round(float(np.dot(ep, eh)), 4))
    eval_df["stage1_cosine"] = cos_sims

    # Stage 2: V5 Ensemble
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
        cand_fallbacks = ["models/argument_nli_deberta_v3_v3", "cross-encoder/nli-deberta-v3-base"]
        seed_dirs = [next((d for d in cand_fallbacks if os.path.exists(d) or d.startswith("cross-encoder/")), cand_fallbacks[0])]

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

    eval_df["p_entail"] = [round(float(p), 4) for p in predict_stage2_ensemble(eval_df["premise"].tolist(), eval_df["hypothesis"].tolist())]
    eval_df["stage1_pass_tau075"] = eval_df["stage1_cosine"] >= 0.75
    eval_df["stage2_pass_f1"] = eval_df["p_entail"] >= 0.52
    eval_df["stage2_pass_prec90"] = eval_df["p_entail"] >= 0.83
    eval_df["twostage_pass_f1"] = eval_df["stage1_pass_tau075"] & eval_df["stage2_pass_f1"]
    eval_df["twostage_pass_prec90"] = eval_df["stage1_pass_tau075"] & eval_df["stage2_pass_prec90"]

    os.makedirs("results", exist_ok=True)
    eval_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

print(f"Dataset ready: {len(eval_df)} total evaluated pairs.")

# ---------------------------------------------------------------------------
# 2. Derive 85% and 90% Recall Thresholds on True Positives
# ---------------------------------------------------------------------------
tp_full = eval_df[(eval_df["length_regime"] == "Full (180w)") & (eval_df["perspective_type"] == "True Positive")]["stage1_cosine"].values
tp_concise = eval_df[(eval_df["length_regime"] == "Concise (58w)") & (eval_df["perspective_type"] == "True Positive")]["stage1_cosine"].values
tp_all = eval_df[eval_df["perspective_type"] == "True Positive"]["stage1_cosine"].values

# Calibrate: 85% recall means threshold at 15th percentile; 90% recall at 10th percentile
tau85_full = float(np.percentile(tp_full, 15.0))
tau90_full = float(np.percentile(tp_full, 10.0))

tau85_concise = float(np.percentile(tp_concise, 15.0))
tau90_concise = float(np.percentile(tp_concise, 10.0))

tau85_global = float(np.percentile(tp_all, 15.0))
tau90_global = float(np.percentile(tp_all, 10.0))

print("\n" + "=" * 115)
print("STAGE 1 CALIBRATED THRESHOLDS (TARGET RECALL CONSTRAINTS)")
print("=" * 115)
print(f"Full (180w) Regime:    tau*(85% Recall) = {tau85_full:.4f}  |  tau*(90% Recall) = {tau90_full:.4f}")
print(f"Concise (58w) Regime: tau*(85% Recall) = {tau85_concise:.4f}  |  tau*(90% Recall) = {tau90_concise:.4f}")
print(f"Unified Global:        tau*(85% Recall) = {tau85_global:.4f}  |  tau*(90% Recall) = {tau90_global:.4f}")
print("=" * 115)

# ---------------------------------------------------------------------------
# 3. Comprehensive Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_threshold(tau_val, threshold_label):
    print("\n" + "#" * 115)
    print(f" AUDIT FOR STAGE 1 THRESHOLD: {threshold_label} (tau = {tau_val:.4f})  |  Stage 2: theta* = 0.52 (F1) / 0.83 (Prec90)")
    print("#" * 115)
    hdr = f"{'Length Regime':<15} | {'Perspective Type':<25} | {'Pairs':<5} | {'S1 Cos':<6} | {'S1 Pass':<8} | {'S2 P':<6} | {'S2 Alone':<9} | {'2-Stage F1':<10} | {'2-Stage Rej':<11} | {'2-Stage P90':<11}"
    print(hdr)
    print("-" * 115)

    types_order = ["True Positive", "Opposite-Stance", "Same-Stance Diff-Reason", "Unrelated"]
    
    summary_data = []
    for regime in ["Full (180w)", "Concise (58w)"]:
        regime_sub = eval_df[eval_df["length_regime"] == regime]
        for ptype in types_order:
            sub = regime_sub[regime_sub["perspective_type"] == ptype]
            n = len(sub)
            s1_cos = sub["stage1_cosine"].mean()
            
            # S1 pass at current tau
            s1_pass_mask = sub["stage1_cosine"] >= tau_val
            s1_pass_pct = s1_pass_mask.mean() * 100
            
            s2_p = sub["p_entail"].mean()
            s2_alone = (sub["p_entail"] >= 0.52).mean() * 100
            
            # 2-Stage pass
            two_pass_f1 = (s1_pass_mask & (sub["p_entail"] >= 0.52))
            two_pass_p90 = (s1_pass_mask & (sub["p_entail"] >= 0.83))
            
            two_stage_f1_pct = two_pass_f1.mean() * 100
            two_stage_rej_pct = 100.0 - two_stage_f1_pct
            two_stage_p90_pct = two_pass_p90.mean() * 100
            
            print(f"{regime:<15} | {ptype:<25} | {n:<5} | {s1_cos:.4f} | {s1_pass_pct:>7.2f}% | {s2_p:.4f} | {s2_alone:>8.2f}% | {two_stage_f1_pct:>9.2f}% | {two_stage_rej_pct:>10.2f}% | {two_stage_p90_pct:>10.2f}%")
            
            summary_data.append({
                "regime": regime,
                "type": ptype,
                "n": n,
                "s1_pass_count": int(s1_pass_mask.sum()),
                "twostage_accept_f1": int(two_pass_f1.sum()),
                "twostage_accept_p90": int(two_pass_p90.sum())
            })
            
    print("-" * 115)
    # Summary Precision & Recall for each regime
    sum_df = pd.DataFrame(summary_data)
    for regime in ["Full (180w)", "Concise (58w)"]:
        r_df = sum_df[sum_df["regime"] == regime]
        tp_acc = r_df[r_df["type"] == "True Positive"]["twostage_accept_f1"].values[0]
        opp_acc = r_df[r_df["type"] == "Opposite-Stance"]["twostage_accept_f1"].values[0]
        ssdr_acc = r_df[r_df["type"] == "Same-Stance Diff-Reason"]["twostage_accept_f1"].values[0]
        unrel_acc = r_df[r_df["type"] == "Unrelated"]["twostage_accept_f1"].values[0]
        total_acc = tp_acc + opp_acc + ssdr_acc + unrel_acc
        
        prec = (tp_acc / max(total_acc, 1)) * 100
        rec = (tp_acc / 154) * 100
        f1 = (2 * prec * rec) / max(prec + rec, 1e-4)
        
        s1_tp_pass = r_df[r_df["type"] == "True Positive"]["s1_pass_count"].values[0]
        s1_rec = (s1_tp_pass / 154) * 100
        
        print(f"[{regime} System Metrics at tau={tau_val:.4f}]:")
        print(f"  * Stage 1 TP Recall:          {s1_rec:.2f}% ({s1_tp_pass}/154 True Positives passed S1)")
        print(f"  * Total Accepted by 2-Stage:   {total_acc} pairs (TP: {tp_acc}, Opposite: {opp_acc}, SSDR: {ssdr_acc}, Unrelated: {unrel_acc})")
        print(f"  * Two-Stage System Precision:  {prec:.2f}% ({tp_acc}/{total_acc} accepted are True Positives)")
        print(f"  * Two-Stage System Recall:     {rec:.2f}% ({tp_acc}/154 True Positives accepted)")
        print(f"  * Two-Stage System F1-Score:   {f1:.2f}%")
        print(f"  * False Rebuttal Leakage:      Opposite-Stance={opp_acc}/154 ({opp_acc/154*100:.2f}%), Unrelated={unrel_acc}/154 ({unrel_acc/154*100:.2f}%)")

# ---------------------------------------------------------------------------
# 4. Run Evaluations Across All Targeted Thresholds
# ---------------------------------------------------------------------------
# Baseline
evaluate_threshold(0.75, "Strict Baseline (Old)")

# 85% Recall Threshold (using regime-specific or global)
evaluate_threshold(tau85_global, f"Calibrated 85% Recall Target (Global tau* = {tau85_global:.4f})")

# 90% Recall Threshold (using regime-specific or global)
evaluate_threshold(tau90_global, f"Calibrated 90% Recall Target (Global tau* = {tau90_global:.4f})")

# Also run Concise-specific 85% and 90%
if not np.isclose(tau85_concise, tau85_global, atol=0.005):
    evaluate_threshold(tau85_concise, f"Calibrated 85% Recall Target (Concise-Specific tau* = {tau85_concise:.4f})")

if not np.isclose(tau90_concise, tau90_global, atol=0.005):
    evaluate_threshold(tau90_concise, f"Calibrated 90% Recall Target (Concise-Specific tau* = {tau90_concise:.4f})")

print("\n" + "=" * 115)
print("BENCHMARK AUDIT COMPLETED.")
print("=" * 115)
