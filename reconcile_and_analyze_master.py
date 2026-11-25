import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

master_csv = "results/metrics/master_1118_predictions.csv"
if not os.path.exists(master_csv):
    print(f"Waiting for {master_csv}...")
    sys.exit(0)

df = pd.read_csv(master_csv)
print(f"Loaded {len(df)} rows from {master_csv}")

# Map labels to integer: contradiction: 0, entailment: 1, neutral: 2
label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
y_true_3class = df["label"].map(label_map).values
y_true_binary = (df["label"] == "entailment").astype(int).values

models = [
    ("v3", "v3_p_entail", "v3_p_contra", "v3_p_neutral", "Version 3 (Coverage-Aware)"),
    ("base_fwd", "base_fwd_p_entail", "base_fwd_p_contra", "base_fwd_p_neutral", "Forward Zero-Shot Baseline"),
    ("base_inv", "base_inv_p_entail", "base_inv_p_contra", "base_inv_p_neutral", "Legacy Inverted Baseline")
]

# 1. Evaluate 3-Class Argmax Classification
print("\n" + "="*80)
print("1. THREE-CLASS ARGMAX CLASSIFICATION PERFORMANCE (N=1,118)")
print("="*80)
argmax_results = []
for m_key, p_ent, p_con, p_neu, m_name in models:
    # argmax index: 0=contra, 1=entail, 2=neutral
    probs_3class = np.column_stack([df[p_con], df[p_ent], df[p_neu]])
    preds_3class = np.argmax(probs_3class, axis=1)
    
    acc = accuracy_score(y_true_3class, preds_3class)
    macro_f1 = f1_score(y_true_3class, preds_3class, average="macro", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true_3class, preds_3class, labels=[0, 1, 2], zero_division=0)
    
    # Argmax confusion matrix: rows=true (contra, entail, neutral), cols=pred (contra, entail, neutral)
    cm = confusion_matrix(y_true_3class, preds_3class, labels=[0, 1, 2])
    # Entailment False Positives under argmax: true is contra (0) or neutral (2), but pred is entail (1)
    fp_argmax = cm[0, 1] + cm[2, 1]
    tp_argmax = cm[1, 1]
    fn_argmax = cm[1, 0] + cm[1, 2]
    tn_argmax = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]
    
    argmax_results.append({
        "Model": m_name,
        "Accuracy": round(float(acc) * 100, 2),
        "Macro_F1": round(float(macro_f1), 4),
        "Entailment_F1": round(float(f1_cls[1]), 4),
        "Contra_F1": round(float(f1_cls[0]), 4),
        "Neutral_F1": round(float(f1_cls[2]), 4),
        "TP_argmax": int(tp_argmax),
        "FP_argmax": int(fp_argmax),
        "FN_argmax": int(fn_argmax),
        "TN_argmax": int(tn_argmax)
    })
df_argmax = pd.DataFrame(argmax_results)
print(df_argmax.to_string(index=False))

# 2. Evaluate Binary Operating Points (threshold on p_entail)
print("\n" + "="*80)
print("2. BINARY VERIFICATION AT COMMON THRESHOLDS (N=1,118; Pos=540, Neg=578)")
print("="*80)
binary_results = []
for theta in [0.50, 0.72, 0.80, 0.90]:
    for m_key, p_ent, p_con, p_neu, m_name in models:
        pred_bin = (df[p_ent] >= theta).astype(int).values
        tp = int(np.sum((pred_bin == 1) & (y_true_binary == 1)))
        fp = int(np.sum((pred_bin == 1) & (y_true_binary == 0)))
        fn = int(np.sum((pred_bin == 0) & (y_true_binary == 1)))
        tn = int(np.sum((pred_bin == 0) & (y_true_binary == 0)))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        binary_results.append({
            "Threshold": theta,
            "Model": m_name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1": round(f1, 4)
        })
df_bin = pd.DataFrame(binary_results)
print(df_bin.to_string(index=False))

# 3. Reconciliation of Positive Subsets: Human (352) + LLM (188) = 540
print("\n" + "="*80)
print("3. POSITIVE SUBSET RECONCILIATION: Human-Human (352) vs LLM-to-Human (188)")
print("="*80)
pos_recon = []
human_mask = (df["pair_category"] == "within_cluster_paraphrase").values
llm_mask = df["pair_category"].str.startswith("long_llm_to_human_positive").values

for theta in [0.50, 0.72, 0.80, 0.90]:
    for m_key, p_ent, p_con, p_neu, m_name in models:
        tp_human = int((df.loc[human_mask, p_ent] >= theta).sum())
        tp_llm = int((df.loc[llm_mask, p_ent] >= theta).sum())
        tp_total = tp_human + tp_llm
        
        pos_recon.append({
            "Threshold": theta,
            "Model": m_name,
            "Human_Paraphrase_TP": f"{tp_human}/352 ({tp_human/352*100:.1f}%)",
            "LLM_to_Human_TP": f"{tp_llm}/188 ({tp_llm/188*100:.1f}%)",
            "Total_TP": f"{tp_total}/540 ({tp_total/540*100:.1f}%)"
        })
df_pos_recon = pd.DataFrame(pos_recon)
print(df_pos_recon.to_string(index=False))

# 4. Reconciliation of Negative Subsets (578 total)
print("\n" + "="*80)
print("4. NEGATIVE SUBSET RECONCILIATION (578 total: 240 + 180 + 120 + 38)")
print("="*80)
neg_recon = []
c_same_stance = (df["pair_category"] == "same_stance_different_reasoning")
c_same_topic = (df["pair_category"] == "same_topic_diff_concept")
c_opp_stance = (df["pair_category"] == "opposite_stance_conflict")
c_unrelated = (df["pair_category"] == "unrelated_different_claim")

for theta in [0.50, 0.72, 0.80, 0.90]:
    for m_key, p_ent, p_con, p_neu, m_name in models:
        fp_ss = int((df.loc[c_same_stance, p_ent] >= theta).sum())
        fp_st = int((df.loc[c_same_topic, p_ent] >= theta).sum())
        fp_opp = int((df.loc[c_opp_stance, p_ent] >= theta).sum())
        fp_unrel = int((df.loc[c_unrelated, p_ent] >= theta).sum())
        fp_total = fp_ss + fp_st + fp_opp + fp_unrel
        
        neg_recon.append({
            "Threshold": theta,
            "Model": m_name,
            "FP_Same_Stance (240)": f"{fp_ss} ({fp_ss/240*100:.1f}%)",
            "FP_Same_Topic (180)": f"{fp_st} ({fp_st/180*100:.1f}%)",
            "FP_Opposite (120)": f"{fp_opp} ({fp_opp/120*100:.1f}%)",
            "FP_Unrelated (38)": f"{fp_unrel} ({fp_unrel/38*100:.1f}%)",
            "Total_FP (578)": f"{fp_total} ({fp_total/578*100:.1f}%)",
            "Rejection_Rate": f"{(578-fp_total)}/578 ({(578-fp_total)/578*100:.1f}%)"
        })
df_neg_recon = pd.DataFrame(neg_recon)
print(df_neg_recon.to_string(index=False))

# 5. Exact Paired McNemar Test on 188 LLM Positives: Forward Baseline vs V3
print("\n" + "="*80)
print("5. EXACT PAIRED MCNEMAR TEST ON 188 LLM POSITIVES: FORWARD BASELINE vs V3")
print("="*80)
llm_sub = df[llm_mask].copy()
mcnemar_fwd_v3 = []
for theta in [0.50, 0.72, 0.80, 0.90]:
    v3_det = (llm_sub["v3_p_entail"] >= theta).astype(int).values
    fwd_det = (llm_sub["base_fwd_p_entail"] >= theta).astype(int).values
    inv_det = (llm_sub["base_inv_p_entail"] >= theta).astype(int).values
    
    # Forward vs V3
    b_fwd = int(np.sum((v3_det == 1) & (fwd_det == 0)))  # V3 detects, Forward misses
    c_fwd = int(np.sum((v3_det == 0) & (fwd_det == 1)))  # Forward detects, V3 misses
    p_val_fwd = float(stats.binom.cdf(min(b_fwd, c_fwd), b_fwd + c_fwd, 0.5) * 2.0) if (b_fwd + c_fwd) > 0 else 1.0
    
    # Inverted vs V3
    b_inv = int(np.sum((v3_det == 1) & (inv_det == 0)))  # V3 detects, Inverted misses
    c_inv = int(np.sum((v3_det == 0) & (inv_det == 1)))  # Inverted detects, V3 misses
    p_val_inv = float(stats.binom.cdf(min(b_inv, c_inv), b_inv + c_inv, 0.5) * 2.0) if (b_inv + c_inv) > 0 else 1.0
    
    mcnemar_fwd_v3.append({
        "Threshold": theta,
        "Forward_Baseline_TP": int(np.sum(fwd_det)),
        "V3_TP": int(np.sum(v3_det)),
        "Delta_TP": int(np.sum(v3_det)) - int(np.sum(fwd_det)),
        "Discordant_b (V3 only)": b_fwd,
        "Discordant_c (Fwd only)": c_fwd,
        "McNemar_p_val (Fwd vs V3)": f"{p_val_fwd:.6e}",
        "Inverted_Baseline_TP": int(np.sum(inv_det)),
        "McNemar_p_val (Inv vs V3)": f"{p_val_inv:.6e}"
    })
df_mcnemar = pd.DataFrame(mcnemar_fwd_v3)
print(df_mcnemar.to_string(index=False))

# 6. Length Tier Breakdown on 188 LLM Positives for BOTH Forward & Inverted Baselines
print("\n" + "="*80)
print("6. LENGTH TIER BREAKDOWN ON 188 LLM POSITIVES: FORWARD BASELINE vs V3")
print("="*80)
length_results = []
for theta in [0.50, 0.80]:
    for tier in ["short", "medium", "long", "very_long", "ALL"]:
        sub = llm_sub if tier == "ALL" else llm_sub[llm_sub["length_tier"] == tier]
        n_t = len(sub)
        tp_v3 = int((sub["v3_p_entail"] >= theta).sum())
        tp_fwd = int((sub["base_fwd_p_entail"] >= theta).sum())
        tp_inv = int((sub["base_inv_p_entail"] >= theta).sum())
        
        # McNemar Fwd vs V3
        b = int(((sub["v3_p_entail"] >= theta) & (sub["base_fwd_p_entail"] < theta)).sum())
        c = int(((sub["v3_p_entail"] < theta) & (sub["base_fwd_p_entail"] >= theta)).sum())
        p_val = float(stats.binom.cdf(min(b, c), b + c, 0.5) * 2.0) if (b + c) > 0 else 1.0
        
        length_results.append({
            "Threshold": theta,
            "Length_Tier": tier,
            "Total_Pairs": n_t,
            "Forward_Baseline_TP": tp_fwd,
            "Forward_Recall": f"{tp_fwd/n_t*100:.1f}%",
            "Inverted_Baseline_TP": tp_inv,
            "Inverted_Recall": f"{tp_inv/n_t*100:.1f}%",
            "V3_TP": tp_v3,
            "V3_Recall": f"{tp_v3/n_t*100:.1f}%",
            "Delta_Recall (V3 vs Fwd)": f"{(tp_v3 - tp_fwd)/n_t*100:+.1f} pp",
            "McNemar_p_value": f"{p_val:.4e}"
        })
df_length = pd.DataFrame(length_results)
print(df_length.to_string(index=False))

# Save all unified publication tables
df_argmax.to_csv("results/paper_tables/table_unified_3class_argmax.csv", index=False)
df_bin.to_csv("results/paper_tables/table_unified_binary_operating_points.csv", index=False)
df_pos_recon.to_csv("results/paper_tables/table_unified_positive_reconciliation.csv", index=False)
df_neg_recon.to_csv("results/paper_tables/table_unified_negative_reconciliation.csv", index=False)
df_mcnemar.to_csv("results/paper_tables/table_unified_mcnemar_forward_vs_v3.csv", index=False)
df_length.to_csv("results/paper_tables/table_unified_length_breakdown_forward_vs_v3.csv", index=False)
print("\nSaved all unified tables to results/paper_tables/")

