"""
E3.4 & E3.5: Independent Validation & Sampling-Weighted Production Audit
Evaluates:
  1. E3.4: Claim-disjoint validation on non-test claims from unified 100 audit benchmark.
  2. E3.5: Full Horvitz-Thompson sampling-weighted production audit (N_pop=250, n_sample=100).
Compares frozen baseline M0 against experimental reasoning-aware M2.
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath("."))
from src.extraction.argument_component_extractor import ArgumentComponentExtractor, compute_stable_text_id, EXTRACTION_VERSION

def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def main():
    print("=" * 90)
    print("EXPERIMENTS E3.4 & E3.5: INDEPENDENT VALIDATION & PRODUCTION AUDIT")
    print("=" * 90)
    
    # 1. Load trained models on 341 Train partition
    b341 = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    train = b341[(b341['partition'] == 'train') & (b341['human_ground_truth_label'].isin(['class_2', 'class_1']))]
    y_tr = (train['human_ground_truth_label'] == 'class_2').astype(int).values
    
    feats_m0 = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc_m0 = StandardScaler()
    X_tr_m0 = sc_m0.fit_transform(train[feats_m0].values)
    clf_m0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m0, y_tr)
    
    feats_m2 = ['cosine_similarity', 'p_entail', 'p_neutral', 'reasoning_cosine_v1_1']
    sc_m2 = StandardScaler()
    X_tr_m2 = sc_m2.fit_transform(train[feats_m2].values)
    clf_m2 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m2, y_tr)
    
    print("Models loaded successfully.")
    print(f"M0 weights: {dict(zip(feats_m0, clf_m0.coef_[0].round(4)))}")
    print(f"M2 weights: {dict(zip(feats_m2, clf_m2.coef_[0].round(4)))}")
    
    # 2. Load unified 100-pair audit dataset
    u100_path = "data/audit/unified_production_audit_benchmark_100.csv"
    u100 = pd.read_csv(u100_path)
    print(f"\nLoaded {len(u100)} pairs from {u100_path}")
    
    # 3. Load extraction cache
    cache_path = "data/calibration/e3_1_extracted_components_cache.json"
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    npz_path = "data/calibration/e3_1_component_embeddings_cache.npz"
    npz = np.load(npz_path)
    text_ids_cached = list(npz['text_ids'])
    tid_to_idx = {tid: i for i, tid in enumerate(text_ids_cached)}
    claim_embs_cached = npz['claim_embeddings']
    reason_embs_cached = npz['reasoning_embeddings']
    warrant_embs_cached = npz['warrant_embeddings']
    
    extractor = ArgumentComponentExtractor(version=EXTRACTION_VERSION)
    
    # Check which texts need extraction and embedding
    texts_to_embed = {}
    for _, row in u100.iterrows():
        for col, src in [('gold_perspective_text', 'gold'), ('generated_viewpoint_text', 'generated')]:
            txt = row[col]
            tid = compute_stable_text_id(txt)
            if tid not in cache:
                rec = extractor.extract_single(txt, claim_topic=row['claim_text'])
                cache[tid] = rec
            if tid not in tid_to_idx:
                texts_to_embed[tid] = cache[tid]
                
    print(f"Texts requiring new BGE embedding: {len(texts_to_embed)}")
    
    if len(texts_to_embed) > 0:
        from src.embeddings.embed_model import BGEEmbedder
        embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", normalize=True)
        new_tids = list(texts_to_embed.keys())
        new_claim_txts = [texts_to_embed[tid]["central_claim"] for tid in new_tids]
        new_reason_txts = [texts_to_embed[tid]["core_reasoning"] for tid in new_tids]
        new_warrant_txts = [texts_to_embed[tid]["underlying_warrant"] for tid in new_tids]
        
        new_c_embs = embedder.embed(new_claim_txts, batch_size=32)
        new_r_embs = embedder.embed(new_reason_txts, batch_size=32)
        new_w_embs = embedder.embed(new_warrant_txts, batch_size=32)
        
        # Append to caches
        all_tids = text_ids_cached + new_tids
        all_c_embs = np.vstack([claim_embs_cached, new_c_embs])
        all_r_embs = np.vstack([reason_embs_cached, new_r_embs])
        all_w_embs = np.vstack([warrant_embs_cached, new_w_embs])
        
        # Save updated cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
            
        np.savez_compressed(
            npz_path,
            text_ids=np.array(all_tids),
            claim_embeddings=all_c_embs,
            reasoning_embeddings=all_r_embs,
            warrant_embeddings=all_w_embs
        )
        print("Updated cache and embedding NPZ.")
        tid_to_idx = {tid: i for i, tid in enumerate(all_tids)}
        claim_embs_cached = all_c_embs
        reason_embs_cached = all_r_embs
        warrant_embs_cached = all_w_embs
        
    # 4. Compute pairwise reasoning similarities for u100
    r_cosines = []
    c_cosines = []
    w_cosines = []
    
    for _, row in u100.iterrows():
        g_id = compute_stable_text_id(row['gold_perspective_text'])
        h_id = compute_stable_text_id(row['generated_viewpoint_text'])
        g_idx = tid_to_idx[g_id]
        h_idx = tid_to_idx[h_id]
        
        c_cos = float(np.dot(claim_embs_cached[g_idx], claim_embs_cached[h_idx]))
        r_cos = float(np.dot(reason_embs_cached[g_idx], reason_embs_cached[h_idx]))
        w_cos = float(np.dot(warrant_embs_cached[g_idx], warrant_embs_cached[h_idx]))
        
        c_cosines.append(round(max(-1.0, min(1.0, c_cos)), 4))
        r_cosines.append(round(max(-1.0, min(1.0, r_cos)), 4))
        w_cosines.append(round(max(-1.0, min(1.0, w_cos)), 4))
        
    u100['claim_cosine_v1_1'] = c_cosines
    u100['reasoning_cosine_v1_1'] = r_cosines
    u100['warrant_cosine_v1_1'] = w_cosines
    
    # Stance and Stage 1 filters
    is_contra = (
        ((u100['stance_gold'] == 'support') & (u100['stance_generated'] == 'undermine')) |
        ((u100['stance_gold'] == 'undermine') & (u100['stance_generated'] == 'support'))
    )
    s1_pass = u100['cosine_similarity'] >= 0.55
    s2a_pass = s1_pass & (~is_contra)
    u100['pass_s2a'] = s2a_pass
    
    # Compute probe probabilities for pairs that pass Stage 1 and Stage 2a
    u100['prob_m0'] = 0.0
    u100['prob_m2'] = 0.0
    
    if s2a_pass.sum() > 0:
        s2a_idx = u100[s2a_pass].index
        u100.loc[s2a_idx, 'prob_m0'] = clf_m0.predict_proba(sc_m0.transform(u100.loc[s2a_idx, feats_m0].values))[:, 1]
        u100.loc[s2a_idx, 'prob_m2'] = clf_m2.predict_proba(sc_m2.transform(u100.loc[s2a_idx, feats_m2].values))[:, 1]
    
    # Decisions under cascade
    u100['pass_m0_020'] = s2a_pass & (u100['prob_m0'] >= 0.20)
    u100['pass_m2_018'] = s2a_pass & (u100['prob_m2'] >= 0.18)
    u100['pass_m2_031'] = s2a_pass & (u100['prob_m2'] >= 0.31)
    
    # Save enriched u100 features
    out_enriched = "data/audit/unified_production_audit_benchmark_100_enriched_e3.csv"
    u100.to_csv(out_enriched, index=False)
    print(f"Saved enriched 100 audit dataset to: {out_enriched}")
    
    # =========================================================================
    # EXPERIMENT E3.4: CLAIM-DISJOINT INDEPENDENT VALIDATION
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT E3.4: CLAIM-DISJOINT INDEPENDENT VALIDATION")
    print("=" * 90)
    
    # Load 341 partition claims
    with open("data/calibration/stage2_claim_disjoint_partitions_human_341.json", "r") as f:
        p_json = json.load(f)
    test_claims = set(p_json['partitions']['test']['claim_ids'])
    train_claims = set(p_json['partitions']['train']['claim_ids'])
    
    val_df = u100[~u100['claim_id'].isin(test_claims)].copy()
    print(f"Independent Validation Set (Excluding 17 Test Claims): {len(val_df)} pairs across {val_df['claim_id'].nunique()} claims")
    print("Label Breakdown:")
    print(val_df['human_audit_decision'].value_counts().to_string())
    
    def evaluate_cohort(cohort_name, c_df):
        print(f"\n--- {cohort_name} (N = {len(c_df)}) ---")
        c2 = (c_df['human_audit_decision'] == 'class_2')
        c1 = (c_df['human_audit_decision'] == 'class_1')
        c0a = (c_df['human_audit_decision'] == 'class_0a')
        c0b = (c_df['human_audit_decision'] == 'class_0b')
        c_neg = ~c2
        
        models_eval = [
            ("M0 Baseline (T*=0.20)", c_df['pass_m0_020']),
            ("M2 Reasoning (T*=0.18)", c_df['pass_m2_018']),
            ("M2 Reasoning (T*=0.31)", c_df['pass_m2_031'])
        ]
        
        res_list = []
        for m_name, pass_col in models_eval:
            tp = (pass_col & c2).sum()
            fp = (pass_col & c_neg).sum()
            fn = ((~pass_col) & c2).sum()
            tn = ((~pass_col) & c_neg).sum()
            
            fp1 = (pass_col & c1).sum()
            fp0a = (pass_col & c0a).sum()
            fp0b = (pass_col & c0b).sum()
            
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            spec = tn / max(tn + fp, 1)
            rej_c1 = (c1.sum() - fp1) / max(c1.sum(), 1)
            rej_c0a = (c0a.sum() - fp0a) / max(c0a.sum(), 1)
            
            prec_ci = wilson_ci(tp, tp + fp)
            rec_ci = wilson_ci(tp, tp + fn)
            
            print(f"{m_name:24s} | Rec: {rec*100:5.1f}% ({tp}/{c2.sum()}) | Prec: {prec*100:5.1f}% ({tp}/{tp+fp}) [95% CI: {prec_ci[0]*100:.1f}-{prec_ci[1]*100:.1f}%] | F1: {f1:.4f} | Rej C1: {rej_c1*100:5.1f}% ({c1.sum()-fp1}/{c1.sum()}) | Tot FP: {fp} (C1={fp1}, C0a={fp0a}, C0b={fp0b})")
            res_list.append({
                "Cohort": cohort_name,
                "Model": m_name,
                "Recall": rec,
                "Precision": prec,
                "F1": f1,
                "Rej_C1": rej_c1,
                "Rej_C0a": rej_c0a,
                "Total_FP": fp,
                "FP_C1": fp1,
                "FP_C0a": fp0a,
                "FP_C0b": fp0b
            })
        return res_list
        
    e3_4_results = evaluate_cohort("Cohort 1: Non-Test Claims (Claim-Disjoint from Test)", val_df)
    
    # Also strictly fresh claims (disjoint from both train and test)
    fresh_df = u100[~u100['claim_id'].isin(test_claims | train_claims)].copy()
    e3_4_fresh = evaluate_cohort("Cohort 2: Fresh Claims (Disjoint from both Train & Test)", fresh_df)
    
    # =========================================================================
    # EXPERIMENT E3.5: FULL POPULATION SAMPLING-WEIGHTED PRODUCTION AUDIT
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT E3.5: FULL POPULATION SAMPLING-WEIGHTED PRODUCTION AUDIT")
    print("=" * 90)
    
    # Horvitz-Thompson strata sizes (N_pop = 250)
    prod_strata_sizes = {
        'ACCEPT_CLASS2_EQUIVALENT': 74,
        'REJECT_STAGE2B_DIFFERENT_WARRANT': 70,
        'REJECT_STAGE2A_STANCE': 56,
        'REJECT_STAGE1': 50
    }
    
    sample_strata_counts = u100['final_decision'].value_counts().to_dict()
    strata_weights = {dec: prod_strata_sizes[dec] / max(sample_strata_counts.get(dec, 0), 1) for dec in prod_strata_sizes}
    u100['stratum_weight'] = u100['final_decision'].map(strata_weights)
    
    print("Strata Sampling Weights (Horvitz-Thompson Estimator):")
    for dec, w in strata_weights.items():
        print(f"  {dec:<35}: Pop N={prod_strata_sizes[dec]}, Sample n={sample_strata_counts.get(dec, 0)}, Weight W={w:.4f}")
        
    models_prod = [
        ("M0 Frozen Baseline (T*=0.20)", u100['pass_m0_020']),
        ("M2 Reasoning (T*=0.18)", u100['pass_m2_018']),
        ("M2 Reasoning (T*=0.31)", u100['pass_m2_031'])
    ]
    
    prod_audit_records = []
    print("\n--- POPULATION-WEIGHTED ESTIMATES (N = 250) ---")
    for m_name, pass_col in models_prod:
        pop_tp = 0.0
        pop_fp = 0.0
        pop_fn = 0.0
        pop_tn = 0.0
        
        pop_fp_c1 = 0.0
        pop_fp_c0a = 0.0
        pop_fp_c0b = 0.0
        
        pop_c2 = 0.0
        pop_c1 = 0.0
        pop_c0a = 0.0
        pop_c0b = 0.0
        
        for _, r in u100.iterrows():
            w = r['stratum_weight']
            lbl = r['human_audit_decision']
            pred = pass_col.loc[_]
            
            if lbl == 'class_2':
                pop_c2 += w
                if pred:
                    pop_tp += w
                else:
                    pop_fn += w
            else:
                if lbl == 'class_1':
                    pop_c1 += w
                    if pred:
                        pop_fp_c1 += w
                elif lbl == 'class_0a':
                    pop_c0a += w
                    if pred:
                        pop_fp_c0a += w
                elif lbl == 'class_0b':
                    pop_c0b += w
                    if pred:
                        pop_fp_c0b += w
                if pred:
                    pop_fp += w
                else:
                    pop_tn += w
                    
        wt_rec = pop_tp / max(pop_c2, 1e-8)
        wt_prec = pop_tp / max(pop_tp + pop_fp, 1e-8)
        wt_f1 = 2 * wt_prec * wt_rec / max(wt_prec + wt_rec, 1e-8)
        wt_c1_rej = (pop_c1 - pop_fp_c1) / max(pop_c1, 1e-8)
        wt_c0a_rej = (pop_c0a - pop_fp_c0a) / max(pop_c0a, 1e-8)
        
        print(f"\n{m_name}:")
        print(f"  Population Total Positives Predicted: {pop_tp + pop_fp:.1f} / 250.0")
        print(f"  Weighted Class 2 Recall:    {wt_rec*100:5.2f}% (est. {pop_tp:.1f}/{pop_c2:.1f})")
        print(f"  Weighted Class 2 Precision: {wt_prec*100:5.2f}% (est. {pop_tp:.1f}/{pop_tp+pop_fp:.1f})")
        print(f"  Weighted Class 2 F1-Score:  {wt_f1:6.4f}")
        print(f"  Weighted Class 1 Rejection: {wt_c1_rej*100:5.2f}% (est. {pop_c1-pop_fp_c1:.1f}/{pop_c1:.1f})")
        print(f"  Weighted False Positives:   {pop_fp:.1f} (C1: {pop_fp_c1:.1f}, C0a: {pop_fp_c0a:.1f}, C0b: {pop_fp_c0b:.1f})")
        
        prod_audit_records.append({
            "model": m_name,
            "weighted_recall": round(wt_rec, 4),
            "weighted_precision": round(wt_prec, 4),
            "weighted_f1": round(wt_f1, 4),
            "weighted_c1_rejection": round(wt_c1_rej, 4),
            "est_population_tp": round(pop_tp, 1),
            "est_population_fp": round(pop_fp, 1),
            "est_population_fn": round(pop_fn, 1),
            "est_population_tn": round(pop_tn, 1)
        })
        
    # Save audit report
    out_rep = "results/calibration/e3_5_production_audit_comparison.json"
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(prod_audit_records, f, indent=2)
    print(f"\nSaved production audit comparison to: {out_rep}")

if __name__ == '__main__':
    main()
