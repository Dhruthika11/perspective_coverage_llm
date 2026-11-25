import pandas as pd
import numpy as np
import json
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

def main():
    print("=" * 80)
    print("STAGE 2 INDEPENDENT AUDIT POST-SYNTHESIS & BENCHMARK C RE-CALIBRATION")
    print("=" * 80)

    # 1. Load Audit Queue
    audit_path = 'data/audit/independent_quality_audit_90_pairs.csv'
    df_audit = pd.read_csv(audit_path)
    assert df_audit['re_audit_label'].notna().all(), "Some audit pairs still missing labels!"
    print(f"Loaded {len(df_audit)} completed audit pairs from {audit_path}")

    # Standardize label strings
    df_audit['re_audit_label'] = df_audit['re_audit_label'].str.strip().str.lower()
    df_audit['human_ground_truth_label'] = df_audit['human_ground_truth_label'].str.strip().str.lower()
    df_audit['jury_consensus_label'] = df_audit['jury_consensus_label'].str.strip().str.lower()

    classes = ['class_2', 'class_1', 'class_0a', 'class_0b']

    # -------------------------------------------------------------
    # 2. Control Sample Analysis (N = 35)
    # -------------------------------------------------------------
    controls = df_audit[df_audit['audit_stratum'] == 'control_agreement'].copy()
    control_agree = (controls['human_ground_truth_label'] == controls['re_audit_label']).mean() * 100.0
    control_kappa = cohen_kappa_score(controls['human_ground_truth_label'], controls['re_audit_label'], labels=classes)
    cm_control = confusion_matrix(controls['human_ground_truth_label'], controls['re_audit_label'], labels=classes)

    print("\n" + "-" * 50)
    print("1. CONTROL SAMPLE VALIDATION (N = 35)")
    print("-" * 50)
    print(f"Test-Retest Agreement: {control_agree:.2f}% ({(controls['human_ground_truth_label'] == controls['re_audit_label']).sum()}/{len(controls)})")
    print(f"Cohen's Kappa (Retest Reliability): {control_kappa:.4f}")
    print("\nConfusion Matrix (Rows: Original Manual, Cols: Re-Audit):")
    print(pd.DataFrame(cm_control, index=classes, columns=classes))

    # -------------------------------------------------------------
    # 3. Disagreement Sample Analysis (N = 55)
    # -------------------------------------------------------------
    disags = df_audit[df_audit['audit_stratum'] == 'disagreement'].copy()
    print("\n" + "-" * 50)
    print("2. DISAGREEMENT SAMPLE ADJUDICATION (N = 55)")
    print("-" * 50)

    # Compare re-audit with original human vs LLM jury
    side_with_human = (disags['re_audit_label'] == disags['human_ground_truth_label']).sum()
    side_with_llm = (disags['re_audit_label'] == disags['jury_consensus_label']).sum()
    other_verdict = len(disags) - side_with_human - side_with_llm

    print(f"Total Disagreements: {len(disags)}")
    print(f"  Re-Audit Sided with Original Human: {side_with_human} / {len(disags)} ({side_with_human/len(disags)*100:.2f}%)")
    print(f"  Re-Audit Sided with LLM Jury:       {side_with_llm} / {len(disags)} ({side_with_llm/len(disags)*100:.2f}%)")
    print(f"  Re-Audit Identified New / Polarity: {other_verdict} / {len(disags)} ({other_verdict/len(disags)*100:.2f}%)")

    # The crucial 27 cases: Human C2 vs LLM C1
    c2_v_c1 = disags[(disags['human_ground_truth_label'] == 'class_2') & (disags['jury_consensus_label'] == 'class_1')]
    print(f"\nBreakdown of the 27 Human Class 2 vs LLM Class 1 Disagreements:")
    c2_v_c1_counts = c2_v_c1['re_audit_label'].value_counts()
    for lab, cnt in c2_v_c1_counts.items():
        print(f"  Final Adjudicated {lab}: {cnt} / {len(c2_v_c1)} ({cnt/len(c2_v_c1)*100:.2f}%)")

    # Categorize all 55 disagreements based on adjudication outcome
    disag_categories = []
    for _, r in disags.iterrows():
        orig_h = r['human_ground_truth_label']
        llm = r['jury_consensus_label']
        fin = r['re_audit_label']
        pid = r['pair_id']
        
        if orig_h == 'class_2' and llm == 'class_1':
            if fin == 'class_2':
                cat = "LLM Conservative Bias: Missed semantic paraphrase / equivalent warrant"
            elif fin == 'class_1':
                cat = "Human Self-Correction: LLM correctly identified distinct warrant"
            else:
                cat = "Re-evaluation / Boundary adjustment"
        elif orig_h == 'class_1' and llm == 'class_2':
            if fin == 'class_1':
                cat = "Human Reaffirmed: Divergent warrant maintained (LLM false equivalence)"
            elif fin == 'class_2':
                cat = "Human Self-Correction: Accepted equivalence (agreed with LLM)"
            else:
                cat = "Re-evaluation / Boundary adjustment"
        elif llm in ['split', 'split_1_1_1']:
            cat = f"Resolved Unstable LLM Split -> Adjudicated as {fin}"
        elif (orig_h in ['class_0a', 'class_0b']) or (llm in ['class_0a', 'class_0b']) or (fin in ['class_0a', 'class_0b']):
            cat = "Polarity / Relevance Boundary Clarification"
        else:
            cat = "Nuanced Granularity Adjudication"

        disag_categories.append({
            "pair_id": pid,
            "claim": r['claim_text'],
            "original_human": orig_h,
            "llm_consensus": llm,
            "final_adjudicated": fin,
            "category": cat
        })

    df_disag_cat = pd.DataFrame(disag_categories)
    print("\nDisagreement Categorization Summary:")
    print(df_disag_cat['category'].value_counts())

    # -------------------------------------------------------------
    # 4. Finalize Benchmark Version C (N = 341)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. UPDATING FINAL BENCHMARK VERSION C (N = 341)")
    print("-" * 50)

    df_vA = pd.read_csv('results/calibration/benchmark_341_version_A_manual.csv')
    df_vB = pd.read_csv('results/calibration/benchmark_341_version_B_llm_majority.csv')
    
    # Audit lookup dict: pair_id -> re_audit_label
    audit_lookup = dict(zip(df_audit['pair_id'], df_audit['re_audit_label']))

    # Construct Version C
    df_vC = df_vA.copy()
    final_adjudicated_labels = []
    source_audit_status = []

    for _, row in df_vC.iterrows():
        pid = row['pair_id']
        h_label = row['human_ground_truth_label']
        if pid in audit_lookup:
            final_adjudicated_labels.append(audit_lookup[pid])
            source_audit_status.append('independently_audited')
        else:
            # Pair was an un-sampled control agreement (unanimous between human and LLM)
            final_adjudicated_labels.append(h_label)
            source_audit_status.append('unanimous_consensus_unaltered')

    df_vC['adjudicated_label'] = final_adjudicated_labels
    df_vC['evaluation_label'] = final_adjudicated_labels
    df_vC['audit_status'] = source_audit_status
    df_vC['benchmark_version'] = 'Version C (Adjudicated Reference Set)'
    vC_path = 'results/calibration/benchmark_341_version_C_adjudicated.csv'
    df_vC.to_csv(vC_path, index=False)
    print(f"Saved finalized Version C to {vC_path}")

    # Agreement between Version C and Versions A & B
    agree_C_A = (df_vC['adjudicated_label'] == df_vA['human_ground_truth_label']).mean() * 100.0
    kappa_C_A = cohen_kappa_score(df_vA['human_ground_truth_label'], df_vC['adjudicated_label'], labels=classes)

    valid_b_mask = df_vB['jury_consensus_label'].isin(classes)
    agree_C_B = (df_vC.loc[valid_b_mask, 'adjudicated_label'] == df_vB.loc[valid_b_mask, 'jury_consensus_label']).mean() * 100.0
    kappa_C_B = cohen_kappa_score(df_vB.loc[valid_b_mask, 'jury_consensus_label'], df_vC.loc[valid_b_mask, 'adjudicated_label'], labels=classes)

    print(f"Version C vs Version A (Human Manual): Agreement = {agree_C_A:.2f}%, Cohen's Kappa = {kappa_C_A:.4f}")
    print(f"Version C vs Version B (LLM Consensus): Agreement = {agree_C_B:.2f}%, Cohen's Kappa = {kappa_C_B:.4f}")

    print("\nClass Distribution in Finalized Benchmark Version C:")
    print(df_vC['adjudicated_label'].value_counts())

    # -------------------------------------------------------------
    # 5. Evaluate Frozen M0 and Soft Gate across Benchmark Versions
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("4. M0 & M_SOFT EVALUATION ACROSS BENCHMARK VERSIONS")
    print("-" * 50)

    with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
        probe_cfg = json.load(f)

    scaler_mean = np.array(probe_cfg['scaler_mean'])
    scaler_scale = np.array(probe_cfg['scaler_scale'])
    w_cos = probe_cfg['coefficients']['cosine_similarity']
    w_ent = probe_cfg['coefficients']['p_entail']
    w_neu = probe_cfg['coefficients']['p_neutral']
    intercept = probe_cfg['intercept']
    tau_s1 = 0.55
    t_probe = probe_cfg['calibrated_threshold']

    # Ensure stances are merged
    df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
    eval_df = df_vC.copy()
    if 'stance_gold' not in eval_df.columns:
        eval_df = eval_df.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

    eval_df['stage1_pass'] = eval_df['cosine_similarity'] >= tau_s1
    g_st = eval_df['stance_gold'].astype(str).str.lower().str.strip()
    h_st = eval_df['stance_generated'].astype(str).str.lower().str.strip()
    is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
    eval_df['stage2a_pass'] = eval_df['stage1_pass'] & (~is_contra)

    z_cos = (eval_df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
    z_ent = (eval_df['p_entail'] - scaler_mean[1]) / scaler_scale[1]
    z_neu = (eval_df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
    probe_prob = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
    eval_df['probe_prob'] = probe_prob
    eval_df['m0_accept'] = eval_df['stage2a_pass'] & (probe_prob >= t_probe)

    # Soft match weight
    # W_soft = cos * (1 - 0.7 * p_contra) * (p_entail + 0.3 * p_neutral)
    eval_df['p_contra'] = 1.0 - eval_df['p_entail'] - eval_df['p_neutral']
    eval_df['p_contra'] = eval_df['p_contra'].clip(lower=0.0)
    eval_df['w_soft'] = eval_df['cosine_similarity'] * (1.0 - 0.7 * eval_df['p_contra']) * (eval_df['p_entail'] + 0.3 * eval_df['p_neutral'])
    eval_df['msoft_accept'] = eval_df['stage1_pass'] & (eval_df['w_soft'] >= 0.25)

    eval_results = []
    versions = [
        ("Version A (Human Manual)", df_vA['human_ground_truth_label']),
        ("Version B (LLM Majority)", df_vB['llm_majority_resolved']),
        ("Version C (Final Adjudicated)", df_vC['adjudicated_label'])
    ]

    for v_name, label_series in versions:
        for m_name, pred_col in [("M0 (Hard Cascade, T*=0.20)", 'm0_accept'),
                                 ("M_soft (Gated Weight, thresh=0.25)", 'msoft_accept')]:
            y_true = (label_series == 'class_2')
            y_pred = eval_df[pred_col]

            tp = int(np.sum(y_true & y_pred))
            fp = int(np.sum((~y_true) & y_pred))
            fn = int(np.sum(y_true & (~y_pred)))
            tn = int(np.sum((~y_true) & (~y_pred)))

            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            spec = tn / max(tn + fp, 1)

            c1_mask = (label_series == 'class_1')
            c1_tot = int(c1_mask.sum())
            c1_rej = int((c1_mask & (~y_pred)).sum())
            c1_rej_rate = c1_rej / max(c1_tot, 1)

            c0a_mask = (label_series == 'class_0a')
            c0a_tot = int(c0a_mask.sum())
            c0a_rej = int((c0a_mask & (~y_pred)).sum())
            c0a_rej_rate = c0a_rej / max(c0a_tot, 1)

            eval_results.append({
                "Benchmark": v_name,
                "Model": m_name,
                "Positives (C2)": int(y_true.sum()),
                "Negatives": int((~y_true).sum()),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Recall (C2)": f"{rec*100:.2f}%",
                "Precision (C2)": f"{prec*100:.2f}%",
                "F1": f"{f1:.4f}",
                "Specificity": f"{spec*100:.2f}%",
                "C1 Rejection": f"{c1_rej_rate*100:.2f}% ({c1_rej}/{c1_tot})",
                "C0a Rejection": f"{c0a_rej_rate*100:.2f}% ({c0a_rej}/{c0a_tot})"
            })

    df_eval = pd.DataFrame(eval_results)
    print("\nPERFORMANCE MATRIX ACROSS BENCHMARK VERSIONS:")
    print(df_eval[['Benchmark', 'Model', 'Positives (C2)', 'Recall (C2)', 'Precision (C2)', 'F1', 'C1 Rejection', 'C0a Rejection']].to_string(index=False))

    # Save final report to JSON
    out_report = {
        "control_sample": {
            "n": len(controls),
            "agreement_pct": float(control_agree),
            "cohen_kappa": float(control_kappa),
            "confusion_matrix": cm_control.tolist()
        },
        "disagreement_sample": {
            "n": len(disags),
            "sided_with_original_human": int(side_with_human),
            "sided_with_llm": int(side_with_llm),
            "other_verdict": int(other_verdict),
            "c2_vs_c1_breakdown": {k: int(v) for k, v in c2_v_c1_counts.items()}
        },
        "benchmark_versions_comparison": {
            "version_C_vs_version_A": {
                "agreement_pct": float(agree_C_A),
                "cohen_kappa": float(kappa_C_A)
            },
            "version_C_vs_version_B": {
                "agreement_pct": float(agree_C_B),
                "cohen_kappa": float(kappa_C_B)
            },
            "class_distribution_version_C": {k: int(v) for k, v in df_vC['adjudicated_label'].value_counts().items()}
        },
        "eval_matrix": eval_results
    }

    report_path = 'results/calibration/post_audit_adjudicated_report.json'
    with open(report_path, 'w') as f:
        json.dump(out_report, f, indent=2)
    print(f"\nSaved full report to {report_path}")

if __name__ == '__main__':
    main()
