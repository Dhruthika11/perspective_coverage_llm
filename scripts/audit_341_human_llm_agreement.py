import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

def main():
    csv_path = 'results/calibration/stage2_cross_encoder_predictions_341.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pairs from {csv_path}")

    # Standard class ordering
    classes = ['class_2', 'class_1', 'class_0a', 'class_0b']
    class_labels_display = ['Class 2 (Equiv)', 'Class 1 (Hard Neg)', 'Class 0a (Opposite)', 'Class 0b (Noise)']

    # 1. Inspect Labels
    y_human = df['human_ground_truth_label'].astype(str).str.strip().tolist()
    y_jury_raw = df['jury_consensus_label'].astype(str).str.strip().tolist()

    # Determine resolved LLM majority for all 341 pairs:
    # If jury_consensus_label != 'split', use it directly.
    # If 'split', use Gemini-3-Flash vote as primary tiebreaker (or highest confidence valid vote).
    resolved_llm_majority = []
    split_count = 0
    for idx, row in df.iterrows():
        maj = str(row['jury_consensus_label']).strip()
        if maj in classes:
            resolved_llm_majority.append(maj)
        else:
            split_count += 1
            # Tiebreaker: Gemini vote if valid, else Qwen, else GPT
            g_vote = str(row['Gemini-3-Flash_label']).strip()
            q_vote = str(row['Qwen-3.8-27B_label']).strip()
            o_vote = str(row['GPT-OSS-120B_label']).strip()
            
            if g_vote in classes:
                resolved_llm_majority.append(g_vote)
            elif q_vote in classes:
                resolved_llm_majority.append(q_vote)
            elif o_vote in classes:
                resolved_llm_majority.append(o_vote)
            else:
                resolved_llm_majority.append('class_1') # fallback

    df['llm_majority_resolved'] = resolved_llm_majority

    # Subset 1: Definitive Consensus (N = 312, excluding 29 splits)
    mask_def = df['jury_consensus_label'].isin(classes)
    df_def = df[mask_def].copy()
    y_human_def = df_def['human_ground_truth_label'].tolist()
    y_jury_def = df_def['jury_consensus_label'].tolist()

    # Subset 2: Full 341 pairs (with tie-broken splits)
    y_human_all = y_human
    y_jury_all = resolved_llm_majority

    # ---------------------------------------------------------
    # PART 1: MEASURE AGREEMENT
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: INTER-ANNOTATOR AGREEMENT (HUMAN MANUAL vs LLM JURY MAJORITY)")
    print("=" * 80)

    for name, yh, yj in [("Definitive Consensus (N=312)", y_human_def, y_jury_def), 
                         ("Full Benchmark (N=341, with tiebreak)", y_human_all, y_jury_all)]:
        print(f"\n--- Evaluation Subset: {name} ---")
        acc = np.mean(np.array(yh) == np.array(yj)) * 100.0
        kappa = cohen_kappa_score(yh, yj, labels=classes)
        print(f"Overall Agreement: {acc:.2f}%")
        print(f"Cohen's Kappa (κ): {kappa:.4f}")

        # Confusion Matrix (Rows = Human Reference, Cols = LLM Majority)
        cm = confusion_matrix(yh, yj, labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
        print("\n4x4 Confusion Matrix (Rows: Human Manual Ref, Cols: LLM Majority):")
        print(cm_df.to_string())

        # Per-Class Precision, Recall, F1 (treating Human Manual as Reference)
        report = classification_report(yh, yj, labels=classes, target_names=class_labels_display, output_dict=True)
        rep_df = pd.DataFrame(report).transpose()
        print("\nPer-Class Performance Metrics (Human Manual as Reference Ground Truth):")
        print(rep_df[['precision', 'recall', 'f1-score', 'support']].to_string())

        # Specific Class 2 vs Class 1 Analysis
        # True C2 predicted as C1 (Missed Equivalence)
        c2_idx = classes.index('class_2')
        c1_idx = classes.index('class_1')
        
        missed_equiv = cm[c2_idx, c1_idx] # Human=C2, LLM=C1
        total_c2 = sum(cm[c2_idx, :])
        pct_missed_equiv = (missed_equiv / total_c2) * 100.0 if total_c2 > 0 else 0.0

        false_equiv = cm[c1_idx, c2_idx] # Human=C1, LLM=C2
        total_c1 = sum(cm[c1_idx, :])
        pct_false_equiv = (false_equiv / total_c1) * 100.0 if total_c1 > 0 else 0.0

        print("\nCRITICAL CLASS 2 vs CLASS 1 CONFUSION ANALYSIS:")
        print(f"  1. LLM False Equivalence (Human = Class 1 Hard Negative, LLM = Class 2 Match):")
        print(f"     Count: {false_equiv} / {total_c1} ({pct_false_equiv:.2f}%)")
        print(f"     Meaning: The LLM majority mistakenly declared equivalence when there was a divergent warrant.")
        print(f"  2. LLM Missed Equivalence (Human = Class 2 Match, LLM = Class 1 Hard Negative):")
        print(f"     Count: {missed_equiv} / {total_c2} ({pct_missed_equiv:.2f}%)")
        print(f"     Meaning: The LLM majority rejected a valid perspective match due to surface lexical differences.")

    # ---------------------------------------------------------
    # PART 2: DISAGREEMENT AUDIT & CATEGORIZATION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: CATEGORIZATION OF DISAGREEMENTS")
    print("=" * 80)

    disagreements = []
    for idx, row in df.iterrows():
        h_label = row['human_ground_truth_label']
        llm_label = row['llm_majority_resolved']
        pid = row['pair_id']
        claim = row['claim_text']
        gold = row['gold_perspective_text']
        gen = row['generated_viewpoint_text']
        g_vote = row['Gemini-3-Flash_label']
        q_vote = row['Qwen-3.8-27B_label']
        o_vote = row['GPT-OSS-120B_label']
        c_type = row['jury_consensus_type']
        rationale = row['human_adjudication_rationale']

        if h_label != llm_label:
            # Categorize
            category = ""
            details = ""
            if h_label == 'class_1' and llm_label == 'class_2':
                category = "LLM missed a difference in the underlying warrant"
                details = "LLM accepted shared stance but overlooked warrant divergence."
            elif h_label == 'class_2' and llm_label == 'class_1':
                category = "LLM missed equivalence expressed using different wording"
                details = "Human identified semantic paraphrase; LLM treated lexical divergence as distinct argument."
            elif c_type == 'split_1_1_1' or 'borderline' in str(rationale).lower() or 'sub-aspect' in str(rationale).lower():
                category = "Genuine ambiguity in the definition of equivalent reasoning"
                details = "3-way split or high semantic nuance between broad stance and sub-aspect scope."
            elif (h_label in ['class_0a', 'class_0b'] and llm_label in ['class_1', 'class_2']) or \
                 (h_label in ['class_1', 'class_2'] and llm_label in ['class_0a', 'class_0b']):
                category = "Manual label or LLM label boundary check (Stance/Noise discrepancy)"
                details = "Discrepancy at the stance polarity or topic relevance boundary."
            else:
                category = "Genuine ambiguity in the definition of equivalent reasoning"
                details = "Subtle rhetorical variation."

            disagreements.append({
                "pair_id": pid,
                "claim_id": row['claim_id'],
                "claim": claim,
                "gold": gold,
                "gen": gen,
                "human_label": h_label,
                "llm_label": llm_label,
                "jury_consensus_type": c_type,
                "gemini": g_vote,
                "qwen": q_vote,
                "gpt": o_vote,
                "category": category,
                "details": details,
                "human_rationale": rationale
            })

    df_dis = pd.DataFrame(disagreements)
    print(f"Total Disagreements between Human Manual and Resolved LLM Majority: {len(df_dis)} / {len(df)} ({len(df_dis)/len(df)*100:.2f}%)")
    cat_counts = df_dis['category'].value_counts()
    print("\nBreakdown by Disagreement Category:")
    for cat, count in cat_counts.items():
        print(f"  - {cat}: {count} pairs ({count/len(df_dis)*100:.1f}%)")

    # Save disagreements
    dis_out_path = 'results/calibration/disagreements_human_vs_llm_majority_341.csv'
    df_dis.to_csv(dis_out_path, index=False)
    print(f"\nSaved detailed disagreements to {dis_out_path}")

    # ---------------------------------------------------------
    # PART 3: PRESERVE THREE BENCHMARK VERSIONS & EVALUATE M0
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: PRESERVE THREE BENCHMARK VERSIONS & EVALUATE M0 ROBUSTNESS")
    print("=" * 80)

    # Version A: Human Manual Reference
    df_vA = df.copy()
    df_vA['evaluation_label'] = df_vA['human_ground_truth_label']
    df_vA['benchmark_version'] = 'Version A (Human Manual Reference)'
    vA_path = 'results/calibration/benchmark_341_version_A_manual.csv'
    df_vA.to_csv(vA_path, index=False)
    print(f"Saved Version A (Human Manual) -> {vA_path}")

    # Version B: Three-LLM Majority Consensus
    df_vB = df.copy()
    df_vB['evaluation_label'] = df_vB['llm_majority_resolved']
    df_vB['benchmark_version'] = 'Version B (LLM Jury Majority)'
    vB_path = 'results/calibration/benchmark_341_version_B_llm_majority.csv'
    df_vB.to_csv(vB_path, index=False)
    print(f"Saved Version B (LLM Majority) -> {vB_path}")

    # Version C: Adjudicated Disagreements (Refined Reference Set)
    # Refined label: uses human adjudication where audited, preserving refined gold labels
    refined_labels = []
    for idx, row in df.iterrows():
        # In audited pairs where human provided detailed rationale, preserve human adjudication
        # If human noted reconsideration or confirmed agreement, use refined consensus
        h_lab = row['human_ground_truth_label']
        maj = row['llm_majority_resolved']
        rat = str(row['human_adjudication_rationale']).lower()
        if "agreed with 2-of-3" in rat or "unanimous" in rat:
            refined_labels.append(maj)
        elif "overrode" in rat or "adjudicated" in rat:
            refined_labels.append(h_lab)
        else:
            refined_labels.append(h_lab)
    df['refined_adjudicated_label'] = refined_labels

    df_vC = df.copy()
    df_vC['evaluation_label'] = refined_labels
    df_vC['benchmark_version'] = 'Version C (Adjudicated Disagreements Refined)'
    vC_path = 'results/calibration/benchmark_341_version_C_adjudicated.csv'
    df_vC.to_csv(vC_path, index=False)
    print(f"Saved Version C (Refined Adjudicated) -> {vC_path}")

    # Evaluate M0 on Version A vs Version B vs Version C
    # Load Frozen Warrant Probe
    with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
        config = json.load(f)

    scaler_mean = np.array(config['scaler_mean'])
    scaler_scale = np.array(config['scaler_scale'])
    w_cos = config['coefficients']['cosine_similarity']
    w_ent = config['coefficients']['p_entail']
    w_neu = config['coefficients']['p_neutral']
    intercept = config['intercept']
    tau_s1 = 0.55
    t_probe = config['calibrated_threshold'] # 0.20

    # Load 1700 benchmark for stances
    df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
    df = df.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

    df['stage1_pass'] = df['cosine_similarity'] >= tau_s1
    g_st = df['stance_gold'].astype(str).str.lower().str.strip()
    h_st = df['stance_generated'].astype(str).str.lower().str.strip()
    is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
    df['stage2a_pass'] = df['stage1_pass'] & (~is_contra)

    z_cos = (df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
    z_ent = (df['p_entail'] - scaler_mean[1]) / scaler_scale[1]
    z_neu = (df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
    probe_prob = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
    df['m0_accept'] = df['stage2a_pass'] & (probe_prob >= t_probe)

    m0_comparison = []
    for v_name, label_col in [("Version A (Human Manual)", "human_ground_truth_label"),
                              ("Version B (LLM Majority)", "llm_majority_resolved"),
                              ("Version C (Refined Adjudicated)", "refined_adjudicated_label")]:
        y_true_binary = (df[label_col] == 'class_2')
        y_pred_binary = df['m0_accept']

        tp = int(np.sum(y_true_binary & y_pred_binary))
        fp = int(np.sum((~y_true_binary) & y_pred_binary))
        fn = int(np.sum(y_true_binary & (~y_pred_binary)))
        tn = int(np.sum((~y_true_binary) & (~y_pred_binary)))

        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        spec = tn / max(tn + fp, 1)

        # Class 1 rejection
        c1_mask = (df[label_col] == 'class_1')
        c1_total = int(c1_mask.sum())
        c1_rejected = int((c1_mask & (~y_pred_binary)).sum())
        c1_rej_rate = c1_rejected / max(c1_total, 1)

        # Class 0a rejection
        c0a_mask = (df[label_col] == 'class_0a')
        c0a_total = int(c0a_mask.sum())
        c0a_rejected = int((c0a_mask & (~y_pred_binary)).sum())
        c0a_rej_rate = c0a_rejected / max(c0a_total, 1)

        m0_comparison.append({
            "Benchmark Version": v_name,
            "Total Positives (C2)": int(y_true_binary.sum()),
            "Total Negatives (Non-C2)": int((~y_true_binary).sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Recall (C2)": f"{rec*100:.2f}%",
            "Precision (C2)": f"{prec*100:.2f}%",
            "F1-Score": f"{f1:.4f}",
            "Specificity": f"{spec*100:.2f}%",
            "Class 1 Rejection": f"{c1_rej_rate*100:.2f}% ({c1_rejected}/{c1_total})",
            "Class 0a Rejection": f"{c0a_rej_rate*100:.2f}% ({c0a_rejected}/{c0a_total})"
        })

    df_m0_comp = pd.DataFrame(m0_comparison)
    print("\nROBUSTNESS EVALUATION OF FROZEN M0 ACROSS BENCHMARK VERSIONS:")
    print(df_m0_comp.to_string(index=False))

    # Save summary report
    summary_report = {
        "overall_agreement_definitive_n312": float(np.mean(np.array(y_human_def) == np.array(y_jury_def)) * 100.0),
        "kappa_definitive_n312": float(cohen_kappa_score(y_human_def, y_jury_def, labels=classes)),
        "overall_agreement_all_n341": float(np.mean(np.array(y_human_all) == np.array(y_jury_all)) * 100.0),
        "kappa_all_n341": float(cohen_kappa_score(y_human_all, y_jury_all, labels=classes)),
        "disagreement_counts": {cat: int(cnt) for cat, cnt in cat_counts.items()},
        "m0_comparison": m0_comparison
    }
    with open('results/calibration/human_vs_llm_agreement_and_versions_report.json', 'w') as f:
        json.dump(summary_report, f, indent=2)

if __name__ == '__main__':
    main()
