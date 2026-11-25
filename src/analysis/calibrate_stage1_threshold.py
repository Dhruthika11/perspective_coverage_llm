"""
Stage 1 (BGE Dense Retrieval) Threshold Calibration Script.

Analyzes the predictions from results/twostage_matched_benchmark_predictions.csv
to find the optimal operating threshold tau* for Stage 1.

Evaluates:
  1. Target Recall Rule: tau* s.t. Recall(TP) >= 80%, 85%, 90%
  2. Recall-Weighted F2-Score: argmax F2(tau)
  3. Complete Cascade Impact: How each tau affects final Two-Stage Precision & Recall
"""

import os
import sys
import numpy as np
import pandas as pd

def calibrate_stage1(pred_file="results/twostage_matched_benchmark_predictions.csv"):
    if not os.path.exists(pred_file):
        print(f"Error: Could not find {pred_file}")
        return

    df = pd.read_csv(pred_file)
    print("=" * 110)
    print("STAGE 1 (BGE DENSE RETRIEVAL) CALIBRATION & SENSITIVITY ANALYSIS")
    print(f"Loaded {len(df)} predictions from {pred_file}")
    print("=" * 110)

    for regime in ["Full (180w)", "Concise (58w)"]:
        sub_df = df[df["length_regime"] == regime].copy()
        tp_df = sub_df[sub_df["perspective_type"] == "True Positive"]
        opp_df = sub_df[sub_df["perspective_type"] == "Opposite-Stance"]
        ssdr_df = sub_df[sub_df["perspective_type"] == "Same-Stance Diff-Reason"]
        unrel_df = sub_df[sub_df["perspective_type"] == "Unrelated"]

        n_tp = len(tp_df)
        n_opp = len(opp_df)
        n_ssdr = len(ssdr_df)
        n_unrel = len(unrel_df)

        print(f"\n{'='*45} {regime} {'='*45}")
        print(f"Mean Stage 1 Cosine Similarities:")
        print(f"  - True Positive:             {tp_df['stage1_cosine'].mean():.4f} (std: {tp_df['stage1_cosine'].std():.4f}, median: {tp_df['stage1_cosine'].median():.4f})")
        print(f"  - Opposite-Stance:           {opp_df['stage1_cosine'].mean():.4f} (std: {opp_df['stage1_cosine'].std():.4f}, median: {opp_df['stage1_cosine'].median():.4f})")
        print(f"  - Same-Stance Diff-Reason:   {ssdr_df['stage1_cosine'].mean():.4f} (std: {ssdr_df['stage1_cosine'].std():.4f}, median: {ssdr_df['stage1_cosine'].median():.4f})")
        print(f"  - Unrelated:                 {unrel_df['stage1_cosine'].mean():.4f} (std: {unrel_df['stage1_cosine'].std():.4f}, median: {unrel_df['stage1_cosine'].median():.4f})")

        thresholds = np.arange(0.40, 0.81, 0.01)
        sweep_rows = []

        for tau in thresholds:
            tau = round(float(tau), 2)
            # Stage 1 pass masks
            s1_pass_tp = tp_df["stage1_cosine"] >= tau
            s1_pass_opp = opp_df["stage1_cosine"] >= tau
            s1_pass_ssdr = ssdr_df["stage1_cosine"] >= tau
            s1_pass_unrel = unrel_df["stage1_cosine"] >= tau

            tp_recall = s1_pass_tp.mean() * 100
            unrel_rej = (1.0 - s1_pass_unrel.mean()) * 100
            opp_rej = (1.0 - s1_pass_opp.mean()) * 100
            ssdr_rej = (1.0 - s1_pass_ssdr.mean()) * 100

            # Two-stage combined pass (Stage 1 >= tau AND Stage 2 p_entail >= 0.52)
            pipe_tp = (s1_pass_tp & (tp_df["p_entail"] >= 0.52)).sum()
            pipe_opp = (s1_pass_opp & (opp_df["p_entail"] >= 0.52)).sum()
            pipe_ssdr = (s1_pass_ssdr & (ssdr_df["p_entail"] >= 0.52)).sum()
            pipe_unrel = (s1_pass_unrel & (unrel_df["p_entail"] >= 0.52)).sum()

            pipe_total_accepted = pipe_tp + pipe_opp + pipe_ssdr + pipe_unrel
            pipe_prec = (pipe_tp / pipe_total_accepted * 100) if pipe_total_accepted > 0 else 0.0
            pipe_recall = (pipe_tp / n_tp * 100) if n_tp > 0 else 0.0
            
            p_frac = pipe_prec / 100.0
            r_frac = pipe_recall / 100.0
            pipe_f1 = (2 * p_frac * r_frac / (p_frac + r_frac) * 100) if (p_frac + r_frac) > 0 else 0.0
            pipe_f2 = (5 * p_frac * r_frac / (4 * p_frac + r_frac) * 100) if (4 * p_frac + r_frac) > 0 else 0.0

            sweep_rows.append({
                "tau": tau,
                "s1_tp_recall": tp_recall,
                "s1_unrel_rej": unrel_rej,
                "s1_opp_rej": opp_rej,
                "s1_ssdr_rej": ssdr_rej,
                "pipe_accepted": pipe_total_accepted,
                "pipe_tp": pipe_tp,
                "pipe_prec": pipe_prec,
                "pipe_recall": pipe_recall,
                "pipe_f1": pipe_f1,
                "pipe_f2": pipe_f2
            })

        sweep_df = pd.DataFrame(sweep_rows)

        # Print Key Calibration Operating Points
        print(f"\nOperating Points Table (Sampled across tau):")
        print(f"{'tau':<6} | {'S1 TP Recall':<12} | {'S1 Unrel Rej':<12} | {'S1 Opp Rej':<10} | {'Pipe Accepted':<13} | {'Pipe Prec':<10} | {'Pipe Recall':<11} | {'Pipe F1':<8} | {'Pipe F2':<8}")
        print("-" * 105)
        for target_tau in [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]:
            row = sweep_df[np.isclose(sweep_df["tau"], target_tau)].iloc[0]
            print(f"{row['tau']:<6.2f} | {row['s1_tp_recall']:>10.2f}% | {row['s1_unrel_rej']:>10.2f}% | {row['s1_opp_rej']:>8.2f}% | {int(row['pipe_accepted']):>13} | {row['pipe_prec']:>8.2f}% | {row['pipe_recall']:>9.2f}% | {row['pipe_f1']:>6.2f}% | {row['pipe_f2']:>6.2f}%")

        # Optimal Points
        # 1. Target S1 Recall >= 85%
        rec85_cands = sweep_df[sweep_df["s1_tp_recall"] >= 85.0]
        if not rec85_cands.empty:
            best_rec85 = rec85_cands.sort_values(by="tau", ascending=False).iloc[0]
            print(f"\n[Recommendation 1: 85% S1 Recall Target]")
            print(f"  tau* = {best_rec85['tau']:.2f} -> S1 TP Recall: {best_rec85['s1_tp_recall']:.2f}%, S1 Unrel Rej: {best_rec85['s1_unrel_rej']:.2f}%, Pipe Prec: {best_rec85['pipe_prec']:.2f}%, Pipe Recall: {best_rec85['pipe_recall']:.2f}%")

        # 2. Target S1 Recall >= 90%
        rec90_cands = sweep_df[sweep_df["s1_tp_recall"] >= 90.0]
        if not rec90_cands.empty:
            best_rec90 = rec90_cands.sort_values(by="tau", ascending=False).iloc[0]
            print(f"\n[Recommendation 2: 90% S1 Recall Target]")
            print(f"  tau* = {best_rec90['tau']:.2f} -> S1 TP Recall: {best_rec90['s1_tp_recall']:.2f}%, S1 Unrel Rej: {best_rec90['s1_unrel_rej']:.2f}%, Pipe Prec: {best_rec90['pipe_prec']:.2f}%, Pipe Recall: {best_rec90['pipe_recall']:.2f}%")

        # 3. Max Pipeline F1
        best_f1 = sweep_df.sort_values(by="pipe_f1", ascending=False).iloc[0]
        print(f"\n[Recommendation 3: Max Pipeline F1]")
        print(f"  tau* = {best_f1['tau']:.2f} -> Pipe F1: {best_f1['pipe_f1']:.2f}%, Pipe Prec: {best_f1['pipe_prec']:.2f}%, Pipe Recall: {best_f1['pipe_recall']:.2f}%")

        # 4. Max Pipeline F2 (Recall-biased)
        best_f2 = sweep_df.sort_values(by="pipe_f2", ascending=False).iloc[0]
        print(f"\n[Recommendation 4: Max Pipeline F2 (Recall-Biased)]")
        print(f"  tau* = {best_f2['tau']:.2f} -> Pipe F2: {best_f2['pipe_f2']:.2f}%, Pipe Prec: {best_f2['pipe_prec']:.2f}%, Pipe Recall: {best_f2['pipe_recall']:.2f}%")

    print("\n" + "=" * 110)

if __name__ == "__main__":
    calibrate_stage1()
