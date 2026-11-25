"""
Compute Detailed Precision Profile at Frozen Stage 1 85% Recall Threshold.

Evaluates Precision under two definitions:
  1. Strict Precision: TP / (TP + Opposite + SSDR + Unrelated)
     (Only exact matching perspective cluster counts as correct)
  2. Stance-Consistent Precision: (TP + SSDR) / (TP + Opposite + SSDR + Unrelated)
     (Any valid perspective supporting the claim counts as correct;
      only Opposite rebuttals and Unrelated noise are considered errors)

Sweeps Stage 2 Verifier Threshold theta in [0.50, 0.90] to map the Precision-Recall Frontier.
"""

import os
import sys
import numpy as np
import pandas as pd

def compute_precision_profile(pred_path="results/twostage_matched_benchmark_predictions.csv"):
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} not found.")
        return

    df = pd.read_csv(pred_path)
    print("=" * 115)
    print("FROZEN 85% RECALL THRESHOLD: DETAILED PRECISION & VERIFIER SWEEP AUDIT")
    print("=" * 115)

    # We evaluate both Concise-calibrated (0.5943) and Global-calibrated (0.5786)
    frozen_configs = [
        ("Concise-Specific tau* = 0.5943", 0.5943),
        ("Unified Global tau* = 0.5786", 0.5786)
    ]

    for label, tau_star in frozen_configs:
        print(f"\n{'='*45} {label} {'='*45}")

        for regime in ["Concise (58w)", "Full (180w)"]:
            sub = df[df["length_regime"] == regime].copy()
            
            # Apply frozen Stage 1 threshold
            s1_passed = sub["stage1_cosine"] >= tau_star
            sub_s1 = sub[s1_passed].copy()

            n_tp_total = 154
            n_opp_total = 154
            n_ssdr_total = 154
            n_unrel_total = 154

            print(f"\n--- Regime: {regime} (Stage 1 tau* = {tau_star:.4f}) ---")
            print(f"Candidates Passing Stage 1:")
            tp_s1 = (sub_s1["perspective_type"] == "True Positive").sum()
            opp_s1 = (sub_s1["perspective_type"] == "Opposite-Stance").sum()
            ssdr_s1 = (sub_s1["perspective_type"] == "Same-Stance Diff-Reason").sum()
            unrel_s1 = (sub_s1["perspective_type"] == "Unrelated").sum()
            print(f"  * True Positive:           {tp_s1}/154 ({tp_s1/n_tp_total*100:.2f}%)")
            print(f"  * Opposite-Stance:         {opp_s1}/154 ({opp_s1/n_opp_total*100:.2f}%)")
            print(f"  * Same-Stance Diff-Reason: {ssdr_s1}/154 ({ssdr_s1/n_ssdr_total*100:.2f}%)")
            print(f"  * Unrelated:               {unrel_s1}/154 ({unrel_s1/n_unrel_total*100:.2f}%)")

            # Sweep Stage 2 Verifier Threshold theta
            sweep_thetas = [0.50, 0.52, 0.60, 0.70, 0.75, 0.80, 0.83, 0.85, 0.88, 0.90]
            print(f"\nStage 2 Verifier (theta) Precision & Recall Tradeoff Table:")
            hdr = f"{'theta':<7} | {'TP':<4} | {'Opp':<4} | {'SSDR':<4} | {'Unrel':<5} | {'Total':<5} | {'Strict Prec':<11} | {'Stance Prec':<11} | {'TP Recall':<10} | {'F1 (Strict)':<11}"
            print(hdr)
            print("-" * 105)

            for theta in sweep_thetas:
                v_pass = sub_s1["p_entail"] >= theta
                
                tp_acc = (v_pass & (sub_s1["perspective_type"] == "True Positive")).sum()
                opp_acc = (v_pass & (sub_s1["perspective_type"] == "Opposite-Stance")).sum()
                ssdr_acc = (v_pass & (sub_s1["perspective_type"] == "Same-Stance Diff-Reason")).sum()
                unrel_acc = (v_pass & (sub_s1["perspective_type"] == "Unrelated")).sum()
                tot_acc = tp_acc + opp_acc + ssdr_acc + unrel_acc

                strict_prec = (tp_acc / tot_acc * 100) if tot_acc > 0 else 0.0
                stance_prec = ((tp_acc + ssdr_acc) / tot_acc * 100) if tot_acc > 0 else 0.0
                tp_recall = (tp_acc / n_tp_total * 100)
                
                p_f = strict_prec / 100.0
                r_f = tp_recall / 100.0
                f1_strict = (2 * p_f * r_f / (p_f + r_f) * 100) if (p_f + r_f) > 0 else 0.0

                note = ""
                if theta == 0.52:
                    note = " <- Optimal F1"
                elif theta == 0.83:
                    note = " <- High Prec (P90)"

                row = f"{theta:<7.2f} | {tp_acc:<4} | {opp_acc:<4} | {ssdr_acc:<4} | {unrel_acc:<5} | {tot_acc:<5} | {strict_prec:>10.2f}% | {stance_prec:>10.2f}% | {tp_recall:>9.2f}% | {f1_strict:>10.2f}%{note}"
                print(row)

if __name__ == "__main__":
    compute_precision_profile()
