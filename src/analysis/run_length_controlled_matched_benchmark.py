"""
Length-Controlled Matched Benchmark Evaluation for V5 Perspective Verifier.

Evaluates 167 matched quadruplets across 4 perspective types:
  1. True Positive (Valid Ground-Truth Perspective)
  2. Opposite-Stance (Same Claim, Rebuttal/Opposite Polarity)
  3. Same-Stance Different-Reasoning (Same Claim, Same Stance, Different Perspective Cluster)
  4. Unrelated (Completely Different Claim)

At two length regimes:
  - Full Discourse Argument: Mean = 179.5 words (~180w)
  - Concise Argument: Mean = 65.3 words (~58w)

Operating Thresholds:
  - Optimal F1: theta* = 0.52
  - High-Precision: theta* = 0.83
  - Temperature Scaling: T* = 1.9104
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PairDataset(Dataset):
    def __init__(self, premises, hypotheses, tokenizer, max_length=256):
        self.encodings = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }

def evaluate_matched_quadruplets(
    dataset_path="data/audit/matched_quadruplets_167.csv",
    v5_model_dir="models/ensemble_data_only",
    output_csv="results/length_controlled_matched_benchmark.csv",
    output_json="results/length_controlled_matched_benchmark.json",
    batch_size=64,
    temperature=1.9104,
    theta_f1=0.52,
    theta_prec90=0.83,
    max_quads=None
):
    print("=" * 85)
    print("RUNNING LENGTH-CONTROLLED MATCHED QUADRUPLET BENCHMARK")
    print("=" * 85)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}")

    df = pd.read_csv(dataset_path)
    if max_quads is not None and max_quads < len(df):
        df = df.iloc[:max_quads].copy()
    print(f"Loaded {len(df)} matched quadruplets.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve model path
    cand_paths = [
        os.path.join(v5_model_dir, "seed_42"),
        v5_model_dir,
        "models/argument_nli_deberta_v3_v3",
        "cross-encoder/nli-deberta-v3-base"
    ]
    model_path = next((d for d in cand_paths if os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin")) or d.startswith("cross-encoder/")), cand_paths[0])
    print(f"Loading V5 Verifier from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Construct evaluation records
    types = [
        ("True Positive", "hyp_true_positive"),
        ("Opposite-Stance", "hyp_opposite_stance"),
        ("Same-Stance Diff-Reason", "hyp_same_stance_diff_reason"),
        ("Unrelated", "hyp_unrelated")
    ]

    records = []
    for length_regime, col_prem in [("Full (180w)", "full_premise_180w"), ("Concise (58w)", "concise_premise_58w")]:
        for ptype_label, col_hyp in types:
            for idx, r in df.iterrows():
                prem = str(r[col_prem]).strip()
                hyp = str(r[col_hyp]).strip()
                records.append({
                    "quad_id": r["quad_id"],
                    "claim_id": r["claim_id"],
                    "length_regime": length_regime,
                    "perspective_type": ptype_label,
                    "premise_words": len(prem.split()),
                    "hyp_words": len(hyp.split()),
                    "premise": prem,
                    "hypothesis": hyp
                })

    eval_df = pd.DataFrame(records)
    print(f"Total pairs to evaluate: {len(eval_df)} ({len(df)} quadruplets x 4 types x 2 lengths)")

    # Run batched inference
    premises = eval_df["premise"].tolist()
    hypotheses = eval_df["hypothesis"].tolist()
    ds = PairDataset(premises, hypotheses, tokenizer, max_length=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    print("Scoring pairs with V5 Cross-Encoder...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            scaled_logits = logits / max(temperature, 1e-4)
            probs = F.softmax(scaled_logits, dim=-1)
            # Entailment is index 1
            entail_idx = 1 if logits.shape[-1] >= 2 else 0
            all_probs.extend(probs[:, entail_idx].cpu().numpy().tolist())

    eval_df["p_entailment"] = [round(float(p), 4) for p in all_probs]
    eval_df["accepted_f1"] = eval_df["p_entailment"] >= theta_f1
    eval_df["accepted_prec90"] = eval_df["p_entailment"] >= theta_prec90

    # Aggregate results by Length Regime and Perspective Type
    summary_rows = []
    for length_regime in ["Full (180w)", "Concise (58w)"]:
        for ptype_label, _ in types:
            sub = eval_df[(eval_df["length_regime"] == length_regime) & (eval_df["perspective_type"] == ptype_label)]
            n = len(sub)
            mean_p = sub["p_entailment"].mean()
            median_p = sub["p_entailment"].median()
            acc_rate_f1 = sub["accepted_f1"].mean()
            acc_rate_prec90 = sub["accepted_prec90"].mean()
            rej_rate_f1 = 1.0 - acc_rate_f1

            summary_rows.append({
                "length_regime": length_regime,
                "perspective_type": ptype_label,
                "n_pairs": n,
                "mean_p_entail": round(float(mean_p), 4),
                "median_p_entail": round(float(median_p), 4),
                "accept_rate_f1": round(float(acc_rate_f1 * 100), 2),
                "rejection_rate_f1": round(float(rej_rate_f1 * 100), 2),
                "accept_rate_prec90": round(float(acc_rate_prec90 * 100), 2)
            })

    summary_df = pd.DataFrame(summary_rows)

    # Save outputs
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    eval_df.to_csv(output_csv, index=False)
    
    summary_json_path = output_json
    summary_dict = {
        "title": "Length-Controlled Matched Quadruplet Benchmark (167 Arguments x 4 Types x 2 Lengths)",
        "configuration": {
            "model_path": model_path,
            "temperature": temperature,
            "theta_f1": theta_f1,
            "theta_prec90": theta_prec90,
            "total_pairs": len(eval_df)
        },
        "summary": summary_rows
    }
    import json
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Print publication summary table
    print("\n" + "=" * 105)
    print("      LENGTH-CONTROLLED MATCHED QUADRUPLET AUDIT TABLE (V5 VERIFIER SPECIFICITY)")
    print("=" * 105)
    print(f"{'Length Regime':<16} | {'Perspective Type':<26} | {'Pairs':<6} | {'Mean P':<8} | {'Accept Rate (F1)':<18} | {'Reject Rate':<12} | {'Accept (Prec90)':<15}")
    print("-" * 105)
    for r in summary_rows:
        print(f"{r['length_regime']:<16} | {r['perspective_type']:<26} | {r['n_pairs']:<6} | {r['mean_p_entail']:<8.4f} | {r['accept_rate_f1']:>6.2f}%            | {r['rejection_rate_f1']:>6.2f}%     | {r['accept_rate_prec90']:>6.2f}%")
    print("=" * 105 + "\n")

    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate length-controlled matched benchmark.")
    parser.add_argument("--dataset_path", type=str, default="data/audit/matched_quadruplets_167.csv")
    parser.add_argument("--v5_model_dir", type=str, default="models/ensemble_data_only")
    parser.add_argument("--output_csv", type=str, default="results/length_controlled_matched_benchmark.csv")
    parser.add_argument("--output_json", type=str, default="results/length_controlled_matched_benchmark.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.9104)
    parser.add_argument("--max_quads", type=int, default=None)
    args = parser.parse_args()

    evaluate_matched_quadruplets(
        dataset_path=args.dataset_path,
        v5_model_dir=args.v5_model_dir,
        output_csv=args.output_csv,
        output_json=args.output_json,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_quads=args.max_quads
    )
