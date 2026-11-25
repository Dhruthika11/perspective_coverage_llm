"""
Computes True Two-Stage Perspective Coverage across all 12 models on GPU:
  - Qwen 2.5 Family: 0.5B, 1.5B, 3B, 7B, 14B, 32B
  - Mistral Family: 7B
  - Meta LLaMA Family: 1B, 3B, 8B
  - Google Gemma Family: 2B, 9B, 12B

Pipeline:
  Stage 1: Cosine Similarity >= 0.75
  Stage 2: Calibrated Argument-NLI (p_entail >= 0.80, argmax == 1)

Generates:
  1. Per-perspective coverage flags (Stage 1 vs Stage 2)
  2. Per-claim coverage scores (Overall, Support, Undermine)
  3. Master Cross-Model Comparison Table across all 12 models
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Two-Stage Coverage Computation on device: {device}")

from src.analysis.similarity_matrix import load_embeddings, cosine_sim_matrix

# Comprehensive 12+ Model Scaling Ladder Definition
ALL_MODELS = [
    # --- Qwen 2.5 Scaling Ladder ---
    {"key": "qwen2_5_0_5b_instruct", "name": "Qwen2.5-0.5B", "params_b": 0.5, "family": "Qwen"},
    {"key": "qwen2_5_1_5b_instruct", "name": "Qwen2.5-1.5B", "params_b": 1.5, "family": "Qwen"},
    {"key": "qwen2_5_3b_instruct",   "name": "Qwen2.5-3B",   "params_b": 3.0, "family": "Qwen"},
    {"key": "qwen2_5_7b_instruct",   "name": "Qwen2.5-7B",   "params_b": 7.0, "family": "Qwen"},
    {"key": "qwen2_5_14b_instruct",  "name": "Qwen2.5-14B",  "params_b": 14.0, "family": "Qwen"},
    {"key": "qwen2_5_32b_instruct",  "name": "Qwen2.5-32B",  "params_b": 32.0, "family": "Qwen"},
    # --- Mistral Family ---
    {"key": "mistral_7b_instruct_v0_3", "name": "Mistral-7B", "params_b": 7.0, "family": "Mistral"},
    # --- Meta LLaMA 3.x Family ---
    {"key": "llama_3_2_1b_instruct", "name": "Llama-3.2-1B", "params_b": 1.0, "family": "Llama"},
    {"key": "llama_3_2_3b_instruct", "name": "Llama-3.2-3B", "params_b": 3.0, "family": "Llama"},
    {"key": "llama_3_1_8b_instruct", "name": "Llama-3.1-8B", "params_b": 8.0, "family": "Llama"},
    # --- Google Gemma Family ---
    {"key": "gemma_2_2b_it",        "name": "Gemma-2-2B",   "params_b": 2.0, "family": "Gemma"},
    {"key": "gemma_2_9b_it",        "name": "Gemma-2-9B",   "params_b": 9.0, "family": "Gemma"},
    {"key": "gemma_3_12b_it",       "name": "Gemma-3-12B",  "params_b": 12.0, "family": "Gemma"},
]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def compute_all_models_two_stage_coverage(
    model_checkpoint="models/argument_nli_deberta_v3",
    output_dir="results",
    batch_size=64
):
    print("=" * 95)
    print("     TRUE TWO-STAGE PERSPECTIVE COVERAGE COMPUTATION (ALL 12 MODELS)")
    print("=" * 95)

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    # Load Stage 2 Fine-Tuned DeBERTa-v3 Model
    print(f"Loading Stage 2 DeBERTa-v3 model from {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)
    model.eval()

    # Load Perspective Metadata & Text
    persps_df = pd.read_csv("data/processed/perspectives.csv")
    persp_map = dict(zip(persps_df["perspective_id"], persps_df["text"]))
    print(f"Loaded {len(persps_df)} gold perspectives across {persps_df['claim_id'].nunique()} claims.")

    all_model_summaries = []
    processed_count = 0
    skipped_count = 0

    for m_info in ALL_MODELS:
        model_key = m_info["key"]
        display_name = m_info["name"]
        params_b = m_info["params_b"]
        family = m_info["family"]

        emb_dir = f"results/{model_key}/embeddings"
        gen_file = f"data/llm_outputs/{model_key}/processed_generations.csv"

        # Check if embeddings and generations exist
        has_emb = os.path.exists(emb_dir) and os.path.exists(os.path.join(emb_dir, "perspective_embeddings.npy")) and os.path.exists(os.path.join(emb_dir, "llm_output_embeddings.npy"))
        has_gen = os.path.exists(gen_file)

        if not has_emb or not has_gen:
            # Check alternative naming if any
            print(f"\n[-] Model [{display_name}] ({model_key}): Embeddings or generations not found -> Skipped.")
            skipped_count += 1
            continue

        processed_count += 1
        print(f"\n[+] Processing Model [{processed_count}]: {display_name} ({family}, {params_b}B params)")

        # Load Embeddings & Indices
        cfg = {"paths": {"embeddings_dir": emb_dir}}
        p_emb, p_idx, l_emb, l_idx = load_embeddings(cfg)

        gens_df = pd.read_csv(gen_file)
        gen_map = dict(zip(zip(gens_df["claim_id"], gens_df["sample_id"]), gens_df["combined_text"]))

        claim_ids = sorted(set(p_idx["claim_id"]).intersection(set(l_idx["claim_id"])))
        print(f"    Claims in common: {len(claim_ids)}")

        # 1. Collect all Candidate Matches passing Stage 1 (tau >= 0.75)
        candidate_records = []
        all_perspective_records = []

        start_time = time.time()
        for cid in claim_ids:
            p_mask = (p_idx["claim_id"] == cid).values
            l_mask = (l_idx["claim_id"] == cid).values

            p_emb_claim = p_emb[p_mask]
            l_emb_claim = l_emb[l_mask]
            p_idx_claim = p_idx[p_mask].reset_index(drop=True)
            l_idx_claim = l_idx[l_mask].reset_index(drop=True)

            sim = cosine_sim_matrix(l_emb_claim, p_emb_claim)  # (n_llm, n_persp)
            n_persp = sim.shape[1]

            for j in range(n_persp):
                pid = int(p_idx_claim.loc[j, "perspective_id"])
                stance = str(p_idx_claim.loc[j, "stance"]).lower()
                max_sim = float(sim[:, j].max())
                best_i = int(sim[:, j].argmax())

                stage1_pass = bool(max_sim >= 0.75)

                persp_rec = {
                    "claim_id": cid,
                    "perspective_id": pid,
                    "stance": stance,
                    "best_similarity": max_sim,
                    "stage1_covered": stage1_pass,
                    "stage2_pass": False,
                    "stage2_covered": False,
                    "p_entail": 0.0,
                    "p_contra": 0.0,
                    "p_neutral": 0.0
                }

                if stage1_pass:
                    p_text = persp_map.get(pid, "")
                    s_id = int(l_idx_claim.loc[best_i, "sample_id"])
                    g_text = gen_map.get((cid, s_id), "")
                    candidate_records.append({
                        "rec_idx": len(all_perspective_records),
                        "premise": p_text,
                        "hypothesis": g_text
                    })

                all_perspective_records.append(persp_rec)

        print(f"    Stage 1 Candidates (tau >= 0.75): {len(candidate_records)} / {len(all_perspective_records)} ({len(candidate_records)/len(all_perspective_records):.2%})")

        # 2. Run Stage 2 Inference on Candidate Matches
        if candidate_records:
            print(f"    Running Stage 2 DeBERTa Inference on {len(candidate_records)} candidates in batches of {batch_size}...")
            for b_i in range(0, len(candidate_records), batch_size):
                b_cands = candidate_records[b_i : b_i + batch_size]
                premises = [c["premise"] for c in b_cands]
                hypotheses = [c["hypothesis"] for c in b_cands]

                inputs = tokenizer(
                    premises,
                    hypotheses,
                    truncation=True,
                    max_length=256,
                    padding=True,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                for k, c in enumerate(b_cands):
                    rec_idx = c["rec_idx"]
                    p_contra, p_entail, p_neutral = probs[k]
                    pred = preds[k]

                    # Calibrated Strategy 1 Decision Rule: Argmax is Entailment AND p_entail >= 0.80
                    stage2_pass = bool((pred == 1) and (p_entail >= 0.80))

                    all_perspective_records[rec_idx]["stage2_pass"] = stage2_pass
                    all_perspective_records[rec_idx]["stage2_covered"] = stage2_pass
                    all_perspective_records[rec_idx]["p_entail"] = float(p_entail)
                    all_perspective_records[rec_idx]["p_contra"] = float(p_contra)
                    all_perspective_records[rec_idx]["p_neutral"] = float(p_neutral)

        elapsed = time.time() - start_time
        print(f"    Inference completed in {elapsed:.1f}s")

        # 3. Aggregate Coverage Scores per Claim
        persp_res_df = pd.DataFrame(all_perspective_records)

        claim_metrics = []
        for cid, grp in persp_res_df.groupby("claim_id"):
            n_tot = len(grp)
            n_st1 = grp["stage1_covered"].sum()
            n_st2 = grp["stage2_covered"].sum()

            cov_st1 = n_st1 / n_tot if n_tot > 0 else 0.0
            cov_st2 = n_st2 / n_tot if n_tot > 0 else 0.0

            sup_grp = grp[grp["stance"].str.contains("sup", na=False)]
            und_grp = grp[grp["stance"].str.contains("und", na=False)]

            sup_cov_st1 = sup_grp["stage1_covered"].mean() if len(sup_grp) > 0 else np.nan
            sup_cov_st2 = sup_grp["stage2_covered"].mean() if len(sup_grp) > 0 else np.nan

            und_cov_st1 = und_grp["stage1_covered"].mean() if len(und_grp) > 0 else np.nan
            und_cov_st2 = und_grp["stage2_covered"].mean() if len(und_grp) > 0 else np.nan

            claim_metrics.append({
                "claim_id": cid,
                "n_perspectives": n_tot,
                "stage1_covered": n_st1,
                "stage2_covered": n_st2,
                "stage1_coverage_score": cov_st1,
                "stage2_coverage_score": cov_st2,
                "support_coverage_stage1": sup_cov_st1,
                "support_coverage_stage2": sup_cov_st2,
                "undermine_coverage_stage1": und_cov_st1,
                "undermine_coverage_stage2": und_cov_st2
            })

        claim_res_df = pd.DataFrame(claim_metrics)

        # 4. Save Model-Level Coverage Files
        m_metrics_dir = f"results/{model_key}/metrics"
        os.makedirs(m_metrics_dir, exist_ok=True)
        persp_res_df.to_csv(os.path.join(m_metrics_dir, "two_stage_per_perspective_coverage.csv"), index=False)
        claim_res_df.to_csv(os.path.join(m_metrics_dir, "two_stage_per_claim_coverage.csv"), index=False)

        # 5. Model Summary Stats
        mean_st1 = claim_res_df["stage1_coverage_score"].mean()
        mean_st2 = claim_res_df["stage2_coverage_score"].mean()
        sup_st2 = claim_res_df["support_coverage_stage2"].mean()
        und_st2 = claim_res_df["undermine_coverage_stage2"].mean()

        overestimation_drop = (mean_st1 - mean_st2)
        drop_pct = (overestimation_drop / mean_st1) * 100 if mean_st1 > 0 else 0.0

        n_zero_cov = int((claim_res_df["stage2_coverage_score"] == 0.0).sum())
        n_full_cov = int((claim_res_df["stage2_coverage_score"] == 1.0).sum())

        all_model_summaries.append({
            "Model": display_name,
            "Family": family,
            "Params_B": params_b,
            "Model_Key": model_key,
            "SingleStage_Cosine (tau=0.75)": round(mean_st1, 4),
            "True_TwoStage_Coverage": round(mean_st2, 4),
            "False_Match_Drop": round(overestimation_drop, 4),
            "Drop_Pct": f"{drop_pct:.1f}%",
            "Support_Coverage": round(sup_st2, 4),
            "Undermine_Coverage": round(und_st2, 4),
            "Stance_Gap (Sup-Und)": round(sup_st2 - und_st2, 4),
            "Zero_Coverage_Claims": n_zero_cov,
            "Total_Claims": len(claim_res_df)
        })

    print(f"\nProcessing Complete! Processed: {processed_count} models, Skipped: {skipped_count} models.")

    if all_model_summaries:
        summary_df = pd.DataFrame(all_model_summaries)
        # Sort by Family then by Params_B
        summary_df = summary_df.sort_values(by=["Family", "Params_B"])

        print("\n" + "=" * 110)
        print("          CROSS-MODEL COMPARISON ACROSS ALL EVALUATED MODELS (TWO-STAGE)")
        print("=" * 110)
        cols = ["Model", "Family", "Params_B", "SingleStage_Cosine (tau=0.75)", "True_TwoStage_Coverage", "Drop_Pct", "Support_Coverage", "Undermine_Coverage", "Stance_Gap (Sup-Und)", "Zero_Coverage_Claims"]
        print(summary_df[cols].to_string(index=False))
        print("=" * 110)

        # Save Cross-Model Comparisons with NpEncoder
        summary_df.to_csv("results/two_stage_cross_model_comparison.csv", index=False)
        with open("results/two_stage_cross_model_comparison.json", "w", encoding="utf-8") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2, cls=NpEncoder)

        print("\nSaved cross-model two-stage evaluation to:")
        print("  - results/two_stage_cross_model_comparison.csv")
        print("  - results/two_stage_cross_model_comparison.json")
    else:
        print("\nNo models were evaluated.")

if __name__ == "__main__":
    compute_all_models_two_stage_coverage()
