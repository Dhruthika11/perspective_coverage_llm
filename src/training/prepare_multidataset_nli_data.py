"""
Prepares multi-dataset training, validation, and test pairs for fine-tuning
Argument-NLI (Cross-Encoder DeBERTa-v3) with Approach A:
Trains directly on Realistic Long AI-Generated Candidate Viewpoints (Mistral, Qwen, Gemma)
paired against concise Gold Perspectives, eliminating length/discourse asymmetry.
"""
import os
import json
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

def prepare_multidataset_nli_data(
    output_dir="data/nli_training",
    total_target_train=6000,
    total_target_val=1000,
    total_target_test=1000
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Preparing Multi-Dataset Argument-NLI data (Approach A: Long AI Generations)...")

    # Load claims and perspectives
    claims_df = pd.read_csv("data/processed/claims.csv")
    persp_df = pd.read_csv("data/processed/perspectives.csv")
    print(f"Loaded {len(claims_df)} claims and {len(persp_df)} perspectives.")

    # Load generation files and merge with predicted stances
    models = ["mistral_7b_instruct_v0_3", "qwen2_5_32b_instruct", "gemma_3_12b_it"]
    gen_dfs = {}
    for m in models:
        gen_path = f"data/llm_outputs/{m}/processed_generations.csv"
        stance_path = f"results/{m}/metrics/llm_sample_stances.csv"
        if os.path.exists(gen_path) and os.path.exists(stance_path):
            gdf = pd.read_csv(gen_path)
            sdf = pd.read_csv(stance_path)
            merged = pd.merge(gdf, sdf, on=["claim_id", "sample_id", "viewpoint_id"], how="inner")
            gen_dfs[m] = merged
            print(f"Loaded {len(merged)} stance-aligned long generations for {m}")

    # 1. Strict Claim-Level Split (70% Train, 15% Val, 15% Test)
    all_cids = claims_df["claim_id"].unique().tolist()
    random.shuffle(all_cids)
    
    n_train = int(len(all_cids) * 0.70)
    n_val = int(len(all_cids) * 0.15)
    
    train_cids = set(all_cids[:n_train])
    val_cids = set(all_cids[n_train:n_train + n_val])
    test_cids = set(all_cids[n_train + n_val:])
    
    print(f"Claim Split: Train={len(train_cids)}, Val={len(val_cids)}, Test={len(test_cids)}")

    # Helper to clean and format AI text
    def clean_ai_text(txt):
        txt = str(txt).strip()
        # Remove markdown heading noise if present
        for prefix in ["**Ethical Viewpoint:**", "**Economic Viewpoint:**", "**Social Viewpoint:**", "**Political Viewpoint:**", "**Pragmatic Viewpoint:**"]:
            if txt.startswith(prefix):
                txt = txt[len(prefix):].strip()
        return txt

    # Mine pairs for a given split
    def mine_pairs_for_claims(cids, target_pairs, split_name):
        cids_list = list(cids)
        claim_map = dict(zip(claims_df["claim_id"], claims_df["claim_text"]))
        persp_sub = persp_df[persp_df["claim_id"].isin(cids)]
        
        # Group human perspectives by (claim_id, stance)
        grouped_persps = {}
        for _, r in persp_sub.iterrows():
            cid = r["claim_id"]
            st = str(r["stance"]).lower()
            txt = str(r["text"]).strip()
            if not txt:
                continue
            if cid not in grouped_persps:
                grouped_persps[cid] = {"support": [], "undermine": []}
            if "sup" in st:
                grouped_persps[cid]["support"].append(txt)
            else:
                grouped_persps[cid]["undermine"].append(txt)

        # Targets per class: 40% Entailment, 30% Contradiction, 30% Neutral
        n_entail = int(target_pairs * 0.40)
        n_contra = int(target_pairs * 0.30)
        n_neutral = target_pairs - n_entail - n_contra
        
        print(f"Mining {split_name}: Target Entailment={n_entail}, Contradiction={n_contra}, Neutral={n_neutral}")
        
        entail_pairs = []
        contra_pairs = []
        neutral_pairs = []

        # 1. ENTAILMENT & CONTRADICTION using Long AI Generations
        for cid in cids_list:
            c_text = claim_map.get(cid, "")
            human_sups = grouped_persps.get(cid, {}).get("support", [])
            human_unds = grouped_persps.get(cid, {}).get("undermine", [])

            for m_name, g_df in gen_dfs.items():
                m_gens = g_df[g_df["claim_id"] == cid]
                if m_gens.empty:
                    continue

                for _, grow in m_gens.iterrows():
                    ai_text = clean_ai_text(grow.get("combined_text", grow.get("perspective_summary", "")))
                    ai_stance = str(grow.get("predicted_stance", "")).lower()
                    
                    # Ensure realistic candidate length (at least 20 words)
                    if len(ai_text.split()) < 20:
                        continue

                    # ENTAILMENT: Match with human perspective of SAME stance
                    matching_humans = human_sups if "sup" in ai_stance else human_unds
                    for htxt in matching_humans[:2]:
                        entail_pairs.append({
                            "claim_text": c_text,
                            "text_a": htxt,           # Short gold perspective (~12 words)
                            "text_b": ai_text[:600],  # Long AI candidate (~70 words)
                            "label": "entailment",
                            "label_id": 1,
                            "pair_category": f"llm_human_entailment_{m_name}",
                            "dataset_source": "perspectrum"
                        })

                    # CONTRADICTION: Match with human perspective of OPPOSITE stance
                    opp_humans = human_unds if "sup" in ai_stance else human_sups
                    for htxt in opp_humans[:2]:
                        contra_pairs.append({
                            "claim_text": c_text,
                            "text_a": htxt,           # Short gold perspective (~12 words)
                            "text_b": ai_text[:600],  # Long AI candidate (~70 words)
                            "label": "contradiction",
                            "label_id": 0,
                            "pair_category": f"llm_human_opposite_{m_name}",
                            "dataset_source": "perspectrum"
                        })

        # 2. NEUTRAL: Short Gold vs Long AI on completely DIFFERENT claims
        for i in range(len(cids_list) - 1):
            c1 = cids_list[i]
            c2 = cids_list[i+1]
            p1_list = grouped_persps.get(c1, {}).get("support", []) + grouped_persps.get(c1, {}).get("undermine", [])
            
            # Pick AI generation from c2
            for m_name, g_df in gen_dfs.items():
                m_gens2 = g_df[g_df["claim_id"] == c2]
                if not m_gens2.empty and p1_list:
                    ai_cand = clean_ai_text(m_gens2.iloc[0].get("combined_text", ""))
                    if len(ai_cand.split()) >= 20:
                        neutral_pairs.append({
                            "claim_text": claim_map.get(c1, ""),
                            "text_a": p1_list[0],        # Short gold perspective (~12 words)
                            "text_b": ai_cand[:600],     # Long AI candidate (~70 words)
                            "label": "neutral",
                            "label_id": 2,
                            "pair_category": f"llm_human_cross_unrelated_{m_name}",
                            "dataset_source": "perspectrum"
                        })

        # Include benchmark calibration pool from Kialo & IBM Debater
        benchmark_pairs = []
        if os.path.exists("data/calibration/universal_calibration_pairs.csv"):
            uni_df = pd.read_csv("data/calibration/universal_calibration_pairs.csv")
            for _, r in uni_df.iterrows():
                ds = r.get("dataset", "debater")
                lbl = r.get("human_label", 0)
                nli_lbl = "entailment" if lbl == 1 else "contradiction"
                benchmark_pairs.append({
                    "claim_text": r.get("claim_text", ""),
                    "text_a": r.get("human_perspective", r.get("human_text", "")),
                    "text_b": clean_ai_text(r.get("llm_viewpoint", r.get("llm_text", "")))[:600],
                    "label": nli_lbl,
                    "label_id": 1 if nli_lbl == "entailment" else 0,
                    "pair_category": f"benchmark_{ds}",
                    "dataset_source": ds
                })

        random.shuffle(entail_pairs)
        random.shuffle(contra_pairs)
        random.shuffle(neutral_pairs)
        
        print(f"Mined pool: Entail={len(entail_pairs)}, Contra={len(contra_pairs)}, Neutral={len(neutral_pairs)}")

        selected = (
            entail_pairs[:n_entail] +
            contra_pairs[:n_contra] +
            neutral_pairs[:n_neutral] +
            benchmark_pairs[:200]
        )
        random.shuffle(selected)
        return pd.DataFrame(selected)

    train_df = mine_pairs_for_claims(train_cids, total_target_train, "Train")
    val_df = mine_pairs_for_claims(val_cids, total_target_val, "Val")
    test_df = mine_pairs_for_claims(test_cids, total_target_test, "Test")

    train_path = os.path.join(output_dir, "train_nli_pairs.csv")
    val_path = os.path.join(output_dir, "val_nli_pairs.csv")
    test_path = os.path.join(output_dir, "test_nli_pairs.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nApproach A Data Preparation Complete!")
    print(f"  Train: {len(train_df)} pairs saved to {train_path}")
    print(f"  Val:   {len(val_df)} pairs saved to {val_path}")
    print(f"  Test:  {len(test_df)} pairs saved to {test_path}")

    print("\nTrain Label Distribution:")
    print(train_df["label"].value_counts())
    print("\nTrain Word Length Statistics:")
    print("  Premise (text_a) mean words:", round(train_df["text_a"].str.split().str.len().mean(), 1))
    print("  Hypothesis (text_b) mean words:", round(train_df["text_b"].str.split().str.len().mean(), 1))
    print("\nPair Categories in Train:")
    print(train_df["pair_category"].value_counts().head(10))

if __name__ == "__main__":
    prepare_multidataset_nli_data()
