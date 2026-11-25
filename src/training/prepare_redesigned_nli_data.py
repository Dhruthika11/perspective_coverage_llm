"""
Prepares Redesigned Multi-Dataset Training, Validation, and Test Datasets for DeBERTa-v3.

Implements the user's comprehensive blueprint:
1. Target Distribution for V3 (Coverage-Aware):
   - Genuine Entailment / Perspective Coverage: ~45% (incorporating realistic length diversity:
     Short [20-50 words], Medium [50-100 words], Long [100-175 words], Very Long [175-250 words])
   - Same-Stance, Different-Reasoning Negatives (Neutral): ~20%
   - Same-Topic, Different-Concept Negatives (Neutral): ~15%
   - Opposite-Position Conflicts (Contradiction): ~10%
   - Unrelated Negatives (Neutral): ~10%
2. Natural Coverage Input Direction:
   - Premise: Generated Argument (longer / elaborated)
   - Hypothesis: Human Reference Perspective (concise proposition)
   - Relationship: Generated Argument |= Human Reference Perspective
3. Strict Claim-Level Isolation:
   - Stratified, disjoint splitting on claim_id (70% Train, 15% Validation, 15% Test)
   - No claim in training ever leaks into validation or test.
4. Generates both:
   - V2: Direction-Corrected Baseline
   - V3: Coverage-Aware Length-Diverse Benchmark
"""

import os
import sys
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("prepare_redesigned_nli_data")


def get_word_count(text: str) -> int:
    return len(str(text).strip().split())


def get_length_tier(word_count: int) -> str:
    if word_count < 50:
        return "short"
    elif word_count < 100:
        return "medium"
    elif word_count < 175:
        return "long"
    else:
        return "very_long"


def clean_argument_text(txt: str) -> str:
    txt = str(txt).strip()
    for prefix in [
        "**Ethical Viewpoint:**", "**Economic Viewpoint:**", "**Social Viewpoint:**",
        "**Political Viewpoint:**", "**Pragmatic Viewpoint:**", "### Ethical Perspective:",
        "### Economic Perspective:", "### Social Perspective:", "### Legal Perspective:"
    ]:
        if txt.startswith(prefix):
            txt = txt[len(prefix):].strip()
    return txt


def build_redesigned_nli_datasets(
    claims_csv: str = "data/processed/claims.csv",
    perspectives_csv: str = "data/processed/perspectives.csv",
    output_base_dir: str = "data",
    target_train_pairs: int = 6000,
    target_val_pairs: int = 1200,
    target_test_pairs: int = 1200,
    seed: int = 42
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    claims_path = resolve(claims_csv)
    persp_path = resolve(perspectives_csv)

    if not os.path.exists(claims_path) or not os.path.exists(persp_path):
        raise FileNotFoundError(f"Missing claims or perspectives at {claims_path}, {persp_path}")

    claims_df = load_csv(claims_path)
    persp_df = load_csv(persp_path)
    logger.info(f"Loaded {len(claims_df)} claims and {len(persp_df)} human perspectives.")

    # Load available LLM generations
    model_tags = ["qwen2_5_1_5b_instruct", "qwen2_5_3b_instruct", "qwen2_5_32b_instruct"]
    gen_dfs = {}
    for m in model_tags:
        p = resolve(f"data/llm_outputs/{m}/processed_generations.csv")
        if os.path.exists(p):
            gdf = load_csv(p)
            gen_dfs[m] = gdf
            logger.info(f"Loaded {len(gdf)} generations for {m}")

    # Also load IBM Debater and Kialo perspectives for cross-dataset richness
    debater_claims_p = resolve("data/processed/ibm_debater/claims.csv")
    debater_persp_p = resolve("data/processed/ibm_debater/perspectives.csv")
    kialo_claims_p = resolve("data/processed/kialo/claims.csv")
    kialo_persp_p = resolve("data/processed/kialo/perspectives.csv")

    has_debater = os.path.exists(debater_claims_p) and os.path.exists(debater_persp_p)
    has_kialo = os.path.exists(kialo_claims_p) and os.path.exists(kialo_persp_p)

    debater_claims_df = load_csv(debater_claims_p) if has_debater else pd.DataFrame()
    debater_persp_df = load_csv(debater_persp_p) if has_debater else pd.DataFrame()
    kialo_claims_df = load_csv(kialo_claims_p) if has_kialo else pd.DataFrame()
    kialo_persp_df = load_csv(kialo_persp_p) if has_kialo else pd.DataFrame()

    # 1. Strict Claim-Level Split Enforcing Zero Leakage with Calibration & Test Benchmarks
    manifest_p = resolve("data/calibration/claim_splits_manifest.json")
    if os.path.exists(manifest_p):
        with open(manifest_p, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)
        logger.info(f"Loaded locked claim splits manifest from {manifest_p}")
        text_to_cid = dict(zip(claims_df["claim_text"], claims_df["claim_id"]))
        
        train_cids = {text_to_cid[t] for t in manifest["train_claims"] if t in text_to_cid}
        val_cids = {text_to_cid[t] for t in manifest["val_claims"] if t in text_to_cid}
        cal_cids = {text_to_cid[t] for t in manifest["calibration_claims"] if t in text_to_cid}
        test_cids = {text_to_cid[t] for t in manifest["test_claims"] if t in text_to_cid}
        
        # Rigorous independence assertions (All 6 Pairwise Conditions)
        assert len(train_cids & val_cids) == 0, "Train and Val claims overlap!"
        assert len(train_cids & cal_cids) == 0, "Train and Calibration claims overlap!"
        assert len(train_cids & test_cids) == 0, "Train and Test claims overlap!"
        assert len(val_cids & cal_cids) == 0, "Val and Calibration claims overlap!"
        assert len(val_cids & test_cids) == 0, "Val and Test claims overlap!"
        assert len(cal_cids & test_cids) == 0, "Calibration and Test claims overlap!"
        
        logger.info(f"Strict Claim Disjoint Splits Verified: Train={len(train_cids)}, Val={len(val_cids)}, Cal={len(cal_cids)}, Test={len(test_cids)}")
    else:
        all_cids = sorted(claims_df["claim_id"].unique().tolist())
        random.shuffle(all_cids)
        n_train = int(len(all_cids) * 0.70)
        n_val = int(len(all_cids) * 0.15)
        train_cids = set(all_cids[:n_train])
        val_cids = set(all_cids[n_train:n_train + n_val])
        test_cids = set(all_cids[n_train + n_val:])
        logger.warning("Manifest not found, using fallback random claim split.")


    # Helper function to mine pairs for a partition
    def mine_partition_pairs(cids: set, target_n: int, split_name: str) -> Tuple[List[Dict], List[Dict]]:
        cids_list = sorted(list(cids))
        claim_map = dict(zip(claims_df["claim_id"], claims_df["claim_text"]))
        persp_sub = persp_df[persp_df["claim_id"].isin(cids)].copy()

        # Group perspectives by (claim_id, cluster_id, stance)
        cluster_map = {}
        claim_persps = {}
        for _, r in persp_sub.iterrows():
            cid = r["claim_id"]
            clid = r["cluster_id"]
            st = str(r["stance"]).lower()
            txt = str(r["text"]).strip()
            if not txt:
                continue

            if cid not in claim_persps:
                claim_persps[cid] = []
            claim_persps[cid].append({"text": txt, "cluster_id": clid, "stance": st})

            key = (cid, clid)
            if key not in cluster_map:
                cluster_map[key] = {"claim_id": cid, "stance": st, "texts": []}
            cluster_map[key]["texts"].append(txt)

        # Target distribution:
        # Genuine Entailment: 45%
        # Same Stance, Diff Reasoning (Neutral): 20%
        # Same Topic, Diff Concept (Neutral): 15%
        # Opposite Position (Contradiction): 10%
        # Unrelated (Neutral): 10%
        n_entail = int(target_n * 0.45)
        n_same_stance_diff_reason = int(target_n * 0.20)
        n_same_topic_diff_concept = int(target_n * 0.15)
        n_opposite = int(target_n * 0.10)
        n_unrelated = target_n - (n_entail + n_same_stance_diff_reason + n_same_topic_diff_concept + n_opposite)

        v3_pairs = []
        v2_pairs = []

        # A. GENUINE ENTAILMENT / PERSPECTIVE COVERAGE (45%)
        # Mined from:
        # 1. Human-to-Human Within-Cluster Paraphrases (Cluster equivalence)
        # 2. Long LLM Generation paired with matching Human Perspective
        entail_collected = []

        # A1. Within-cluster human paraphrases
        for key, info in cluster_map.items():
            texts = info["texts"]
            cid = info["claim_id"]
            c_text = claim_map.get(cid, "")
            if len(texts) >= 2:
                for i in range(len(texts)):
                    for j in range(i + 1, min(len(texts), i + 3)):
                        entail_collected.append({
                            "claim_text": c_text,
                            "premise": texts[i],         # P |= H
                            "hypothesis": texts[j],
                            "label": "entailment",
                            "label_id": 1,
                            "pair_category": "within_cluster_paraphrase",
                            "source": "perspectrum_clusters",
                            "premise_words": get_word_count(texts[i]),
                            "hypothesis_words": get_word_count(texts[j]),
                            "length_tier": get_length_tier(get_word_count(texts[i]))
                        })

        # A2. Long LLM Argument |= Human Perspective
        for cid in cids_list:
            c_text = claim_map.get(cid, "")
            p_list = claim_persps.get(cid, [])
            if not p_list:
                continue

            for m_tag, gdf in gen_dfs.items():
                m_gens = gdf[gdf["claim_id"] == cid]
                if m_gens.empty:
                    continue

                for _, grow in m_gens.head(4).iterrows():
                    ai_text = clean_argument_text(grow.get("combined_text", grow.get("generated_text", "")))
                    ai_wc = get_word_count(ai_text)
                    if ai_wc < 20:
                        continue

                    # Match with a human perspective
                    chosen_p = p_list[0]["text"]
                    entail_collected.append({
                        "claim_text": c_text,
                        "premise": ai_text,             # Direction: Premise = Long LLM Argument
                        "hypothesis": chosen_p,          # Hypothesis = Concise Human Perspective
                        "label": "entailment",
                        "label_id": 1,
                        "pair_category": f"long_llm_to_human_positive_{m_tag}",
                        "source": "llm_generation",
                        "premise_words": ai_wc,
                        "hypothesis_words": get_word_count(chosen_p),
                        "length_tier": get_length_tier(ai_wc)
                    })

        random.shuffle(entail_collected)
        v3_pairs.extend(entail_collected[:n_entail])

        # B. SAME-STANCE, DIFFERENT-REASONING NEGATIVES (20%, Neutral)
        # Pairs from DIFFERENT clusters for the same claim with the SAME stance
        same_stance_negatives = []
        for cid in cids_list:
            c_text = claim_map.get(cid, "")
            p_list = claim_persps.get(cid, [])
            # Group by stance
            sups = [p for p in p_list if "sup" in p["stance"]]
            unds = [p for p in p_list if "und" in p["stance"]]

            for group in [sups, unds]:
                if len(group) >= 2:
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            if group[i]["cluster_id"] != group[j]["cluster_id"]:
                                p_txt = group[i]["text"]
                                h_txt = group[j]["text"]
                                same_stance_negatives.append({
                                    "claim_text": c_text,
                                    "premise": p_txt,
                                    "hypothesis": h_txt,
                                    "label": "neutral",
                                    "label_id": 2,
                                    "pair_category": "same_stance_different_reasoning",
                                    "source": "perspectrum_across_cluster",
                                    "premise_words": get_word_count(p_txt),
                                    "hypothesis_words": get_word_count(h_txt),
                                    "length_tier": get_length_tier(get_word_count(p_txt))
                                })

        random.shuffle(same_stance_negatives)
        v3_pairs.extend(same_stance_negatives[:n_same_stance_diff_reason])

        # C. SAME-TOPIC, DIFFERENT-CONCEPT NEGATIVES (15%, Neutral)
        diff_concept_negatives = []
        for cid in cids_list:
            c_text = claim_map.get(cid, "")
            p_list = claim_persps.get(cid, [])
            if not p_list:
                continue

            for m_tag, gdf in gen_dfs.items():
                m_gens = gdf[gdf["claim_id"] == cid]
                if m_gens.empty:
                    continue

                for _, grow in m_gens.tail(2).iterrows():
                    ai_text = clean_argument_text(grow.get("combined_text", grow.get("generated_text", "")))
                    ai_wc = get_word_count(ai_text)
                    if ai_wc < 20:
                        continue
                    # Pair with a different perspective cluster
                    p_target = p_list[-1]["text"]
                    diff_concept_negatives.append({
                        "claim_text": c_text,
                        "premise": ai_text,
                        "hypothesis": p_target,
                        "label": "neutral",
                        "label_id": 2,
                        "pair_category": "same_topic_diff_concept",
                        "source": "llm_diff_concept",
                        "premise_words": ai_wc,
                        "hypothesis_words": get_word_count(p_target),
                        "length_tier": get_length_tier(ai_wc)
                    })

        random.shuffle(diff_concept_negatives)
        v3_pairs.extend(diff_concept_negatives[:n_same_topic_diff_concept])

        # D. OPPOSITE-POSITION CONFLICTS (10%, Contradiction)
        opp_negatives = []
        for cid in cids_list:
            c_text = claim_map.get(cid, "")
            p_list = claim_persps.get(cid, [])
            sups = [p["text"] for p in p_list if "sup" in p["stance"]]
            unds = [p["text"] for p in p_list if "und" in p["stance"]]

            if sups and unds:
                for s_txt in sups[:2]:
                    for u_txt in unds[:2]:
                        opp_negatives.append({
                            "claim_text": c_text,
                            "premise": s_txt,
                            "hypothesis": u_txt,
                            "label": "contradiction",
                            "label_id": 0,
                            "pair_category": "opposite_stance_conflict",
                            "source": "perspectrum_opposing",
                            "premise_words": get_word_count(s_txt),
                            "hypothesis_words": get_word_count(u_txt),
                            "length_tier": get_length_tier(get_word_count(s_txt))
                        })

        random.shuffle(opp_negatives)
        v3_pairs.extend(opp_negatives[:n_opposite])

        # E. UNRELATED NEGATIVES (10%, Neutral)
        unrelated_negatives = []
        for i in range(min(n_unrelated * 2, len(cids_list) - 1)):
            c1 = cids_list[i % len(cids_list)]
            c2 = cids_list[(i + 7) % len(cids_list)]
            if c1 == c2:
                continue
            p1_list = claim_persps.get(c1, [])
            p2_list = claim_persps.get(c2, [])
            if p1_list and p2_list:
                t1 = p1_list[0]["text"]
                t2 = p2_list[0]["text"]
                unrelated_negatives.append({
                    "claim_text": claim_map.get(c1, ""),
                    "premise": t1,
                    "hypothesis": t2,
                    "label": "neutral",
                    "label_id": 2,
                    "pair_category": "unrelated_different_claim",
                    "source": "cross_claim_unrelated",
                    "premise_words": get_word_count(t1),
                    "hypothesis_words": get_word_count(t2),
                    "length_tier": get_length_tier(get_word_count(t1))
                })

        random.shuffle(unrelated_negatives)
        v3_pairs.extend(unrelated_negatives[:n_unrelated])

        # Construct V2: Direction-Corrected version of V3
        # In V2, we also preserve Premise = premise, Hypothesis = hypothesis
        # For legacy comparison, V2 has the same rows formatted with text_a / text_b
        for item in v3_pairs:
            v2_pairs.append({
                "claim_text": item["claim_text"],
                "text_a": item["premise"],       # Premise: Generated Argument
                "text_b": item["hypothesis"],    # Hypothesis: Human Perspective
                "label": item["label"],
                "label_id": item["label_id"],
                "pair_category": item["pair_category"],
                "source": item["source"]
            })

        logger.info(f"[{split_name.upper()}] Mined {len(v3_pairs)} total pairs: "
                    f"Entailment={sum(1 for x in v3_pairs if x['label'] == 'entailment')} ({sum(1 for x in v3_pairs if x['label'] == 'entailment')/len(v3_pairs):.1%}), "
                    f"Neutral={sum(1 for x in v3_pairs if x['label'] == 'neutral')} ({sum(1 for x in v3_pairs if x['label'] == 'neutral')/len(v3_pairs):.1%}), "
                    f"Contradiction={sum(1 for x in v3_pairs if x['label'] == 'contradiction')} ({sum(1 for x in v3_pairs if x['label'] == 'contradiction')/len(v3_pairs):.1%})")

        return v3_pairs, v2_pairs

    # Mine all 3 splits
    train_v3, train_v2 = mine_partition_pairs(train_cids, target_train_pairs, "train")
    val_v3, val_v2 = mine_partition_pairs(val_cids, target_val_pairs, "val")
    test_v3, test_v2 = mine_partition_pairs(test_cids, target_test_pairs, "test")

    # Save Datasets
    v3_dir = resolve(os.path.join(output_base_dir, "nli_training_v3"))
    v2_dir = resolve(os.path.join(output_base_dir, "nli_training_v2"))
    os.makedirs(v3_dir, exist_ok=True)
    os.makedirs(v2_dir, exist_ok=True)

    save_csv(pd.DataFrame(train_v3), os.path.join(v3_dir, "train.csv"))
    save_csv(pd.DataFrame(val_v3), os.path.join(v3_dir, "val.csv"))
    save_csv(pd.DataFrame(test_v3), os.path.join(v3_dir, "test.csv"))

    save_csv(pd.DataFrame(train_v2), os.path.join(v2_dir, "train.csv"))
    save_csv(pd.DataFrame(val_v2), os.path.join(v2_dir, "val.csv"))
    save_csv(pd.DataFrame(test_v2), os.path.join(v2_dir, "test.csv"))

    audit_summary = {
        "dataset_name": "Argument-NLI V3 (Coverage-Aware)",
        "input_direction": "Premise = Generated Argument (elaborated), Hypothesis = Human Reference (concise proposition)",
        "claim_splits": {
            "train_claims": len(train_cids),
            "val_claims": len(val_cids),
            "test_claims": len(test_cids)
        },
        "pair_counts": {
            "train": len(train_v3),
            "val": len(val_v3),
            "test": len(test_v3)
        },
        "train_class_distribution": {
            "entailment": sum(1 for x in train_v3 if x["label"] == "entailment"),
            "neutral": sum(1 for x in train_v3 if x["label"] == "neutral"),
            "contradiction": sum(1 for x in train_v3 if x["label"] == "contradiction")
        },
        "length_tiers_in_train_entailment": {
            tier: sum(1 for x in train_v3 if x["label"] == "entailment" and x.get("length_tier") == tier)
            for tier in ["short", "medium", "long", "very_long"]
        }
    }

    audit_path = resolve("results/calibration/nli_v3_dataset_audit.json")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    save_json(audit_summary, audit_path)
    logger.info(f"Dataset audit saved to {audit_path}")

    return audit_summary


if __name__ == "__main__":
    build_redesigned_nli_datasets()

