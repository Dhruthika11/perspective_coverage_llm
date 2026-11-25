"""
V4 Pilot Dataset Generator: Implements Priority 2 (Targeted Contrastive Pairs) 
and Priority 3 (4-Class Coverage Taxonomy) with Length-Stratified Balancing.

Balances:
  - 4 Model Families: Gemma-2-9B, Llama-3.1-8B, OLMo-2-7B, Qwen-2.5-7B (125 pairs each)
  - 4 Length Tiers (25% each):
      * Short: 20–50 words (125 pairs)
      * Medium: 50–100 words (125 pairs)
      * Long: 100–175 words (125 pairs)
      * Very Long: 175–250 words (125 pairs)
  - 5 Subtypes / 4 Classes:
      * full_coverage (150 pairs: 38 short, 38 medium, 37 long, 37 very_long)
      * partial_coverage (100 pairs: 25 short, 25 medium, 25 long, 25 very_long)
      * same_policy_diff_reason (100 pairs: 25 short, 25 medium, 25 long, 25 very_long) [Hard Negatives]
      * same_topic_diff_concept (100 pairs: 25 short, 25 medium, 25 long, 25 very_long) [Topical Negatives]
      * opposite_stance_conflict (50 pairs: 12 short, 13 medium, 12 long, 13 very_long) [Contradictions]

Constraints:
  - Strictly disjoint from the locked 1,118-pair test split (Zero Data Contamination).
  - Generates claim-disjoint Train (80%) and Val (20%) splits.
"""

import os
import sys
import json
import re
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("prepare_v4_pilot_dataset")


def parse_argument_record(text: str) -> Tuple[str, str, str, str]:
    """
    Parses raw LLM argument text, extracting clean full text, summary, reasoning, and stance.
    Handles JSON blocks inside markdown code fences or bare JSON.
    """
    text = str(text).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            d = json.loads(text)
            summary = d.get("perspective_summary", "").strip()
            reasoning = d.get("core_reasoning", "").strip()
            stance = d.get("stance", "").strip().upper()
            if summary and reasoning:
                full_text = f"{summary} {reasoning}"
            else:
                full_text = summary or reasoning or text
            return full_text, summary, reasoning, stance
        except Exception:
            pass
    return text, text, "", ""


def compute_jaccard(s1: str, s2: str) -> float:
    w1 = set(re.findall(r"\w+", str(s1).lower()))
    w2 = set(re.findall(r"\w+", str(s2).lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def synthesize_premise_by_tier(
    claim_text: str,
    summary: str,
    reasoning: str,
    tier: str,
    subtype: str,
    stance: str = "SUPPORT"
) -> str:
    """
    Synthesizes argument text to land strictly within the designated word count tier:
      - short: [20, 50] words
      - medium: [50, 100) words
      - long: [100, 175) words
      - very_long: [175, 250] words
    """
    summary = summary.strip().rstrip(".")
    reasoning = reasoning.strip().rstrip(".")
    claim = claim_text.strip().rstrip(".")
    
    if subtype == "partial_coverage":
        if tier == "short":
            txt = f"Regarding {claim}, substantial policy and public considerations indicate general support for this position across societal and civic institutions."
        elif tier == "medium":
            txt = (
                f"In addressing the public debate on whether {claim.lower()}, comprehensive institutional analysis suggests that this overarching policy position is warranted. "
                f"Multiple societal, economic, and practical considerations align with this general stance across diverse governance sectors, highlighting why broad policy consensus supports the fundamental proposition."
            )
        elif tier == "long":
            txt = (
                f"When evaluating the controversy surrounding whether {claim.lower()}, substantial institutional and substantive arguments indicate broad merit for this general stance. "
                f"A comprehensive review of the public discourse shows that multiple societal, economic, and ethical considerations align with this conclusion across contemporary jurisdictions. "
                f"Decision-makers and analysts frequently emphasize that maintaining the status quo produces stagnation and fails to address widespread systemic concerns. "
                f"Consequently, any sound policy assessment regarding {claim.lower()} must recognize this overarching momentum, as adopting supportive measures aligns with prevailing empirical and civic priorities."
            )
        else:  # very_long
            txt = (
                f"A comprehensive analysis of the policy proposition that {claim.lower()} requires a multi-faceted examination of both foundational principles and systemic consequences across society. "
                f"From a high-level institutional standpoint, extensive evidence and scholarly debate affirm that this position is supported by critical overarching priorities across contemporary governance. "
                f"Beyond broad agreement on the general necessity of addressing this issue, socio-economic and structural dynamics consistently reinforce the conclusion that decisive action brings substantial public benefit. "
                f"When examining comparative policy precedents and institutional frameworks across diverse jurisdictions, observers repeatedly find that delaying reform generates severe long-term complications, regulatory bottlenecks, and public dissatisfaction. "
                f"Critics often focus on isolated operational uncertainties, but they fail to account for the broader positive effects that a coherent stance exerts on institutional credibility, civic trust, and strategic alignment. "
                f"Ultimately, a rigorous assessment of {claim.lower()} confirms that this general stance reflects an essential, defensible, and enduring component of modern public policy discourse."
            )
    else:
        # full_coverage, same_policy_diff_reason, same_topic_diff_concept, opposite_stance_conflict
        if tier == "short":
            txt = f"{summary}. {reasoning}."
            if len(txt.split()) < 22:
                txt = f"Regarding {claim}, {summary.lower()}. {reasoning}."
        elif tier == "medium":
            txt = (
                f"In evaluating the debate on {claim.lower()}, a key consideration is that {summary.lower()}. "
                f"Crucially, {reasoning}. "
                f"This perspective highlights why substantive considerations on this issue must thoroughly account for these direct consequences across affected stakeholders."
            )
        elif tier == "long":
            txt = (
                f"When evaluating the controversy surrounding whether {claim.lower()}, substantial institutional and substantive arguments demonstrate that {summary.lower()}. "
                f"Specifically, {reasoning}. "
                f"Furthermore, empirical and practical considerations suggest that failing to recognize this dynamic leads to significant structural inefficiencies, misaligned incentives, and unresolved societal tensions. "
                f"Consequently, any sound decision regarding {claim.lower()} must carefully incorporate these foundational mechanisms, as neglecting them risks producing adverse systemic outcomes across the broader domain."
            )
        else:  # very_long
            txt = (
                f"A comprehensive analysis of the policy proposition that {claim.lower()} requires a multi-faceted examination of both underlying principles and practical consequences across all relevant domains. "
                f"From a critical analytical standpoint, a central argument is that {summary.lower()}. "
                f"In particular, {reasoning}. "
                f"Beyond this primary mechanism, broader socio-economic and structural considerations further reinforce this position. When examining similar policy precedents and institutional frameworks, evidence consistently indicates that ignoring these factors generates severe long-term complications and unintended distortions. "
                f"Opponents often fail to account for the secondary effects that such measures exert on individual decision-making, procedural fairness, and administrative implementation. By contrast, grounding the evaluation in verifiable operational realities confirms that this argument addresses a central, non-trivial facet of the overall controversy. "
                f"Ultimately, a rigorous assessment of {claim.lower()} must prioritize these causal dynamics, ensuring that decisions are guided by robust substantive justifications rather than unexamined assumptions."
            )

    # Word bounds enforcement
    bounds = {
        "short": (20, 50, "Further relevant institutional considerations underscore this core analytical point."),
        "medium": (50, 99, "Additionally, relevant institutional evidence confirms the necessity of taking these factors into account in public discourse."),
        "long": (100, 174, "Furthermore, empirical data and institutional precedents demonstrate that overlooking these dynamics leads to systemic distortions in decision-making and long-term public policy outcomes."),
        "very_long": (175, 250, "In light of these comprehensive observations, scholars and institutional stakeholders maintain that robust policy evaluation requires addressing these structural mechanisms directly rather than relying on unexamined suppositions.")
    }
    min_w, max_w, pad = bounds[tier]
    words = txt.split()
    while len(words) < min_w:
        txt += " " + pad
        words = txt.split()
    if len(words) > max_w:
        txt = " ".join(words[:max_w]) + "."
    return txt


def build_v4_pilot_candidates(
    output_csv: str = "data/nli_training_v4/pilot_500_candidates.csv",
    output_manifest: str = "data/nli_training_v4/pilot_500_manifest.json",
    seed: int = 42
) -> pd.DataFrame:
    random.seed(seed)
    
    # 1. Load PERSPECTRUM claims and perspective clusters
    answers_p = resolve("data/raw/perspectrum/perspectrum_with_answers_v1.0.json")
    pool_p = resolve("data/raw/perspectrum/perspective_pool_v1.0.json")
    
    with open(answers_p, "r", encoding="utf-8") as f:
        claims_data = json.load(f)
    with open(pool_p, "r", encoding="utf-8") as f:
        pool_dict = {item["pId"]: item["text"] for item in json.load(f)}
        
    logger.info(f"Loaded {len(claims_data)} claims and {len(pool_dict)} perspectives from PERSPECTRUM.")

    # 2. Identify test set claims to strictly enforce Zero Data Contamination
    test_csv = resolve("data/nli_training_v3/test.csv")
    test_cids = set()
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        test_claims_text = set(test_df["claim_text"].dropna().unique())
        test_cids = {c["cId"] for c in claims_data if c.get("text", "").strip() in test_claims_text}
        logger.info(f"Identified {len(test_cids)} locked test claim IDs to exclude from V4 pilot training data.")

    # 3. Load Multi-Model Extracted Arguments
    multimodel_csv = resolve("data/llm_outputs/multimodel_extracted_arguments.csv")
    args_by_model = defaultdict(lambda: defaultdict(list))
    
    if os.path.exists(multimodel_csv):
        logger.info(f"Loading multi-model arguments from {multimodel_csv}...")
        mm_df = pd.read_csv(multimodel_csv)
        non_test_mm = mm_df[~mm_df["claim_id"].astype(int).isin(test_cids)].copy()
        logger.info(f"Filtered to {len(non_test_mm)} arguments from non-test claims.")
        
        for _, row in non_test_mm.iterrows():
            cid = int(row["claim_id"])
            m = str(row["model"])
            full_txt, summary, reasoning, stance = parse_argument_record(row["argument_text"])
            args_by_model[m][cid].append({
                "model": m,
                "claim_id": cid,
                "full_text": full_txt,
                "summary": summary,
                "reasoning": reasoning,
                "stance": stance,
                "claim_text": str(row.get("claim_text", ""))
            })
    else:
        # Fallback to local Qwen generations if multimodel is unavailable
        logger.info("Multimodel CSV not found; falling back to local Qwen generations...")
        for q_dir in ["qwen2_5_1_5b_instruct", "qwen2_5_3b_instruct", "qwen2_5_32b_instruct"]:
            dir_path = resolve(f"data/llm_outputs/raw_generations/{q_dir}")
            if not os.path.exists(dir_path):
                continue
            for fname in os.listdir(dir_path):
                if not fname.endswith(".json"):
                    continue
                try:
                    cid = int(fname.split("_")[0])
                    if cid in test_cids:
                        continue
                    fpath = os.path.join(dir_path, fname)
                    with open(fpath, "r", encoding="utf-8") as fp:
                        c_data = json.load(fp)
                    for samp in c_data.get("samples", []):
                        full_txt, summary, reasoning, stance = parse_argument_record(samp)
                        args_by_model[q_dir][cid].append({
                            "model": q_dir,
                            "claim_id": cid,
                            "full_text": full_txt,
                            "summary": summary,
                            "reasoning": reasoning,
                            "stance": stance,
                            "claim_text": c_data.get("claim_text", "")
                        })
                except Exception:
                    pass

    active_models = sorted(list(args_by_model.keys()))
    logger.info(f"Active generator models ({len(active_models)}): {active_models}")

    claim_dict = {c["cId"]: c for c in claims_data if c["cId"] not in test_cids}

    # Length tiers: short (20-50), medium (50-100), long (100-175), very_long (175-250)
    tiers = ["short", "medium", "long", "very_long"]

    # Target counts per subtype across tiers:
    # full_coverage: 150 -> short: 38, medium: 38, long: 37, very_long: 37
    # partial_coverage: 100 -> short: 25, medium: 25, long: 25, very_long: 25
    # same_policy_diff_reason: 100 -> short: 25, medium: 25, long: 25, very_long: 25
    # same_topic_diff_concept: 100 -> short: 25, medium: 25, long: 25, very_long: 25
    # opposite_stance_conflict: 50 -> short: 12, medium: 13, long: 12, very_long: 13
    subtype_tier_quotas = {
        "full_coverage": {"short": 38, "medium": 38, "long": 37, "very_long": 37},
        "partial_coverage": {"short": 25, "medium": 25, "long": 25, "very_long": 25},
        "same_policy_diff_reason": {"short": 25, "medium": 25, "long": 25, "very_long": 25},
        "same_topic_diff_concept": {"short": 25, "medium": 25, "long": 25, "very_long": 25},
        "opposite_stance_conflict": {"short": 12, "medium": 13, "long": 12, "very_long": 13}
    }

    # Tracking generated counts
    counts = {st: {t: 0 for t in tiers} for st in subtype_tier_quotas}
    all_pairs: List[Dict[str, Any]] = []

    # Iterate through models in balanced round-robin fashion
    model_cycle = active_models.copy()
    m_idx = 0

    all_usable_cids = list(claim_dict.keys())
    random.shuffle(all_usable_cids)

    for cid in all_usable_cids:
        c_info = claim_dict[cid]
        claim_text = c_info.get("text", "")
        perspectives = c_info.get("perspectives", [])

        # Rotate model
        m = model_cycle[m_idx % len(model_cycle)]
        m_idx += 1

        args_list = args_by_model[m].get(cid, [])
        if not args_list or not perspectives:
            continue

        # Partition perspectives by stance
        sup_persps = []
        und_persps = []
        for p in perspectives:
            st = p.get("stance_label_3", "").upper()
            pids = p.get("pids", [])
            texts = [pool_dict[pid] for pid in pids if pid in pool_dict]
            if texts:
                if st == "SUPPORT":
                    sup_persps.append(texts[0])
                elif st == "UNDERMINE":
                    und_persps.append(texts[0])

        for arg_obj in args_list:
            arg_text = arg_obj["full_text"]
            arg_summary = arg_obj["summary"] or arg_text
            arg_reasoning = arg_obj["reasoning"] or arg_text
            arg_stance = arg_obj["stance"] or "SUPPORT"

            matching_pool = sup_persps if arg_stance == "SUPPORT" else und_persps
            opposing_pool = und_persps if arg_stance == "SUPPORT" else sup_persps

            # 1. Full Coverage
            for t in tiers:
                if counts["full_coverage"][t] < subtype_tier_quotas["full_coverage"][t] and matching_pool:
                    scored = [(compute_jaccard(arg_text, p_text), p_text) for p_text in matching_pool]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_match = scored[0][1]
                    prem = synthesize_premise_by_tier(claim_text, arg_summary, arg_reasoning, t, "full_coverage", arg_stance)
                    all_pairs.append({
                        "claim_id": cid,
                        "claim_text": claim_text,
                        "human_perspective": best_match,
                        "target_stance": arg_stance,
                        "llm_argument": prem,
                        "generator_model": m,
                        "proposed_label": "full_coverage",
                        "subtype": "full_coverage",
                        "length_tier": t,
                        "premise_words": len(prem.split()),
                        "hypothesis_words": len(best_match.split()),
                        "reasoning_rationale": f"Candidate entails policy stance and directly covers primary perspective rationale (Length Tier: {t})."
                    })
                    counts["full_coverage"][t] += 1
                    break

            # 2. Partial Coverage
            for t in tiers:
                if counts["partial_coverage"][t] < subtype_tier_quotas["partial_coverage"][t] and matching_pool:
                    p_target = matching_pool[0]
                    prem = synthesize_premise_by_tier(claim_text, arg_summary, arg_reasoning, t, "partial_coverage", arg_stance)
                    all_pairs.append({
                        "claim_id": cid,
                        "claim_text": claim_text,
                        "human_perspective": p_target,
                        "target_stance": arg_stance,
                        "llm_argument": prem,
                        "generator_model": m,
                        "proposed_label": "partial_coverage",
                        "subtype": "partial_coverage",
                        "length_tier": t,
                        "premise_words": len(prem.split()),
                        "hypothesis_words": len(p_target.split()),
                        "reasoning_rationale": f"Candidate affirms general policy conclusion but leaves perspective rationale unelaborated (Length Tier: {t})."
                    })
                    counts["partial_coverage"][t] += 1
                    break

            # 3. Same Policy, Different Reasoning (Hard Negative)
            for t in tiers:
                if counts["same_policy_diff_reason"][t] < subtype_tier_quotas["same_policy_diff_reason"][t] and len(matching_pool) >= 2:
                    scored = [(compute_jaccard(arg_text, p_text), p_text) for p_text in matching_pool]
                    scored.sort(key=lambda x: x[0])  # Lowest lexical overlap = distinct rationale
                    distinct_match = scored[0][1]
                    prem = synthesize_premise_by_tier(claim_text, arg_summary, arg_reasoning, t, "same_policy_diff_reason", arg_stance)
                    all_pairs.append({
                        "claim_id": cid,
                        "claim_text": claim_text,
                        "human_perspective": distinct_match,
                        "target_stance": arg_stance,
                        "llm_argument": prem,
                        "generator_model": m,
                        "proposed_label": "no_coverage",
                        "subtype": "same_policy_diff_reason",
                        "length_tier": t,
                        "premise_words": len(prem.split()),
                        "hypothesis_words": len(distinct_match.split()),
                        "reasoning_rationale": f"Both share the policy/stance, but perspective addresses distinct rationale not covered by argument (Length Tier: {t})."
                    })
                    counts["same_policy_diff_reason"][t] += 1
                    break

            # 4. Opposite Stance Conflict (Contradiction)
            for t in tiers:
                if counts["opposite_stance_conflict"][t] < subtype_tier_quotas["opposite_stance_conflict"][t] and opposing_pool:
                    contra_match = opposing_pool[0]
                    opp_stance = "UNDERMINE" if arg_stance == "SUPPORT" else "SUPPORT"
                    prem = synthesize_premise_by_tier(claim_text, arg_summary, arg_reasoning, t, "opposite_stance_conflict", arg_stance)
                    all_pairs.append({
                        "claim_id": cid,
                        "claim_text": claim_text,
                        "human_perspective": contra_match,
                        "target_stance": opp_stance,
                        "llm_argument": prem,
                        "generator_model": m,
                        "proposed_label": "contradiction",
                        "subtype": "opposite_stance_conflict",
                        "length_tier": t,
                        "premise_words": len(prem.split()),
                        "hypothesis_words": len(contra_match.split()),
                        "reasoning_rationale": f"Direct rebuttal: argument supports one stance on claim while perspective supports opposing stance (Length Tier: {t})."
                    })
                    counts["opposite_stance_conflict"][t] += 1
                    break

    # 5. Same Topic, Different Concept (Topical Negative across claims sharing topic)
    topic_claims = [claim_dict[cid] for cid in all_usable_cids]
    for i in range(len(topic_claims)):
        all_done = all(counts["same_topic_diff_concept"][t] >= subtype_tier_quotas["same_topic_diff_concept"][t] for t in tiers)
        if all_done:
            break
        for j in range(i + 1, min(i + 30, len(topic_claims))):
            all_done = all(counts["same_topic_diff_concept"][t] >= subtype_tier_quotas["same_topic_diff_concept"][t] for t in tiers)
            if all_done:
                break
            c1 = topic_claims[i]
            c2 = topic_claims[j]
            t1 = set(c1.get("topics", []))
            t2 = set(c2.get("topics", []))
            if t1.intersection(t2) and c1["cId"] != c2["cId"]:
                m = model_cycle[m_idx % len(model_cycle)]
                m_idx += 1
                c1_args = args_by_model[m].get(c1["cId"], [])
                c2_persps = c2.get("perspectives", [])
                if c1_args and c2_persps:
                    pid = c2_persps[0]["pids"][0]
                    if pid in pool_dict:
                        arg_obj = c1_args[0]
                        for t in tiers:
                            if counts["same_topic_diff_concept"][t] < subtype_tier_quotas["same_topic_diff_concept"][t]:
                                prem = synthesize_premise_by_tier(c1.get("text", ""), arg_obj["summary"], arg_obj["reasoning"], t, "same_topic_diff_concept", arg_obj["stance"])
                                target_p = pool_dict[pid]
                                all_pairs.append({
                                    "claim_id": c1["cId"],
                                    "claim_text": c1.get("text", ""),
                                    "human_perspective": target_p,
                                    "target_stance": c2_persps[0].get("stance_label_3", ""),
                                    "llm_argument": prem,
                                    "generator_model": m,
                                    "proposed_label": "no_coverage",
                                    "subtype": "same_topic_diff_concept",
                                    "length_tier": t,
                                    "premise_words": len(prem.split()),
                                    "hypothesis_words": len(target_p.split()),
                                    "reasoning_rationale": f"Shared debate topic, but claims and entities are completely distinct (Length Tier: {t})."
                                })
                                counts["same_topic_diff_concept"][t] += 1
                                break


    # Shuffle all pairs deterministically
    random.shuffle(all_pairs)
    df_pilot = pd.DataFrame(all_pairs)
    if len(df_pilot) > 500:
        df_pilot = df_pilot.head(500)
    
    df_pilot.insert(0, "pair_id", range(1, len(df_pilot) + 1))
    df_pilot["reviewer_decision"] = ""
    df_pilot["reviewer_notes"] = ""

    # Save CSV
    os.makedirs(os.path.dirname(resolve(output_csv)), exist_ok=True)
    df_pilot.to_csv(resolve(output_csv), index=False)
    logger.info(f"Saved {len(df_pilot)} V4 length-stratified pilot candidates to {output_csv}")

    # Generate claim-disjoint Train/Val splits (80% train, 20% val)
    unique_cids = sorted(list(df_pilot["claim_id"].unique()))
    random.Random(seed).shuffle(unique_cids)
    n_train = int(len(unique_cids) * 0.8)
    train_cids = set(unique_cids[:n_train])
    val_cids = set(unique_cids[n_train:])

    df_train = df_pilot[df_pilot["claim_id"].isin(train_cids)].copy()
    df_val = df_pilot[df_pilot["claim_id"].isin(val_cids)].copy()

    train_path = resolve("data/nli_training_v4/train.csv")
    val_path = resolve("data/nli_training_v4/val.csv")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    logger.info(f"Saved claim-disjoint splits: Train={len(df_train)} pairs ({len(train_cids)} claims) -> {train_path}; Val={len(df_val)} pairs ({len(val_cids)} claims) -> {val_path}")

    # Compute word length statistics per tier
    tier_stats = {}
    for t in tiers:
        sub_t = df_pilot[df_pilot["length_tier"] == t]["premise_words"]
        tier_stats[t] = {
            "count": int(len(sub_t)),
            "mean_words": round(float(sub_t.mean()), 2) if len(sub_t) > 0 else 0.0,
            "min_words": int(sub_t.min()) if len(sub_t) > 0 else 0,
            "max_words": int(sub_t.max()) if len(sub_t) > 0 else 0
        }

    # Subtype-by-tier matrix
    crosstab_sub_tier = pd.crosstab(df_pilot["subtype"], df_pilot["length_tier"]).to_dict()

    manifest = {
        "dataset_name": "Argument-NLI V4 Length-Stratified Contrastive Pilot Candidates",
        "total_pairs": len(df_pilot),
        "class_distribution": df_pilot["proposed_label"].value_counts().to_dict(),
        "subtype_distribution": df_pilot["subtype"].value_counts().to_dict(),
        "length_tier_distribution": df_pilot["length_tier"].value_counts().to_dict(),
        "subtype_by_length_tier": crosstab_sub_tier,
        "model_distribution": df_pilot["generator_model"].value_counts().to_dict(),
        "unique_claims": int(df_pilot["claim_id"].nunique()),
        "test_claims_excluded": len(test_cids),
        "tier_word_count_statistics": tier_stats
    }
    save_json(manifest, resolve(output_manifest))
    logger.info(f"Saved V4 length-stratified manifest to {output_manifest}")

    print("\n" + "="*80)
    print(f"V4 LENGTH-STRATIFIED PILOT CANDIDATES SUMMARY (N = {len(df_pilot)})")
    print("="*80)
    print("Length tier distribution:")
    print(df_pilot["length_tier"].value_counts())
    print("\nWord statistics per length tier:")
    for t, s in tier_stats.items():
        print(f"  {t:10s}: count={s['count']}, min={s['min_words']}, mean={s['mean_words']}, max={s['max_words']}")
    print("\nSubtype x Length Tier breakdown:")
    print(pd.crosstab(df_pilot["subtype"], df_pilot["length_tier"]))
    print("\nModel distribution:")
    print(df_pilot["generator_model"].value_counts())
    print("="*80 + "\n")
    
    return df_pilot


if __name__ == "__main__":
    build_v4_pilot_candidates()
