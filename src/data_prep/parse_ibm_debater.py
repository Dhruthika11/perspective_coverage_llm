"""
Downloads and parses the IBM Project Debater Argument Quality Ranking dataset
into standard flat CSV format:
  data/processed/ibm_debater/claims.csv
  data/processed/ibm_debater/perspectives.csv

Dataset source: ibm-research/argument_quality_ranking_30k
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, load_config
from src.utils.logging_utils import get_logger

logger = get_logger("parse_ibm_debater")


def main(config: dict = None, max_claims: int = None, min_quality: float = 0.5):
    if config is None:
        config = load_config()

    raw_dir = resolve(config.get("dataset", {}).get("datasets", {}).get("ibm_debater", {}).get("raw_dir", "data/raw/ibm_debater/"))
    out_dir = resolve(config.get("dataset", {}).get("datasets", {}).get("ibm_debater", {}).get("processed_dir", "data/processed/ibm_debater/"))
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    raw_csv_path = os.path.join(raw_dir, "argument_quality_ranking_30k.csv")

    if os.path.exists(raw_csv_path):
        logger.info(f"Loading IBM Argument Quality dataset from local cache: {raw_csv_path}")
        full_df = pd.read_csv(raw_csv_path)
    else:
        logger.info("Loading IBM Argument Quality dataset from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("ibm-research/argument_quality_ranking_30k", "argument_quality_ranking")

        # Combine all splits (train, validation, test)
        full_df = pd.concat([
            pd.DataFrame(ds["train"]),
            pd.DataFrame(ds["validation"]),
            pd.DataFrame(ds["test"]),
        ], ignore_index=True)

        logger.info(f"Caching raw IBM dataset to {raw_csv_path}...")
        save_csv(full_df, raw_csv_path)

    logger.info(f"Loaded {len(full_df)} total arguments across {full_df['topic'].nunique()} topics.")

    # Filter by argument quality (WA >= min_quality) to ensure strong debate arguments
    if min_quality is not None and "WA" in full_df.columns:
        filtered_df = full_df[full_df["WA"] >= min_quality].copy()
        logger.info(f"Retained {len(filtered_df)} arguments with quality WA >= {min_quality}")
    else:
        filtered_df = full_df.copy()

    # Map unique topics to claims
    topics = sorted(filtered_df["topic"].unique())
    if max_claims:
        topics = topics[:max_claims]
        filtered_df = filtered_df[filtered_df["topic"].isin(topics)]

    topic2id = {topic: i + 1 for i, topic in enumerate(topics)}

    claims_df = pd.DataFrame([
        {
            "claim_id": cid,
            "claim_text": topic,
            "original_topic": topic,
            "split": "test",
            "source": "ibm_debater",
        }
        for topic, cid in topic2id.items()
    ])

    persp_rows = []
    for pid, (_, row) in enumerate(filtered_df.iterrows(), start=1):
        topic = row["topic"]
        if topic not in topic2id:
            continue
        cid = topic2id[topic]
        # stance_WA: 1 -> support, -1 -> undermine
        stance_val = row.get("stance_WA", 1)
        stance_str = "support" if stance_val == 1 else "undermine"

        wa_val = float(row.get("WA", 0.0)) if pd.notna(row.get("WA")) else 0.0
        mace_val = float(row.get("MACE-P", 0.0)) if pd.notna(row.get("MACE-P")) else 0.0
        conf_val = float(row.get("stance_WA_conf", 1.0)) if pd.notna(row.get("stance_WA_conf")) else 1.0

        persp_rows.append({
            "claim_id": cid,
            "perspective_id": pid,
            "text": str(row["argument"]).strip(),
            "stance": stance_str,
            "wa_quality": wa_val,
            "mace_p_quality": mace_val,
            "stance_conf": conf_val,
            "original_set": str(row.get("set", "unknown")),
            "original_topic": topic,
        })

    persp_df = pd.DataFrame(persp_rows)

    claims_path = os.path.join(out_dir, "claims.csv")
    persp_path = os.path.join(out_dir, "perspectives.csv")

    save_csv(claims_df, claims_path)
    save_csv(persp_df, persp_path)

    logger.info(f"Saved {len(claims_df)} claims to {claims_path}")
    logger.info(f"Saved {len(persp_df)} perspectives to {persp_path}")
    logger.info(f"Stance distribution:\n{persp_df['stance'].value_counts()}")
    return claims_df, persp_df


if __name__ == "__main__":
    main()
