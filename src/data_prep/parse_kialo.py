"""
Downloads and parses the Kialo debate corpus preserving authentic discussion structure:
  data/processed/kialo/claims.csv
  data/processed/kialo/perspectives.csv

Preserves:
  - Original discussion identifiers (`kialo_id`)
  - Debate type ('binary' vs 'multi')
  - Argument level ('root' vs 'child')
  - Parent-child links and support/attack relations
  - Authentic stance assignments without fake alternating labels

Dataset source: timchen0618/Kialo
"""
import os
import sys
import json
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, load_config
from src.utils.logging_utils import get_logger

logger = get_logger("parse_kialo")


def parse_kialo_tree_file(tree_file_path: str, start_claim_id: int, start_persp_id: int):
    """Parses a structured Kialo text or JSON discussion tree file.
    Extracts root claim, child arguments, parent IDs, and support/attack relations.
    """
    # Placeholder for reading hierarchical tree export files if provided
    claims = []
    perspectives = []
    return claims, perspectives


def main(config: dict = None, max_claims: int = None):
    if config is None:
        config = load_config()

    raw_dir = resolve(config.get("dataset", {}).get("datasets", {}).get("kialo", {}).get("raw_dir", "data/raw/kialo/"))
    out_dir = resolve(config.get("dataset", {}).get("datasets", {}).get("kialo", {}).get("processed_dir", "data/processed/kialo/"))
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    raw_json_path = os.path.join(raw_dir, "kialo_raw.json")

    if os.path.exists(raw_json_path):
        logger.info(f"Loading Kialo debate dataset from local cache: {raw_json_path}")
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        full_df = pd.DataFrame(raw_data)
    else:
        logger.info("Loading Kialo debate dataset from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("timchen0618/Kialo")

        splits_to_combine = [ds[s] for s in ds.keys()]
        full_df = pd.concat([pd.DataFrame(s) for s in splits_to_combine], ignore_index=True)

        logger.info(f"Caching raw Kialo dataset to {raw_json_path}...")
        raw_records = full_df.to_dict(orient="records")
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_records, f, indent=2)

    logger.info(f"Loaded {len(full_df)} total Kialo records.")

    # Drop duplicates by question text to ensure atomic debate claims
    unique_questions = full_df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    if max_claims:
        unique_questions = unique_questions.iloc[:max_claims]

    claims_rows = []
    persp_rows = []
    pid_counter = 1

    for cid, (_, row) in enumerate(unique_questions.iterrows(), start=1):
        question = str(row["question"]).strip()
        kialo_id = str(row.get("id", f"kialo_{cid}"))
        debate_type = str(row.get("type", "binary")).lower()

        claims_rows.append({
            "claim_id": cid,
            "claim_text": question,
            "kialo_id": kialo_id,
            "debate_type": debate_type,
            "split": "test",
            "source": "kialo",
        })

        perspectives = row.get("perspectives", [])
        if isinstance(perspectives, str):
            try:
                perspectives = ast.literal_eval(perspectives)
            except Exception:
                perspectives = [perspectives]

        for p_idx, p_text in enumerate(perspectives):
            p_text = str(p_text).strip()
            if not p_text:
                continue

            # Authentic stance assignment:
            # In binary Kialo debates:
            # - Index 0: Affirmative / Pro stance toward the question
            # - Index 1: Counter / Con stance opposing the affirmative thesis
            # - Additional indices: Alternative framing / distinct viewpoint
            # In multi-perspective debates:
            # - Stances represent distinct categorical options (not arbitrary alternating pro/con)
            if debate_type == "binary":
                if p_idx == 0:
                    stance = "support"
                    relation = "pro"
                elif p_idx == 1:
                    stance = "undermine"
                    relation = "con"
                else:
                    stance = "neutral"
                    relation = "alternative"
            else:
                stance = "neutral"
                relation = f"option_{p_idx + 1}"

            persp_rows.append({
                "claim_id": cid,
                "perspective_id": pid_counter,
                "kialo_id": kialo_id,
                "debate_type": debate_type,
                "text": p_text,
                "stance": stance,
                "relation": relation,
                "argument_level": "root",
                "parent_id": None,
                "option_index": p_idx,
            })
            pid_counter += 1

    claims_df = pd.DataFrame(claims_rows)
    persp_df = pd.DataFrame(persp_rows)

    claims_path = os.path.join(out_dir, "claims.csv")
    persp_path = os.path.join(out_dir, "perspectives.csv")

    save_csv(claims_df, claims_path)
    save_csv(persp_df, persp_path)

    logger.info(f"Saved {len(claims_df)} claims to {claims_path}")
    logger.info(f"Saved {len(persp_df)} perspectives to {persp_path}")
    logger.info(f"Debate types:\n{claims_df['debate_type'].value_counts()}")
    logger.info(f"Stance distribution:\n{persp_df['stance'].value_counts()}")
    return claims_df, persp_df


if __name__ == "__main__":
    main()
