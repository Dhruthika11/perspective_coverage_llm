"""
Prepare processed_generations.csv for each of the in-family generator models:
  - qwen2_5_7b_instruct
  - llama_3_1_8b_instruct
  - gemma_2_9b_it
  - olmo_2_1124_7b_instruct

Parses arguments from data/llm_outputs/multimodel_extracted_arguments.csv into standard
processed_generations.csv format required for BGE embeddings and two-stage coverage.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, load_csv
from src.utils.logging_utils import get_logger
from src.training.prepare_clean_logfo_benchmark import parse_argument_text

logger = get_logger("prepare_multimodel_processed_generations")


def prepare_in_family_generations(
    input_csv: str = "data/llm_outputs/multimodel_extracted_arguments.csv",
    output_base: str = "data/llm_outputs"
):
    df = load_csv(resolve(input_csv))
    logger.info(f"Loaded {len(df)} extracted arguments across models: {df['model'].unique().tolist()}")

    for model_name, grp in df.groupby("model"):
        out_dir = resolve(os.path.join(output_base, model_name))
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "processed_generations.csv")

        records = []
        claim_counters = {}
        for _, r in grp.iterrows():
            cid = int(r["claim_id"])
            s_id = claim_counters.get(cid, 0)
            claim_counters[cid] = s_id + 1
            p, s, reasoning, stance = parse_argument_text(r["argument_text"])
            records.append({
                "claim_id": cid,
                "sample_id": s_id,
                "viewpoint_id": 0,
                "generated_text": r["argument_text"],
                "llm_reported_stance": stance,
                "perspective_summary": s,
                "core_reasoning": reasoning,
                "parse_ok": True,
                "parse_method": "structured_argument_parse",
                "combined_text": p if p else r["argument_text"]
            })

        df_out = pd.DataFrame(records)
        save_csv(df_out, out_file)
        logger.info(f"Saved {len(df_out)} processed generations for [{model_name}] to {out_file}")

    print("Successfully prepared processed_generations.csv for all in-family generator models.")


if __name__ == "__main__":
    prepare_in_family_generations()

