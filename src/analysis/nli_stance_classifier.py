"""
NLI-based stance classification for LLM-generated perspectives.

Classifies each LLM generation as either 'support' or 'undermine' relative to
its corresponding claim using a Cross-Encoder Natural Language Inference (NLI) model
(default: cross-encoder/nli-deberta-v3-small).

Direct NLI mapping:
  Premise: Generation text (combined_text or perspective_summary)
  Hypothesis: Claim text
  - Entailment   -> 'support'
  - Contradiction -> 'undermine'
  - Neutral       -> Evaluated via relative margin between entailment and contradiction

Can be run standalone:
  python -m src.analysis.nli_stance_classifier
  python -m src.analysis.nli_stance_classifier --batch-size 32
"""
import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import get_model_tag, load_config, load_csv, resolve, save_csv
from src.utils.logging_utils import get_logger

logger = get_logger("nli_stance_classifier")

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
VALID_STANCES = {"support", "undermine"}

# Regex patterns for splitting dynamic generation prompts into distinct aspect viewpoints
_ASPECT_SPLIT_PATTERN = re.compile(
    r"\n\s*\n(?=(?:\*\*|\#\#\#)?\s*[A-Za-z\s]+(?:Viewpoint|Perspective|Standpoint|Aspect)\s*(?:\*\*)?[:\-])"
)
_NUMBERED_SPLIT_PATTERN = re.compile(
    r"\n\s*\n(?=(?:\*\*|\#\#\#)?\s*\d+[\.\)]\s*(?:\*\*)?)"
)

# Common dynamic prompt patterns that may be echoed by models
_PROMPT_PATTERNS = [
    re.compile(r"^\s*Give distinct viewpoints on the claim:.*?(?:perspective\.|concisely\.)\s*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*CRITICAL: You must not repeat.*?\n", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*Output your viewpoint concisely\.?\s*", re.IGNORECASE),
    re.compile(r"^\s*Here are distinct viewpoints on the claim:.*?:?\s*", re.IGNORECASE),
]


def _clean_text(val: Optional[Union[str, float]]) -> str:
    """Return stripped string or empty string if None, NaN, or whitespace.
    Also strips any dynamic prompt leakage echoed by the model.
    """
    if val is None:
        return ""
    if isinstance(val, float) and (np.isnan(val) or val != val):
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null", "[empty]"):
        return ""
    # Strip prompt prefix echoes if present
    changed = True
    while changed and s:
        orig = s
        for pat in _PROMPT_PATTERNS:
            s = pat.sub("", s).strip()
        changed = (s != orig)
    return s


def extract_dynamic_viewpoints(text: str) -> List[str]:
    """Splits dynamic multi-aspect generation text into distinct individual viewpoints.

    Because generation prompts in generate_llama.py dynamically ask for distinct viewpoints
    from various aspects (Economic, Ethical, Pragmatic, etc.), models often return multi-part
    text formatted as:
      - 'Economic Viewpoint: ... \n\n Financial Viewpoint: ...'
      - '1. From an economic standpoint... \n\n 2. A mandatory vaccination policy...'
      - '**Ethical Perspective:** ...'

    Returns:
      A list of distinct viewpoint strings. If no multi-viewpoint markers are detected,
      returns a single-element list containing the cleaned text.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    # Try aspect header split first (e.g. 'Economic Viewpoint: ...')
    parts = _ASPECT_SPLIT_PATTERN.split(cleaned)
    valid_parts = [p.strip() for p in parts if p.strip()]
    if len(valid_parts) > 1:
        return valid_parts

    # Try numbered list split (e.g. '1. ... \n\n 2. ...')
    parts = _NUMBERED_SPLIT_PATTERN.split(cleaned)
    valid_parts = [p.strip() for p in parts if p.strip()]
    if len(valid_parts) > 1:
        return valid_parts

    return [cleaned]


def _extract_row_text(row: pd.Series, preferred_column: Optional[str] = None) -> str:
    """Extract argument text from a row in processed_generations.csv.

    Accounts for the discrepancy between the header and content:
    The header has:
      [claim_id, sample_id, viewpoint_id, generated_text, llm_reported_stance,
       perspective_summary, core_reasoning, parse_ok, parse_method, combined_text]
    Because dynamic prompts bypassed JSON formatting:
      - 'llm_reported_stance' is null/empty
      - 'core_reasoning' is null/empty
      - 'perspective_summary' and 'combined_text' contain the unparsed multi-paragraph text

    Fallback priority:
      preferred_column -> combined_text -> generated_text -> perspective_summary -> core_reasoning
    """
    if preferred_column and preferred_column in row:
        val = _clean_text(row[preferred_column])
        if val:
            return val

    for col in ("combined_text", "generated_text", "perspective_summary", "core_reasoning"):
        if col in row:
            val = _clean_text(row[col])
            if val:
                return val

    return ""


class NLIStanceClassifier:
    """Classifies the stance (support vs. undermine) of arguments relative to claims
    using a sequence classification / cross-encoder NLI model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None,
        method: str = "direct",
    ):
        """Args:
        model_name: HuggingFace model path (e.g. cross-encoder/nli-deberta-v3-small)
        device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        method: 'direct' (Premise=Argument, Hypothesis=Claim) or 'template'
        """
        self.model_name = model_name
        self.method = method

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading NLI model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Map label names
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.contra_id = None
        self.entail_id = None
        self.neutral_id = None

        for idx, label in id2label.items():
            if "contradiction" in label:
                self.contra_id = idx
            elif "entailment" in label:
                self.entail_id = idx
            elif "neutral" in label:
                self.neutral_id = idx

        if self.contra_id is None or self.entail_id is None:
            # Fallback standard NLI label indices if config labels are numeric
            self.contra_id = 0
            self.entail_id = 1
            self.neutral_id = 2

        logger.info(
            f"NLI label indices: entailment={self.entail_id}, "
            f"contradiction={self.contra_id}, neutral={self.neutral_id}"
        )

    def predict_direct_batch(
        self,
        arguments: List[str],
        claims: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """Classify pairs directly: Premise = Argument, Hypothesis = Claim."""
        results = []
        n = len(arguments)
        iterator = range(0, n, batch_size)
        if show_progress and n > batch_size:
            iterator = tqdm(iterator, desc="Classifying stances with NLI", total=(n + batch_size - 1) // batch_size)

        for start in iterator:
            batch_args = arguments[start : start + batch_size]
            batch_claims = claims[start : start + batch_size]

            clean_pairs = []
            for arg, clm in zip(batch_args, batch_claims):
                c_arg = _clean_text(arg)
                c_clm = _clean_text(clm)
                clean_pairs.append((c_arg if c_arg else "[Empty]", c_clm if c_clm else "[Empty]"))

            inputs = self.tokenizer(
                clean_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=1).cpu().numpy()

            for i, p in enumerate(probs):
                c_arg = _clean_text(batch_args[i])
                if not c_arg:
                    results.append({
                        "predicted_stance": "unclear",
                        "confidence": 0.0,
                        "p_support": 0.0,
                        "p_undermine": 0.0,
                        "p_neutral": 1.0,
                    })
                    continue

                contra_p = float(p[self.contra_id])
                entail_p = float(p[self.entail_id])
                neutral_p = float(p[self.neutral_id]) if self.neutral_id is not None else 0.0

                total_directed = entail_p + contra_p
                if total_directed > 0:
                    p_support = entail_p / total_directed
                    p_undermine = contra_p / total_directed
                else:
                    p_support = 0.5
                    p_undermine = 0.5

                if entail_p >= contra_p:
                    stance = "support"
                    confidence = p_support
                else:
                    stance = "undermine"
                    confidence = p_undermine

                results.append({
                    "predicted_stance": stance,
                    "confidence": float(confidence),
                    "p_support": float(p_support),
                    "p_undermine": float(p_undermine),
                    "p_neutral": float(neutral_p),
                })

        return results

    def predict_template_batch(
        self,
        arguments: List[str],
        claims: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """Classify pairs via hypothesis templates:
        H_sup: 'This argument supports the claim: {claim}'
        H_und: 'This argument opposes the claim: {claim}'
        """
        results = []
        n = len(arguments)
        iterator = range(0, n, batch_size)
        if show_progress and n > batch_size:
            iterator = tqdm(iterator, desc="Classifying stances (template NLI)", total=(n + batch_size - 1) // batch_size)

        for start in iterator:
            batch_args = arguments[start : start + batch_size]
            batch_claims = claims[start : start + batch_size]

            sup_pairs = []
            und_pairs = []
            for arg, clm in zip(batch_args, batch_claims):
                c_arg = _clean_text(arg)
                c_clm = _clean_text(clm)
                pair_arg = c_arg if c_arg else "[Empty]"
                pair_clm = c_clm if c_clm else "[Empty]"
                sup_pairs.append((pair_arg, f"This argument supports the claim: {pair_clm}"))
                und_pairs.append((pair_arg, f"This argument opposes the claim: {pair_clm}"))

            all_pairs = sup_pairs + und_pairs
            inputs = self.tokenizer(
                all_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=1).cpu().numpy()

            k = len(batch_args)
            sup_probs = probs[:k]
            und_probs = probs[k:]

            for i in range(k):
                c_arg = _clean_text(batch_args[i])
                if not c_arg:
                    results.append({
                        "predicted_stance": "unclear",
                        "confidence": 0.0,
                        "p_support": 0.0,
                        "p_undermine": 0.0,
                        "p_neutral": 1.0,
                    })
                    continue

                sup_entail = float(sup_probs[i][self.entail_id])
                und_entail = float(und_probs[i][self.entail_id])

                denom = sup_entail + und_entail
                if denom > 0:
                    p_support = sup_entail / denom
                    p_undermine = und_entail / denom
                else:
                    p_support = 0.5
                    p_undermine = 0.5

                if sup_entail >= und_entail:
                    stance = "support"
                    confidence = p_support
                else:
                    stance = "undermine"
                    confidence = p_undermine

                results.append({
                    "predicted_stance": stance,
                    "confidence": float(confidence),
                    "p_support": float(p_support),
                    "p_undermine": float(p_undermine),
                    "p_neutral": 0.0,
                })

        return results

    def predict(
        self,
        arguments: List[str],
        claims: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        if len(arguments) != len(claims):
            raise ValueError(
                f"arguments and claims must have the same length, got "
                f"{len(arguments)} and {len(claims)}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if not arguments:
            return []

        if self.method == "template":
            return self.predict_template_batch(arguments, claims, batch_size=batch_size, show_progress=show_progress)
        return self.predict_direct_batch(arguments, claims, batch_size=batch_size, show_progress=show_progress)

    def predict_single(
        self,
        argument: str,
        claim: str,
    ) -> Dict[str, Union[str, float]]:
        """Classify a single (argument, claim) pair into 'support' or 'undermine'."""
        return self.predict([argument], [claim], batch_size=1, show_progress=False)[0]


def _build_claim_text_map(
    claims_df: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
) -> Dict[Union[int, str], str]:
    """Build a mapping from claim_id to claim_text supporting both int and str keys."""
    claim_map: Dict[Union[int, str], str] = {}

    if claims_df is not None and "claim_id" in claims_df.columns and "claim_text" in claims_df.columns:
        for _, r in claims_df.iterrows():
            cid = r["claim_id"]
            ctext = _clean_text(r["claim_text"])
            claim_map[cid] = ctext
            claim_map[str(cid)] = ctext
            try:
                claim_map[int(cid)] = ctext
            except (ValueError, TypeError):
                pass

    # If map is empty or config is available, check data/processed/claims.csv or prompts.csv
    if not claim_map:
        for candidate in ("data/processed/claims.csv", "data/processed/prompts.csv"):
            c_path = resolve(candidate) if config is not None else candidate
            if os.path.exists(c_path):
                try:
                    cdf = load_csv(c_path)
                    if "claim_id" in cdf.columns and "claim_text" in cdf.columns:
                        for _, r in cdf.iterrows():
                            cid = r["claim_id"]
                            ctext = _clean_text(r["claim_text"])
                            claim_map[cid] = ctext
                            claim_map[str(cid)] = ctext
                            try:
                                claim_map[int(cid)] = ctext
                            except (ValueError, TypeError):
                                pass
                        logger.info(f"Loaded {len(claim_map)} claim text mappings from {c_path}")
                        break
                except Exception as e:
                    logger.warning(f"Could not load claims from {c_path}: {e}")

    return claim_map


def classify_generations_df(
    generations_df: pd.DataFrame,
    claims_df: Optional[pd.DataFrame] = None,
    classifier: Optional[NLIStanceClassifier] = None,
    model_name: str = DEFAULT_NLI_MODEL,
    method: str = "template",
    batch_size: int = 32,
    text_column: Optional[str] = None,
) -> pd.DataFrame:
    """Classifies stances at the individual dynamic viewpoint level for all rows
    in a processed generations DataFrame.

    Handles the reality that:
      1. Header in processed_generations.csv has [claim_id, sample_id, viewpoint_id,
         generated_text, llm_reported_stance, perspective_summary, core_reasoning,
         parse_ok, parse_method, combined_text].
      2. Content is unparsed from dynamic generation prompts (asking for distinct
         viewpoints by aspect):
         - llm_reported_stance is empty/NaN (no self-reported stance was requested or parsed).
         - core_reasoning is empty/NaN.
         - combined_text / generated_text / perspective_summary hold the actual dynamic text.
      3. Generation text contains multiple distinct aspect viewpoints (e.g. 'Economic Viewpoint: ...
         Financial Viewpoint: ...' or numbered lists).

    This function decomposes each multi-aspect generation into its constituent viewpoints,
    assigns distinct viewpoint_ids (0, 1, 2, ...), and classifies each viewpoint argument
    independently into 'support' or 'undermine'.

    Args:
      generations_df: DataFrame loaded from processed_generations.csv.
      claims_df: Optional DataFrame containing claim_id and claim_text.
      classifier: Pre-instantiated NLIStanceClassifier (optional).
      model_name: HuggingFace NLI model name if classifier is created here.
      method: 'template' (recommended) or 'direct'.
      batch_size: Inference batch size.
      text_column: Optional column override; defaults to fallback chain:
                   combined_text -> generated_text -> perspective_summary.

    Returns:
      A DataFrame containing:
        claim_id, sample_id, viewpoint_id, viewpoint_text,
        llm_reported_stance, predicted_stance, confidence, p_support, p_undermine, p_neutral
    """
    if classifier is None:
        classifier = NLIStanceClassifier(model_name=model_name, method=method)

    claim_text_map = _build_claim_text_map(claims_df)

    df = generations_df.copy()

    arguments: List[str] = []
    claims: List[str] = []
    metadata_rows: List[dict] = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        cid = row["claim_id"]
        # Look up claim text
        c_text = claim_text_map.get(cid)
        if c_text is None:
            c_text = claim_text_map.get(str(cid), "")
        if not c_text and "claim_text" in row:
            c_text = _clean_text(row["claim_text"])

        # Extract argument text with robust fallback across columns
        arg_text = _extract_row_text(row, preferred_column=text_column)

        sample_id = row["sample_id"] if "sample_id" in row else row_idx
        reported_stance = row["llm_reported_stance"] if "llm_reported_stance" in row else None
        # Normalize reported stance
        if pd.isna(reported_stance) or str(reported_stance).strip().lower() in ("nan", "none", ""):
            reported_stance = None

        # Decompose dynamic multi-aspect generation into distinct viewpoints
        vps = extract_dynamic_viewpoints(arg_text)
        if not vps:
            vps = [""]

        for vp_idx, vp_text in enumerate(vps):
            arguments.append(vp_text)
            claims.append(str(c_text) if c_text is not None else "")
            metadata_rows.append({
                "claim_id": cid,
                "sample_id": sample_id,
                "viewpoint_id": vp_idx,
                "viewpoint_text": vp_text,
                "llm_reported_stance": reported_stance,
            })

    logger.info(
        f"Running NLI stance classification on {len(arguments)} distinct viewpoints "
        f"(decomposed from {len(df)} generation samples, method={classifier.method})..."
    )
    predictions = classifier.predict(arguments, claims, batch_size=batch_size)

    pred_df = pd.DataFrame(predictions)
    meta_df = pd.DataFrame(metadata_rows)

    output_df = pd.concat([meta_df, pred_df], axis=1)
    return output_df


def main(
    config: Optional[dict] = None,
    model_tag_override: Optional[str] = None,
) -> pd.DataFrame:
    if config is None:
        config = load_config()

    if model_tag_override:
        config.setdefault("paths", {})
        config["paths"]["llm_outputs_processed"] = f"data/llm_outputs/{model_tag_override}/processed_generations.csv"
        config["paths"]["metrics_dir"] = f"results/{model_tag_override}/metrics/"

    analysis_cfg = config.get("analysis", {})
    nli_model = analysis_cfg.get("nli_model", DEFAULT_NLI_MODEL)
    nli_method = analysis_cfg.get("nli_method", "template")
    batch_size = analysis_cfg.get("nli_batch_size", 32)

    processed_dir = resolve(config["dataset"]["processed_dir"])
    claims_path = os.path.join(processed_dir, "claims.csv")
    claims_df = None
    if os.path.exists(claims_path):
        claims_df = load_csv(claims_path)
    else:
        logger.warning(f"Claims file not found at {claims_path}; will attempt prompts.csv fallback.")

    gen_path = resolve(config["paths"]["llm_outputs_processed"])
    if not os.path.exists(gen_path):
        raise FileNotFoundError(
            f"Processed generations file not found: {gen_path}. "
            f"Run postprocess_generations first."
        )
    generations_df = load_csv(gen_path)

    logger.info(
        f"Loaded {len(generations_df)} generation samples from {gen_path} for NLI stance classification."
    )

    classifier = NLIStanceClassifier(model_name=nli_model, method=nli_method)

    # Classify at individual dynamic viewpoint level
    results_df = classify_generations_df(
        generations_df,
        claims_df,
        classifier=classifier,
        batch_size=batch_size,
    )

    metrics_dir = resolve(config["paths"]["metrics_dir"])
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Standard contract for downstream stages (distribution_divergence.py, stance_crosscheck.py)
    standard_cols = [
        "claim_id", "sample_id", "viewpoint_id", "viewpoint_text",
        "llm_reported_stance", "predicted_stance", "confidence"
    ]
    standard_df = results_df[[c for c in standard_cols if c in results_df.columns]]
    stances_csv_path = os.path.join(metrics_dir, "llm_sample_stances.csv")
    save_csv(standard_df, stances_csv_path)

    # 2. Detailed NLI diagnostics with full continuous probabilities (p_support, p_undermine, p_neutral)
    detailed_csv_path = os.path.join(metrics_dir, "llm_sample_stances_nli.csv")
    save_csv(results_df, detailed_csv_path)

    support_cnt = (results_df["predicted_stance"] == "support").sum()
    undermine_cnt = (results_df["predicted_stance"] == "undermine").sum()
    unclear_cnt = len(results_df) - support_cnt - undermine_cnt
    mean_conf = results_df["confidence"].mean()

    logger.info(
        f"NLI Viewpoint Stance Classification Complete:\n"
        f"  Total viewpoints classified: {len(results_df)} (from {len(generations_df)} samples)\n"
        f"  Saved to: {stances_csv_path}\n"
        f"  Distribution: support={support_cnt} ({support_cnt/len(results_df):.1%}), "
        f"undermine={undermine_cnt} ({undermine_cnt/len(results_df):.1%}), "
        f"unclear={unclear_cnt}\n"
        f"  Mean confidence: {mean_conf:.3f}"
    )

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI Viewpoint Stance Classifier for LLM generations.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face NLI model or model key")
    parser.add_argument("--nli-model", type=str, default=None, help="Hugging Face NLI model")
    parser.add_argument("--llm-model", choices=["1.5b", "3b", "7b", "12b", "32b"], default=None, help="LLM model key to classify")
    parser.add_argument("--model-tag", type=str, default=None, help="Explicit model directory tag (e.g. mistral_7b_instruct_v0_3)")
    parser.add_argument("--method", choices=["direct", "template"], default=None, help="NLI formulation method (default: template)")
    args = parser.parse_args()

    # Load config with model overrides if --llm-model is specified
    if args.llm_model:
        from run_pipeline import build_config
        cfg = build_config(model=args.llm_model)
    else:
        cfg = load_config()

    nli_model_arg = args.nli_model or (args.model if args.model and args.model not in ["1.5b", "3b", "7b", "12b", "32b"] else None)
    if nli_model_arg:
        cfg.setdefault("analysis", {})["nli_model"] = nli_model_arg
    if args.method:
        cfg.setdefault("analysis", {})["nli_method"] = args.method
    if args.batch_size:
        cfg.setdefault("analysis", {})["nli_batch_size"] = args.batch_size

    main(cfg, model_tag_override=args.model_tag)

