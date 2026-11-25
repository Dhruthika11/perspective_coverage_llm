"""
Versioned Perspective Generation Prompts & Parsers (Phase 2).

Defines the 4 controlled prompt conditions:
  - P1: Aspect=Yes (5 aspects), History=Yes, Reasoning Criterion=No (Historical Baseline)
  - P2: Aspect=Yes (5 aspects), History=Yes, Reasoning Criterion=Yes (Improved Main Prompt)
  - P3: Aspect=Yes (5 aspects), History=No,  Reasoning Criterion=Yes (Ablation: No History)
  - P4: Aspect=No  (Open aspect), History=Yes, Reasoning Criterion=Yes (Ablation: No Aspect)

Also provides structured JSON parsing, abstention detection, and schema validation.
"""
import json
import re
from typing import Dict, List, Optional, Tuple, Any

# The 5 standard deliberative aspects defined in the research plan
STANDARD_ASPECTS = [
    "Economic",
    "Moral",
    "Legal",
    "Practical",
    "Social",
]

# Illustrative criteria per aspect (not an exhaustive taxonomy, used as prompting guide)
ASPECT_CRITERIA_GUIDE = {
    "Economic": "affordability, employment, government expenditure, resource allocation, inflation, market efficiency",
    "Moral": "fairness, equality, individual responsibility, human rights, bodily autonomy, harm reduction",
    "Legal": "constitutional obligations, legal rights, statutory authority, regulatory compliance, judicial precedent",
    "Practical": "implementation feasibility, accessibility, operational logistics, infrastructure requirements, enforcement",
    "Social": "community cohesion, social inequality, behavioral impact, public trust, cultural preservation, demographic impact",
}

VALID_STANCES = {"support", "undermine"}


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def format_history_string(history: Optional[List[Dict[str, str]]]) -> str:
    """Formats cumulative generation history into a clean bulleted list."""
    if not history:
        return ""
    lines = []
    for item in history:
        st = item.get("stance", "unknown").upper()
        crit = item.get("reasoning_criterion", "")
        crit_str = f" [{crit}]" if crit and crit.lower() != "none" else ""
        arg = item.get("argument", item.get("perspective_summary", item.get("text", ""))).strip()
        if arg:
            lines.append(f"- ({st}){crit_str}: {arg}")
    return "\n".join(lines)


def build_p1_prompt(claim: str, aspect: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """P1: Historical Baseline Prompt.
    Aspect conditioning: YES
    Cumulative history: YES
    Explicit reasoning criterion: NO
    Abstention: NO
    """
    hist_text = format_history_string(history)
    history_constraint = ""
    if hist_text:
        history_constraint = f"CRITICAL: You must not repeat arguments already made previously. Prior arguments:\n{hist_text}\n"

    prompt = (
        f'You are analyzing the controversial claim: "{claim}"\n\n'
        f"Task: Provide one single, distinct, and focused argument about this claim strictly from an {aspect} perspective.\n"
        f"Do not provide a balanced overview. Take one clear, atomic stance (either SUPPORT or UNDERMINE).\n"
    )
    if history_constraint:
        prompt += f"\n{history_constraint}\n"
    prompt += (
        '\nRespond with ONLY a single JSON object matching this schema:\n'
        '{\n'
        '  "stance": "support" or "undermine",\n'
        '  "perspective_summary": "1 concise sentence stating the core claim",\n'
        '  "core_reasoning": "1-2 sentences explaining the supporting logic"\n'
        '}'
    )
    return prompt.strip()


def build_p2_prompt(claim: str, aspect: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """P2: Improved Main Generation Prompt.
    Aspect conditioning: YES
    Cumulative history: YES
    Explicit reasoning criterion: YES
    Semantic novelty constraint: YES
    Abstention: YES
    """
    sample_criteria = ASPECT_CRITERIA_GUIDE.get(aspect, "core values, operational feasibility, constitutional principles")
    hist_text = format_history_string(history)
    
    history_section = ""
    if hist_text:
        history_section = (
            f"Prior Arguments Generated on This Claim:\n{hist_text}\n\n"
            f"CRITICAL NOVELTY CONSTRAINT: Compare your proposed perspective against the prior arguments above. "
            f"A mere paraphrase or rephrasing of an existing idea is NOT a new perspective. "
            f"Your argument must introduce a DIFFERENT underlying reasoning criterion or value.\n"
        )

    prompt = (
        f'You are analyzing the controversial claim: "{claim}"\n\n'
        f"Task: Formulate one genuinely distinct human perspective on this claim strictly from an {aspect} perspective.\n\n"
        f"Guidelines:\n"
        f"1. Reasoning Criterion: Explicitly identify the underlying value, concern, or criterion that motivates this perspective "
        f"(e.g., for {aspect}: {sample_criteria}).\n"
        f"2. Atomic Stance: Adopt one clear stance (either SUPPORT or UNDERMINE the claim).\n"
        f"3. Semantic Novelty: The argument must be substantive and not a superficial re-wording of previous arguments.\n"
        f"4. Explicit Abstention: If all strong, distinct arguments under the {aspect} perspective have already been stated above, "
        f"or if no meaningful distinct argument exists, you MUST ABSTAIN. Do not invent weak, contrived, or redundant arguments.\n\n"
    )
    if history_section:
        prompt += f"{history_section}\n"

    prompt += (
        'Respond with ONLY a single JSON object matching this schema:\n'
        '{\n'
        '  "status": "argument",\n'
        f'  "aspect": "{aspect}",\n'
        '  "reasoning_criterion": "1-3 word criterion (e.g. affordability, fairness, constitutional authority)",\n'
        '  "stance": "support" or "undermine",\n'
        '  "argument": "1-3 sentences stating the perspective and its underlying justification"\n'
        '}\n\n'
        'If abstaining, respond with:\n'
        '{\n'
        '  "status": "abstain",\n'
        f'  "aspect": "{aspect}",\n'
        '  "reasoning_criterion": "none",\n'
        '  "stance": "none",\n'
        f'  "argument": "No distinct {aspect} arguments remain uncovered."\n'
        '}'
    )
    return prompt.strip()


def build_p3_prompt(claim: str, aspect: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """P3: Prompt Ablation (No History).
    Aspect conditioning: YES
    Cumulative history: NO (history argument ignored)
    Explicit reasoning criterion: YES
    Semantic novelty constraint: YES (in abstract)
    Abstention: YES
    """
    sample_criteria = ASPECT_CRITERIA_GUIDE.get(aspect, "core values, operational feasibility, constitutional principles")
    prompt = (
        f'You are analyzing the controversial claim: "{claim}"\n\n'
        f"Task: Formulate one distinct human perspective on this claim strictly from an {aspect} perspective.\n\n"
        f"Guidelines:\n"
        f"1. Reasoning Criterion: Explicitly identify the underlying value, concern, or criterion that motivates this perspective "
        f"(e.g., for {aspect}: {sample_criteria}).\n"
        f"2. Atomic Stance: Adopt one clear stance (either SUPPORT or UNDERMINE the claim).\n"
        f"3. Explicit Abstention: If no meaningful, valid argument can be made from an {aspect} perspective, you MUST ABSTAIN.\n\n"
        'Respond with ONLY a single JSON object matching this schema:\n'
        '{\n'
        '  "status": "argument",\n'
        f'  "aspect": "{aspect}",\n'
        '  "reasoning_criterion": "1-3 word criterion",\n'
        '  "stance": "support" or "undermine",\n'
        '  "argument": "1-3 sentences stating the perspective and its underlying justification"\n'
        '}\n'
        'If abstaining:\n'
        '{\n'
        '  "status": "abstain",\n'
        f'  "aspect": "{aspect}",\n'
        '  "reasoning_criterion": "none",\n'
        '  "stance": "none",\n'
        f'  "argument": "No valid {aspect} argument exists."\n'
        '}'
    )
    return prompt.strip()


def build_p4_prompt(claim: str, aspect: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
    """P4: Prompt Ablation (No Assigned Aspect).
    Aspect conditioning: NO
    Cumulative history: YES
    Explicit reasoning criterion: YES
    Semantic novelty constraint: YES
    Abstention: YES
    """
    hist_text = format_history_string(history)
    history_section = ""
    if hist_text:
        history_section = (
            f"Prior Arguments Generated on This Claim:\n{hist_text}\n\n"
            f"CRITICAL NOVELTY CONSTRAINT: Compare your proposed perspective against the prior arguments above. "
            f"Your argument must introduce a DIFFERENT underlying reasoning criterion or value.\n"
        )

    prompt = (
        f'You are analyzing the controversial claim: "{claim}"\n\n'
        f"Task: Formulate one distinct human perspective on this claim from ANY valid domain or viewpoint.\n\n"
        f"Guidelines:\n"
        f"1. Reasoning Criterion: Explicitly identify the underlying value, concern, or criterion motivating this perspective.\n"
        f"2. Atomic Stance: Adopt one clear stance (either SUPPORT or UNDERMINE the claim).\n"
        f"3. Explicit Abstention: If all strong, distinct perspectives have already been stated above, you MUST ABSTAIN.\n\n"
    )
    if history_section:
        prompt += f"{history_section}\n"

    prompt += (
        'Respond with ONLY a single JSON object matching this schema:\n'
        '{\n'
        '  "status": "argument",\n'
        '  "aspect": "name of the aspect (e.g. economic, ethical, legal, practical, social, etc.)",\n'
        '  "reasoning_criterion": "1-3 word criterion",\n'
        '  "stance": "support" or "undermine",\n'
        '  "argument": "1-3 sentences stating the perspective and its underlying justification"\n'
        '}\n'
        'If abstaining:\n'
        '{\n'
        '  "status": "abstain",\n'
        '  "aspect": "none",\n'
        '  "reasoning_criterion": "none",\n'
        '  "stance": "none",\n'
        '  "argument": "No distinct perspectives remain uncovered."\n'
        '}'
    )
    return prompt.strip()


PROMPT_BUILDERS = {
    "p1": build_p1_prompt,
    "p2": build_p2_prompt,
    "p3": build_p3_prompt,
    "p4": build_p4_prompt,
}


def build_prompt(
    version: str,
    claim: str,
    aspect: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Dispatches to the correct versioned prompt builder."""
    v = version.lower().strip()
    if v not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown prompt version: {version}. Choose from: {list(PROMPT_BUILDERS.keys())}")
    builder = PROMPT_BUILDERS[v]
    if v == "p4":
        return builder(claim, aspect=aspect, history=history)
    return builder(claim, aspect=aspect or "Economic", history=history)


# ---------------------------------------------------------------------------
# Robust Structured Output Parser
# ---------------------------------------------------------------------------

_FENCE_REGEX = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw_text: str) -> Optional[dict]:
    """Extracts and parses the first JSON object from a string, handling fences and markdown."""
    if not isinstance(raw_text, str):
        return None
    text = raw_text.strip()
    # Check for markdown code fences
    fence_match = _FENCE_REGEX.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find boundaries of the JSON object
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Direct parse attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def parse_generation_output(raw_text: str, prompt_version: str = "p2") -> Dict[str, Any]:
    """Parses raw model generation into a standardized output dictionary.
    
    Standard Schema:
      - status: "argument" | "abstain" | "parse_error"
      - aspect: str
      - reasoning_criterion: str
      - stance: "support" | "undermine" | "neutral" | "unclear" | "none"
      - argument: str
      - is_abstention: bool
      - is_valid: bool
      - parse_method: "json" | "fallback" | "unparsed"
    """
    if not isinstance(raw_text, str):
        raw_text = ""
    v = prompt_version.lower().strip()
    parsed_json = extract_json_object(raw_text)

    # Default record structure
    result = {
        "status": "argument",
        "aspect": "unknown",
        "reasoning_criterion": "unspecified",
        "stance": "unclear",
        "argument": "",
        "is_abstention": False,
        "is_valid": False,
        "parse_method": "unparsed",
        "raw_text": raw_text,
    }

    if parsed_json:
        result["parse_method"] = "json"

        # Check for abstention status
        status_val = str(parsed_json.get("status", "argument")).lower().strip()
        if "abstain" in status_val or status_val == "none":
            result["status"] = "abstain"
            result["is_abstention"] = True
            result["stance"] = "none"
            result["reasoning_criterion"] = str(parsed_json.get("reasoning_criterion", "none")).strip()
            result["argument"] = str(parsed_json.get("argument", "Model abstained.")).strip()
            result["aspect"] = str(parsed_json.get("aspect", "unknown")).strip()
            result["is_valid"] = True
            return result

        # Handle P1 schema: {"stance": ..., "perspective_summary": ..., "core_reasoning": ...}
        if v == "p1" or ("perspective_summary" in parsed_json and "argument" not in parsed_json):
            summary = str(parsed_json.get("perspective_summary", "")).strip()
            reasoning = str(parsed_json.get("core_reasoning", "")).strip()
            combined = f"{summary} {reasoning}".strip()
            stance = str(parsed_json.get("stance", "unclear")).lower().strip()

            result["argument"] = combined if combined else summary
            result["stance"] = stance if stance in VALID_STANCES else "unclear"
            result["aspect"] = str(parsed_json.get("aspect", "unknown")).strip()
            result["reasoning_criterion"] = "none"
            result["is_valid"] = bool(result["argument"] and result["stance"] in VALID_STANCES)
            return result

        # Handle P2 / P3 / P4 schema: {"status": ..., "aspect": ..., "reasoning_criterion": ..., "stance": ..., "argument": ...}
        arg_text = str(parsed_json.get("argument", "")).strip()
        stance = str(parsed_json.get("stance", "unclear")).lower().strip()
        aspect = str(parsed_json.get("aspect", "unknown")).strip()
        criterion = str(parsed_json.get("reasoning_criterion", "unspecified")).strip()

        result["argument"] = arg_text
        result["stance"] = stance if stance in VALID_STANCES else "unclear"
        result["aspect"] = aspect
        result["reasoning_criterion"] = criterion
        result["is_valid"] = bool(arg_text and result["stance"] in VALID_STANCES)
        return result

    # Fallback heuristic parsing (e.g. if model output prose with bold markers)
    lower_raw = raw_text.lower()
    if "abstain" in lower_raw and len(raw_text.split()) < 30:
        result["status"] = "abstain"
        result["is_abstention"] = True
        result["argument"] = raw_text.strip()
        result["is_valid"] = True
        result["parse_method"] = "fallback"
        return result

    # Check for Markdown **Stance:** and **Argument:**
    stance_m = re.search(r"\*\*\s*stance\s*:?\s*\*\*\s*:?\s*([a-zA-Z]+)", raw_text, re.IGNORECASE)
    arg_m = re.search(r"\*\*\s*(argument|summary|reasoning)\s*:?\s*\*\*\s*:?\s*(.*?)(?=\*\*\s*|$)", raw_text, re.IGNORECASE | re.DOTALL)
    crit_m = re.search(r"\*\*\s*(reasoning[ _]?criterion|criterion)\s*:?\s*\*\*\s*:?\s*([^\n]+)", raw_text, re.IGNORECASE)

    if stance_m and arg_m:
        st = stance_m.group(1).lower().strip()
        result["stance"] = st if st in VALID_STANCES else "unclear"
        result["argument"] = arg_m.group(2).strip()
        result["reasoning_criterion"] = crit_m.group(2).strip() if crit_m else "unspecified"
        result["parse_method"] = "fallback"
        result["is_valid"] = bool(result["argument"] and result["stance"] in VALID_STANCES)
        return result

    # Total parse failure
    result["status"] = "parse_error"
    result["argument"] = raw_text.strip()
    result["is_valid"] = False
    return result

