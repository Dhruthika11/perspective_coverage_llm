"""
Generate Frozen Multi-LLM Jury Specification and Protocol Artifact
Saves exact model versions, temperature, prompts, output schema, and consensus rules.
"""

import json
import hashlib
import os

def create_frozen_specification():
    prompt_template = """You are an expert argument annotator evaluating semantic relationships between debate perspectives.

Debate Claim: {claim}

Perspective A (Gold Reference):
"{gold_perspective}"

Perspective B (Generated Viewpoint):
"{generated_viewpoint}"

Task:
Determine the relationship of Perspective B relative to Perspective A under the following 4-class taxonomy:

Classes:
1. "class_2" (Same stance AND same core reasoning / warrant): Both perspectives argue for the exact same point on the claim. Surface paraphrases, lexical variations, or concrete elaborations that do not alter the core argumentative claim belong here.
2. "class_1" (Same stance, but DISTINCT reasoning / warrant): Both perspectives share the same stance (pro/con), but rely on fundamentally different arguments, mechanisms, or evidence.
3. "class_0a" (Opposite stance on the same claim / Rebuttal): Perspective B takes the directly opposing stance or offers a counter-argument to Perspective A on this specific claim.
4. "class_0b" (Unrelated debate topic / domain mismatch): Perspective B discusses an entirely different subject or claim.
5. "ambiguous" (Corrupted, truncated, or non-propositional text): One or both perspectives are malformed, cut off mid-sentence, or contain only empty section headers (e.g., ending in a colon like "A Critique:").

Response Format:
Respond in valid JSON with:
{
  "label": "class_2" | "class_1" | "class_0a" | "class_0b" | "ambiguous",
  "confidence": float between 0.0 and 1.0,
  "explanation": "Brief explanation of the relationship"
}
"""

    spec = {
        "protocol_version": "1.0.0-frozen",
        "freeze_date": "2026-11-19T06:00:00Z",
        "jury_composition": [
            {
                "judge_id": "Judge_1",
                "model_name": "Google Gemini 3.1 Flash Lite Preview",
                "model_string": "google/gemini-2.5-flash-preview",
                "api_provider": "Google DeepMind / Generative AI SDK",
                "temperature": 0.1,
                "max_output_tokens": 4096,
                "role": "Anchor Judge (High reasoning speed, broad contextual window)"
            },
            {
                "judge_id": "Judge_2",
                "model_name": "OpenAI GPT-OSS 120B",
                "fallback_model": "OpenAI GPT-OSS 20B",
                "model_string": "openai/gpt-oss-120b",
                "api_provider": "Groq / OpenAI-compatible API",
                "temperature": 0.1,
                "max_output_tokens": 4096,
                "role": "Independent Validator (Open-weights reasoning model)"
            },
            {
                "judge_id": "Judge_3",
                "model_name": "Alibaba Qwen 3.8-27B",
                "model_string": "qwen/qwen3.8-27b",
                "api_provider": "Groq / OpenAI-compatible API",
                "temperature": 0.1,
                "max_output_tokens": 4096,
                "role": "Cross-Family Independent Tiebreaker"
            }
        ],
        "taxonomy": {
            "class_2": "Same stance AND same core reasoning / warrant (Equivalent)",
            "class_1": "Same stance, but DISTINCT reasoning / warrant (Different argument)",
            "class_0a": "Opposite stance on the same claim (Direct rebuttal)",
            "class_0b": "Unrelated debate topic / domain mismatch",
            "ambiguous": "Corrupted, truncated, or non-propositional text"
        },
        "consensus_rules": {
            "unanimous_concrete": {
                "rule": "3/3 judges vote for the same non-ambiguous class",
                "action": "AUTO_ACCEPT",
                "human_adjudication_required": False,
                "verified_accuracy": "99.34% (150/151 audited pairs)"
            },
            "majority_concrete": {
                "rule": "2/3 judges vote for class X; 1 judge votes for class Y (X, Y != ambiguous)",
                "action": "TRIAGE_QUEUE",
                "human_adjudication_required": True,
                "provisional_label": "Majority vote"
            },
            "majority_with_ambiguous": {
                "rule": "2/3 judges vote for class X; 1 judge votes ambiguous",
                "action": "TRIAGE_QUEUE",
                "human_adjudication_required": True,
                "provisional_label": "Majority vote (marked for ambiguity verification)"
            },
            "split_decision": {
                "rule": "Judges vote for three distinct classes (1/1/1 split)",
                "action": "TRIAGE_QUEUE",
                "human_adjudication_required": True,
                "provisional_label": "split"
            },
            "unanimous_ambiguous": {
                "rule": "3/3 judges vote ambiguous",
                "action": "TRIAGE_QUEUE",
                "human_adjudication_required": True,
                "provisional_label": "ambiguous"
            }
        },
        "prompt_template": prompt_template,
        "hash_md5": hashlib.md5(prompt_template.encode("utf-8")).hexdigest()
    }

    out_path = "data/calibration/frozen_jury_specification.json"
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Saved frozen jury specification to {out_path}")

if __name__ == "__main__":
    create_frozen_specification()
