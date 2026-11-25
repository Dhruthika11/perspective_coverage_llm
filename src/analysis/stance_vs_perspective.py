"""
Decoupling Stance Detection from Perspective Equivalence.

This module formalizes the crucial scientific distinction between two fundamentally
different evaluation tasks:

Task 1: Stance Detection
  - Question: Does argument A support, undermine, or remain neutral toward claim C?
  - Input: Premise = Argument, Hypothesis = Claim
  - Direction: Directed inference from argument premise to proposition hypothesis.
  - Output Classes: SUPPORT (entailment), UNDERMINE (contradiction), NEUTRAL (neutral).

Task 2: Perspective Equivalence
  - Question: Does argument A express essentially the same underlying reasoning,
    justification, and value criterion as argument B?
  - Input: Pair (Perspective A, Perspective B) within the context of Claim C.
  - Direction: Bidirectional / semantic equivalence under stance alignment.
  - Output Classes:
      Class 2 (Equivalent): Core reasoning criterion is identical (e.g., paraphrases).
      Class 1 (Related): Common topical domain but distinct reasoning (e.g., cost vs emissions).
      Class 0 (Different / Opposing): Unrelated reasoning or conflicting stance.

Scientific Assumption Note:
  Two arguments sharing the same stance toward a claim DOES NOT imply they represent
  the same perspective. Conflating stance concordance with perspective equivalence
  leads to severe overestimation of perspective coverage.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class StanceDetector:
    """Classifies the stance of a generated argument relative to a controversial claim.
    
    Formulation:
      Premise: The generated argument
      Hypothesis: The root claim
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def predict_stance(self, argument: str, claim: str) -> Dict[str, Union[str, float]]:
        """Predicts stance: 'support', 'undermine', or 'neutral'."""
        self._lazy_load()
        import torch

        inputs = self._tokenizer(
            argument,
            claim,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # DeBERTa-v3 cross-encoder NLI mapping:
        # idx 0: contradiction -> undermine
        # idx 1: entailment    -> support
        # idx 2: neutral       -> neutral
        p_contra = float(probs[0])
        p_entail = float(probs[1])
        p_neutral = float(probs[2])

        if p_entail > p_contra and p_entail > p_neutral:
            predicted_stance = "support"
            confidence = p_entail
        elif p_contra > p_entail and p_contra > p_neutral:
            predicted_stance = "undermine"
            confidence = p_contra
        else:
            predicted_stance = "neutral"
            confidence = p_neutral

        return {
            "stance": predicted_stance,
            "confidence": confidence,
            "p_support": p_entail,
            "p_undermine": p_contra,
            "p_neutral": p_neutral,
        }


class PerspectiveMatcher:
    """Evaluates whether two arguments represent equivalent human perspectives.
    
    Requires:
      1. Stance concordance (arguments must share the same stance polarity toward the claim).
      2. Reasoning equivalence (both arguments must invoke the same underlying rationale).
    """

    EQUIVALENT = 2
    RELATED = 1
    DIFFERENT = 0

    def __init__(self, cosine_threshold: float = 0.75, verification_threshold: float = 0.80):
        self.cosine_threshold = cosine_threshold
        self.verification_threshold = verification_threshold

    @staticmethod
    def classify_perspective_pair(
        stance_a: str,
        stance_b: str,
        cosine_sim: float,
        reasoning_entailment_prob: float,
        tau_cosine: float = 0.75,
        theta_verify: float = 0.80
    ) -> int:
        """Determines the 3-class equivalence between two perspective representations:
          2 = Equivalent Perspective
          1 = Related but Distinct Reasoning
          0 = Different / Opposing Perspective
        """
        # Rule 1: Opposite or discordant stances cannot represent the same perspective
        if stance_a.lower() != stance_b.lower() and "unclear" not in (stance_a.lower(), stance_b.lower()):
            return PerspectiveMatcher.DIFFERENT

        # Rule 2: Dense retrieval gate (Stage 1)
        if cosine_sim < tau_cosine:
            return PerspectiveMatcher.DIFFERENT

        # Rule 3: Second-stage semantic equivalence verification (Stage 2)
        if reasoning_entailment_prob >= theta_verify:
            return PerspectiveMatcher.EQUIVALENT

        # High similarity but sub-threshold verification implies related but distinct reasoning
        return PerspectiveMatcher.RELATED

