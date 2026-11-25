"""
Standalone script to run a single (Premise, Hypothesis) pair through:
  Stage 1: BGE Dense Embedding Cosine Retrieval (tau = 0.75)
  Stage 2: Calibrated V5 DeBERTa Cross-Encoder Verifier (T* = 1.9104, theta* = 0.52 / 0.83)

Can be executed directly on HPC (GPU) or locally (CPU).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

def compute_bge_cosine_similarity(premise: str, hypothesis: str, bge_model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu") -> float:
    """Computes cosine similarity using BGE-Large embeddings (with CLS pooling and L2 normalization)."""
    print(f"\n[Stage 1] Loading BGE Embedder: {bge_model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(bge_model_name)
    model = AutoModel.from_pretrained(bge_model_name).to(device)
    model.eval()

    texts = [premise, hypothesis]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # BGE official representation: CLS token representation + L2 normalize
        cls_embs = outputs.last_hidden_state[:, 0]
        norm_embs = F.normalize(cls_embs, p=2, dim=1)

    cosine_sim = torch.dot(norm_embs[0], norm_embs[1]).item()
    return float(cosine_sim)

def compute_v5_verification_prob(premise: str, hypothesis: str, v5_model_dir: str = "models/ensemble_data_only", temperature: float = 1.9104, device: str = "cpu") -> float:
    """Computes calibrated entailment probability using V5 DeBERTa-v3 cross-encoder."""
    # Check for seed checkpoints or single model
    candidate_paths = [
        os.path.join(v5_model_dir, "seed_42"),
        v5_model_dir,
        "models/argument_nli_deberta_v3_v3",
        "cross-encoder/nli-deberta-v3-base"
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p) and (os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(os.path.join(p, "pytorch_model.bin"))):
            model_path = p
            break
        elif p.startswith("cross-encoder/"):
            model_path = p
            break

    print(f"\n[Stage 2] Loading V5 Verifier from: {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(premise, hypothesis, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        # Apply temperature scaling: logits / T*
        scaled_logits = logits / max(temperature, 1e-4)
        probs = F.softmax(scaled_logits, dim=-1)

        # Entailment is index 1 for 2-class or index 1 for standard NLI (contradiction=0, entailment=1, neutral=2)
        entail_idx = 1 if scaled_logits.shape[-1] >= 2 else 0
        p_entail = probs[0, entail_idx].item()

    return float(p_entail)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single argument-perspective pair on Stage 1 (BGE) and Stage 2 (V5).")
    parser.add_argument("--premise", type=str, default="Social media causes people to spend less time interacting face-to-face.", help="Argument premise text")
    parser.add_argument("--hypothesis", type=str, default="The use of social media is correlated with personality and brain disorders.", help="Perspective hypothesis text")
    parser.add_argument("--bge_model", type=str, default="BAAI/bge-large-en-v1.5", help="BGE model path or HuggingFace ID")
    parser.add_argument("--v5_model_dir", type=str, default="models/ensemble_data_only", help="Path to V5 verifier weights")
    parser.add_argument("--temperature", type=float, default=1.9104, help="Calibrated temperature parameter")
    parser.add_argument("--tau", type=float, default=0.75, help="Stage 1 cosine threshold")
    parser.add_argument("--theta_f1", type=float, default=0.52, help="Stage 2 optimal F1 threshold")
    parser.add_argument("--theta_prec90", type=float, default=0.83, help="Stage 2 high-precision threshold")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 85)
    print("EVALUATING PAIR THROUGH TWO-STAGE PERSPECTIVE COVERAGE PIPELINE")
    print("=" * 85)
    print(f"Device: {device}")
    print(f"Argument (Premise):\n  \"{args.premise}\"")
    print(f"\nPerspective (Hypothesis):\n  \"{args.hypothesis}\"")
    print("-" * 85)

    # 1. Stage 1: Dense Retrieval Embedding
    cosine_sim = compute_bge_cosine_similarity(args.premise, args.hypothesis, args.bge_model, device)
    stage1_passed = cosine_sim >= args.tau

    print(f"\n--- STAGE 1 RESULTS (BGE Embedding Cosine Retrieval) ---")
    print(f"Cosine Similarity : {cosine_sim:.4f}")
    print(f"Stage 1 Threshold : {args.tau:.2f}")
    print(f"Stage 1 Decision  : {'PASSED (Proceeds to Stage 2)' if stage1_passed else 'FILTERED OUT (Dropped by Stage 1)'}")

    # 2. Stage 2: Cross-Encoder NLI Verification
    p_entail = compute_v5_verification_prob(args.premise, args.hypothesis, args.v5_model_dir, args.temperature, device)
    stage2_passed_f1 = p_entail >= args.theta_f1
    stage2_passed_prec90 = p_entail >= args.theta_prec90

    print(f"\n--- STAGE 2 RESULTS (V5 DeBERTa Cross-Encoder Verifier) ---")
    print(f"Calibrated P(entail) : {p_entail:.4f}")
    print(f"Optimal F1 Decision  (theta={args.theta_f1:.2f}) : {'COVERED' if stage2_passed_f1 else 'REJECTED'}")
    print(f"High-Precision Dec.  (theta={args.theta_prec90:.2f}) : {'COVERED' if stage2_passed_prec90 else 'REJECTED'}")

    print("\n" + "=" * 85)
    print("FINAL SUMMARY")
    print("=" * 85)
    print(f"Stage 1 BGE Cosine Similarity : {cosine_sim:.4f} ({'Passes >= 0.75' if stage1_passed else 'Fails < 0.75'})")
    print(f"Stage 2 Calibrated Entailment : {p_entail:.4f} ({'Covered >= 0.52' if stage2_passed_f1 else 'Rejected < 0.52'})")
    print(f"Overall Classification        : {'FALSE ALARM HARD NEGATIVE: Caught by Stage 2' if (stage1_passed and not stage2_passed_f1) else ('TRUE COVERAGE' if stage2_passed_f1 else 'REJECTED')}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
