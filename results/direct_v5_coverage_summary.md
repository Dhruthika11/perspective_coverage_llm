# Direct V5 Perspective Coverage Report: In-Family Generator Models
> **Evaluator:** Frozen Main V5 Argument-NLI Cross-Encoder ($T^* = 1.9104$)
> **Architecture:** Pure Single-Stage Verification (**No Stage 1 Dense Retrieval Filter**)
> **Operating Points:** Calibrated Optimal $F_1$ ($\theta_{F1}^* = 0.52$) & High-Precision ($\theta_{\text{prec90}}^* = 0.83$)

---

## Master Cross-Model Perspective Coverage Comparison

| Generator Model | Family | Params (B) | Atomic Coverage ($\theta_{F1}^* = 0.52$) | Cluster Coverage ($\theta_{F1}^* = 0.52$) | High-Precision Coverage ($\theta^* = 0.83$) | Support Cov | Undermine Cov | Stance Gap (Sup - Und) |
|---|---|---|---|---|---|---|---|---|

---

## Key Methodological Advantages of Removing Stage 1
1. **Elimination of the 94% Retrieval Bottleneck:** BGE cosine similarity at $\tau \ge 0.75$ prematurely dropped 94.15% of valid long-form LLM perspective pairs due to vector length dilution. Direct verification allows the verifier to inspect every candidate directly.
2. **Zero Vector Embedding Overhead:** No need to compute, cache, or load 1024-dim BGE embeddings; eliminates all embedding dimension mismatch bugs and library kernel dependency issues.
3. **True Cross-Encoder Perspective Entailment:** Within a given claim, candidate perspectives average only 8–12 statements. Evaluating all pairs directly through DeBERTa-v3 cross-encoder is computationally fast and delivers ground-truth discursive coverage.