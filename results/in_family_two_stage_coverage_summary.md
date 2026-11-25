# Final Two-Stage Perspective Coverage Report: In-Family Generator Models
> **Evaluator:** Frozen Main V5 Data-Only Verifier ($T^* = 1.9104$, $\theta_{F1}^* = 0.52$, $\theta_{\text{prec90}}^* = 0.83$)  
> **Stage 1 Retrieval:** BAAI/bge-large-en-v1.5 (Cosine $\tau \ge 0.75$)  
> **Scope:** Generator Families used to train V5 (Qwen, LLaMA, Gemma, OLMo).

---

## Master Cross-Model Perspective Coverage Comparison

| Generator Model | Family | Params (B) | Single-Stage Cosine ($\tau \ge 0.75$) | True Two-Stage Coverage ($\theta_{F1}^* = 0.52$) | High-Precision Coverage ($\theta^* = 0.83$) | False-Match Drop | Support Cov | Undermine Cov | Stance Gap (Sup - Und) |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B** | Qwen | 7.0B | `97.9%` | **`53.2%`** | `10.7%` | `45.6%` | `48.7%` | `48.0%` | `+0.7%` |
| **Llama-3.1-8B** | Llama | 8.0B | `97.9%` | **`56.3%`** | `10.4%` | `42.5%` | `52.2%` | `54.2%` | `-1.9%` |
| **Gemma-2-9B** | Gemma | 9.0B | `97.9%` | **`58.3%`** | `12.0%` | `40.4%` | `54.7%` | `53.6%` | `+1.1%` |
| **OLMo-2-7B** | OLMo | 7.0B | `97.9%` | **`59.5%`** | `11.4%` | `39.3%` | `54.0%` | `56.4%` | `-2.5%` |
| **Qwen2.5-1.5B** | Qwen | 1.5B | `97.9%` | **`59.1%`** | `12.5%` | `39.6%` | `55.7%` | `53.6%` | `+2.1%` |
| **Qwen2.5-3B** | Qwen | 3.0B | `97.9%` | **`57.4%`** | `11.9%` | `41.3%` | `53.5%` | `51.9%` | `+1.6%` |
| **Qwen2.5-32B** | Qwen | 32.0B | `97.9%` | **`54.9%`** | `9.9%` | `44.0%` | `49.8%` | `52.4%` | `-2.6%` |

---

## Key Scientific Observations
1. **The Overestimation Elimination:** Single-stage dense retrieval (BGE cosine similarity) overestimates true coverage by falsely matching topical overlap to entailed coverage. Stage 2 V5 verification filters these out, establishing the true semantic coverage.
2. **Model Family Comparison:** Evaluates how effectively Qwen, LLaMA, Gemma, and OLMo capture diverse human viewpoints across balanced topics.
3. **Stance Bias Diagnostic:** Quantifies the structural asymmetry between supporting and undermining perspectives for each architecture.
