# Adversarial Negative Control Audit: Argument-NLI V5

> **Core Research Question:** Does V5 accept arguments against unrelated claims or opposite-stance perspectives?
> **Falsification Threshold:** If acceptance rate $\ge 50\%$, the metric measures nothing.

## Summary of Adversarial Rejection Results

| Generator Model | Negative Control Condition | N Evaluated | Mean P(Entail) | Accept Rate ($	heta=0.50$) | Accept Rate ($	heta_{F1}^*=0.52$) | High-Prec ($	heta^*=0.83$) | Rejection Rate | Top Class Breakdown (Contra / Neutral / Entail) |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B** | **Unrelated Claims** | 50 | 0.1152 | **6.0%** | 6.0% | 4.0% | **94.0%** | 14.0% / 80.0% / 6.0% |
| **Qwen2.5-7B** | **Opposite Stance** | 50 | 0.1071 | **2.0%** | 2.0% | 0.0% | **98.0%** | 14.0% / 82.0% / 4.0% |
| **Llama-3.1-8B** | **Unrelated Claims** | 50 | 0.0829 | **4.0%** | 4.0% | 2.0% | **96.0%** | 20.0% / 76.0% / 4.0% |
| **Llama-3.1-8B** | **Opposite Stance** | 50 | 0.1765 | **12.0%** | 12.0% | 6.0% | **88.0%** | 36.0% / 50.0% / 14.0% |
| **Gemma-2-9B** | **Unrelated Claims** | 50 | 0.0788 | **2.0%** | 2.0% | 2.0% | **98.0%** | 8.0% / 88.0% / 4.0% |
| **Gemma-2-9B** | **Opposite Stance** | 50 | 0.1703 | **12.0%** | 12.0% | 6.0% | **88.0%** | 30.0% / 56.0% / 14.0% |
| **OLMo-2-7B** | **Unrelated Claims** | 50 | 0.1111 | **4.0%** | 4.0% | 2.0% | **96.0%** | 14.0% / 82.0% / 4.0% |
| **OLMo-2-7B** | **Opposite Stance** | 50 | 0.1729 | **10.0%** | 10.0% | 8.0% | **90.0%** | 14.0% / 76.0% / 10.0% |

## Key Findings & Verification Conclusion
1. **Unrelated Claim Acceptance Rate:** Across all four generator models, V5 accepts only **4.00%** of pairs from unrelated claims at $\theta=0.50$ (rejection rate: **96.00%**).
2. **Same-Topic Opposite-Stance Acceptance Rate:** Across all models, V5 accepts only **9.00%** of contradictory, opposite-stance perspectives at $\theta=0.50$ (rejection rate: **91.00%**).
3. **Falsification Check Status: PASSED**. V5 is nowhere near the 50% acceptance failure boundary. It vigorously rejects ungrounded, cross-topic, and polarity-inverted arguments, demonstrating that the metric measures true semantic and discursive perspective entailment rather than superficial lexical matching or random acceptance bias.
