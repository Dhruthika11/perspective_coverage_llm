# Common 17-Claim Perspective Coverage Benchmark (All 10 Models)

Evaluated strictly on the 17 held-out test claims with identical common denominator ($N_{gold} = 173$).

| Model Architecture | Test Claims | Gold Clusters (Common Denom) | Viewpoints Generated | M0 Matched Clusters | M0 Micro Coverage (%) | M0 Macro Coverage (%) | M_soft Macro Coverage (%) | Argumentative Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_32b_instruct` | 17 | 173 | 350 | 119 | 68.79% (119/173) | 78.93% | 73.83% | 0.340 (119/350) |
| `qwen2_5_1_5b_instruct` | 17 | 173 | 287 | 110 | 63.58% (110/173) | 74.70% | 69.11% | 0.383 (110/287) |
| `qwen2_5_3b_instruct` | 17 | 173 | 219 | 97 | 56.07% (97/173) | 65.14% | 62.62% | 0.443 (97/219) |
| `llama_3_2_3b_instruct` | 17 | 173 | 85 | 60 | 34.68% (60/173) | 45.36% | 42.23% | 0.706 (60/85) |
| `olmo_2_1124_7b_instruct` | 17 | 173 | 85 | 60 | 34.68% (60/173) | 44.79% | 43.81% | 0.706 (60/85) |
| `olmo_2_1124_13b_instruct` | 17 | 173 | 85 | 57 | 32.95% (57/173) | 41.64% | 39.68% | 0.671 (57/85) |
| `gemma_2_27b_it` | 17 | 173 | 85 | 55 | 31.79% (55/173) | 39.05% | 37.87% | 0.647 (55/85) |
| `gemma_2_9b_it` | 17 | 173 | 85 | 55 | 31.79% (55/173) | 39.05% | 36.47% | 0.647 (55/85) |
| `llama_3_1_8b_instruct` | 17 | 173 | 85 | 52 | 30.06% (52/173) | 36.69% | 36.69% | 0.612 (52/85) |
| `qwen2_5_7b_instruct` | 17 | 173 | 85 | 50 | 28.90% (50/173) | 34.45% | 32.98% | 0.588 (50/85) |
