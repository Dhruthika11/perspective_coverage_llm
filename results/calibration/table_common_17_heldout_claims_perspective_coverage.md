# Common-Claim-Set Perspective Coverage Across All 10 Models (Strictly Held-Out 17 Claims)

> **Experimental Rigor:** All 10 models evaluated on the exact same 17 held-out claims (zero overlap with verifier train/cal claims). Common denominator = 84 gold perspective clusters.

| Model | Claims | Viewpoints | Common Gold Clusters | M0 Matched | M0 Micro Cov | M0 Macro Cov | M_soft Macro Cov | Arg Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2_5_32b_instruct` | 17 | 350 | 173 | 119 | 68.79% (119/173) | 78.93% | 73.83% | 0.340 (119/350) |
| `qwen2_5_1_5b_instruct` | 17 | 287 | 173 | 110 | 63.58% (110/173) | 74.70% | 69.11% | 0.383 (110/287) |
| `qwen2_5_3b_instruct` | 17 | 219 | 173 | 97 | 56.07% (97/173) | 65.14% | 62.62% | 0.443 (97/219) |
| `llama_3_2_3b_instruct` | 17 | 85 | 173 | 60 | 34.68% (60/173) | 45.36% | 42.23% | 0.706 (60/85) |
| `olmo_2_1124_7b_instruct` | 17 | 85 | 173 | 60 | 34.68% (60/173) | 44.79% | 43.81% | 0.706 (60/85) |
| `olmo_2_1124_13b_instruct` | 17 | 85 | 173 | 57 | 32.95% (57/173) | 41.64% | 39.68% | 0.671 (57/85) |
| `gemma_2_27b_it` | 17 | 85 | 173 | 55 | 31.79% (55/173) | 39.05% | 37.87% | 0.647 (55/85) |
| `gemma_2_9b_it` | 17 | 85 | 173 | 55 | 31.79% (55/173) | 39.05% | 36.47% | 0.647 (55/85) |
| `llama_3_1_8b_instruct` | 17 | 85 | 173 | 52 | 30.06% (52/173) | 36.69% | 36.69% | 0.612 (52/85) |
| `qwen2_5_7b_instruct` | 17 | 85 | 173 | 50 | 28.90% (50/173) | 34.45% | 32.98% | 0.588 (50/85) |
