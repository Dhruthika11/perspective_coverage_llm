# 10-Model Kuhn-Munkres Matching Robustness Table (Held-Out Test Partition: 17 Claims)

| Model Architecture | Claims | Viewpoints | Gold Clusters (Denom) | Matched vA | Matched vC | Shared | Jaccard | Micro Cov vA | Micro Cov vC | Macro Cov vA | Macro Cov vC | Delta Macro | Arg Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemma_2_27b_it` | 9 | 11 | 14 | 5 | 5 | 5 | 100.00% | 35.71% (5/14) | 35.71% (5/14) | 27.78% | 27.78% | +0.00% | 0.455 (5/11) |
| `olmo_2_1124_13b_instruct` | 6 | 8 | 8 | 3 | 3 | 3 | 100.00% | 37.50% (3/8) | 37.50% (3/8) | 25.00% | 25.00% | +0.00% | 0.375 (3/8) |
| `olmo_2_1124_7b_instruct` | 6 | 9 | 13 | 3 | 3 | 3 | 100.00% | 23.08% (3/13) | 23.08% (3/13) | 29.17% | 29.17% | +0.00% | 0.333 (3/9) |
| `qwen2_5_1_5b_instruct` | 4 | 4 | 4 | 2 | 2 | 2 | 100.00% | 50.00% (2/4) | 50.00% (2/4) | 50.00% | 50.00% | +0.00% | 0.500 (2/4) |
| `qwen2_5_32b_instruct` | 3 | 6 | 6 | 0 | 0 | 0 | 100.00% | 0.00% (0/6) | 0.00% (0/6) | 0.00% | 0.00% | +0.00% | 0.000 (0/6) |
| `qwen2_5_3b_instruct` | 2 | 2 | 2 | 0 | 0 | 0 | 100.00% | 0.00% (0/2) | 0.00% (0/2) | 0.00% | 0.00% | +0.00% | 0.000 (0/2) |
| `qwen2_5_7b_instruct` | 10 | 13 | 14 | 2 | 2 | 2 | 100.00% | 14.29% (2/14) | 14.29% (2/14) | 15.00% | 15.00% | +0.00% | 0.154 (2/13) |
| `gemma_2_9b_it` | 7 | 9 | 10 | 4 | 5 | 4 | 80.00% | 40.00% (4/10) | 50.00% (5/10) | 50.00% | 64.29% | +14.29% | 0.556 (5/9) |
| `llama_3_1_8b_instruct` | 6 | 7 | 7 | 3 | 2 | 2 | 66.67% | 42.86% (3/7) | 28.57% (2/7) | 50.00% | 33.33% | -16.67% | 0.286 (2/7) |
| `llama_3_2_3b_instruct` | 6 | 9 | 8 | 1 | 0 | 0 | 0.00% | 12.50% (1/8) | 0.00% (0/8) | 16.67% | 0.00% | -16.67% | 0.000 (0/9) |

> [!NOTE]
> **Held-Out Test Partition ($N = 17$ Claims, 88 Pairs):**
> 1. **Model-Specific Sum ($N = 22$ edges):** Sum of independent matches across the 10 models.
> 2. **Pooled Multi-Model Matching ($N = 21$ clusters):** Total unique gold clusters covered across the 17 claims.
> 3. **Raw Labeled Pairs ($N = 23$ Class 2 pairs):** Total pairs annotated as Class 2 in held-out test partition.
