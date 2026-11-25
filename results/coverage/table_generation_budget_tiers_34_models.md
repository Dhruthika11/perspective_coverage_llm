# Complete 34-Model Generation Budget Tier Taxonomy (907 Claims)

This table categorizes all 34 evaluated open-source models into four generation budget tiers, documenting total viewpoints, mean viewpoints per claim ($K_c$), standard deviation, and explaining any deviations from the standardized $K=5$ prompt.

| Budget Tier | Model Architecture | Total Viewpoints | Mean $K_c$ | Min $K_c$ | Max $K_c$ | Exact 5 VPs (%) | $M_0$ Macro Cov (%) | Argumentative Efficiency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Unconstrained High-Volume** | `qwen2_5_32b_instruct` | 18,630 | 20.54 | 7 | 53 | 0.0% | **81.41%** | 0.211 |
| **Tier 1: Unconstrained High-Volume** | `smollm2_135m_instruct` | 16,827 | 18.55 | 0 | 57 | 0.3% | **62.45%** | 0.170 |
| **Tier 1: Unconstrained High-Volume** | `qwen2_5_1_5b_instruct` | 14,209 | 15.67 | 5 | 39 | 0.1% | **80.39%** | 0.266 |
| **Tier 1: Unconstrained High-Volume** | `qwen2_5_3b_instruct` | 11,021 | 12.15 | 6 | 41 | 0.0% | **78.99%** | 0.334 |
| **Tier 2: Elevated / Base Models** | `smollm2_360m_instruct` | 7,557 | 8.33 | 0 | 30 | 10.4% | **61.67%** | 0.358 |
| **Tier 2: Elevated / Base Models** | `olmo_2_0425_1b` | 7,448 | 8.21 | 0 | 53 | 13.8% | **30.54%** | 0.182 |
| **Tier 2: Elevated / Base Models** | `olmo_2_1124_7b` | 7,414 | 8.17 | 1 | 43 | 23.8% | **41.77%** | 0.241 |
| **Tier 2: Elevated / Base Models** | `olmo_2_1124_13b` | 7,129 | 7.86 | 2 | 34 | 16.3% | **33.78%** | 0.204 |
| **Tier 2: Elevated / Base Models** | `olmo_2_0325_32b` | 6,231 | 6.87 | 4 | 54 | 45.4% | **55.01%** | 0.375 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_1124_7b_dpo` | 4,535 | 5.00 | 5 | 5 | 100.0% | **55.18%** | **0.517** |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `llama_3_2_3b_instruct` | 4,535 | 5.00 | 5 | 5 | 100.0% | **54.90%** | **0.512** |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_1124_7b_instruct` | 4,535 | 5.00 | 5 | 5 | 100.0% | **52.79%** | **0.500** |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `qwen2_5_0_5b_instruct` | 4,666 | 5.14 | 5 | 11 | 92.2% | **52.45%** | 0.478 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_1124_13b_sft` | 4,532 | 5.00 | 3 | 5 | 99.8% | **51.80%** | 0.488 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_0425_1b_instruct` | 4,540 | 5.01 | 5 | 6 | 99.4% | **51.76%** | 0.495 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `qwen2_5_14b_instruct` | 4,550 | 5.02 | 5 | 6 | 98.3% | **51.66%** | 0.483 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_1124_7b_sft` | 4,535 | 5.00 | 5 | 5 | 100.0% | **51.34%** | 0.491 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_0325_32b_dpo` | 4,535 | 5.00 | 5 | 5 | 100.0% | **51.31%** | 0.487 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_0425_1b_dpo` | 4,536 | 5.00 | 5 | 6 | 99.9% | **50.59%** | 0.484 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `gemma_2_9b_it` | 4,535 | 5.00 | 5 | 5 | 100.0% | **50.26%** | 0.475 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_1124_13b_dpo` | 4,535 | 5.00 | 5 | 5 | 100.0% | **50.10%** | 0.476 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_1124_13b_instruct`| 4,535 | 5.00 | 5 | 5 | 100.0% | **49.15%** | 0.471 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `gemma_2_2b_it` | 4,535 | 5.00 | 5 | 5 | 100.0% | **48.83%** | 0.469 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_0325_32b_instruct`| 4,534 | 5.00 | 4 | 5 | 99.9% | **48.60%** | 0.470 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmo_2_0425_1b_sft` | 4,510 | 4.97 | 3 | 5 | 97.5% | **48.48%** | 0.470 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `gemma_2_27b_it` | 4,535 | 5.00 | 5 | 5 | 100.0% | **47.90%** | 0.450 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `llama_3_1_8b_instruct` | 4,535 | 5.00 | 5 | 5 | 100.0% | **46.41%** | 0.449 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `qwen2_5_7b_instruct` | 4,536 | 5.00 | 5 | 6 | 99.9% | **45.85%** | 0.445 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `olmoe_1b_7b_0924_instruct`| 4,556 | 5.02 | 4 | 8 | 97.7% | **43.59%** | 0.431 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `llama_3_2_1b_instruct` | 4,559 | 5.03 | 4 | 8 | 96.9% | **43.44%** | 0.415 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `smollm2_1_7b_instruct` | 4,464 | 4.92 | 2 | 9 | 86.9% | **41.26%** | 0.416 |
| **Tier 3: Standardized 5 Viewpoints/Claim** | `qwen2_5_7b_instruct_p2` | 4,534 | 5.00 | 4 | 6 | 99.7% | **31.13%** | 0.305 |
| **Tier 4: Truncated / Partial Generation** | `qwen2_5_14b_instruct_p2` | 2,877 | 3.17 | 0 | 6 | 10.6% | **32.35%** | 0.467 |
| **Tier 4: Truncated / Partial Generation** | `qwen2_5_3b_instruct_p2` | 1,541 | 1.70 | 0 | 4 | 0.0% | **23.93%** | **0.588** |
