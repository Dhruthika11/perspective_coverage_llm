# Untouched Claims Evaluation: Full 907 Claims vs. 842 Untouched Claims

Validates that verifier threshold selection on 16 calibration claims induces zero overfitting on external claims.

| Model Architecture | Full Corpus Macro (907 Claims) | Untouched Macro (842 Claims) | Delta (Untouched - Full) | Original Held-Out Macro (17 Claims) | Full Micro Cov (907) | Untouched Micro Cov (842) | Efficiency (907) | Efficiency (842) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2_5_32b_instruct` | **81.41%** | **81.34%** | -0.07% | 78.93% | 77.21% | 77.64% | 0.211 | 0.203 |
| `qwen2_5_1_5b_instruct` | **80.39%** | **80.54%** | +0.15% | 74.70% | 74.17% | 75.18% | 0.266 | 0.257 |
| `qwen2_5_3b_instruct` | **78.99%** | **79.28%** | +0.30% | 65.14% | 72.23% | 73.30% | 0.334 | 0.325 |
| `llama_3_2_3b_instruct` | **54.90%** | **55.19%** | +0.29% | 45.36% | 45.55% | 46.62% | 0.512 | 0.499 |
| `olmo_2_1124_7b_instruct` | **52.79%** | **52.88%** | +0.09% | 44.79% | 44.51% | 45.38% | 0.500 | 0.486 |
| `gemma_2_9b_it` | **50.26%** | **50.54%** | +0.28% | 39.05% | 42.30% | 43.29% | 0.475 | 0.464 |
| `olmo_2_1124_13b_instruct` | **49.15%** | **49.26%** | +0.11% | 41.64% | 41.94% | 42.78% | 0.471 | 0.458 |
| `gemma_2_27b_it` | **47.90%** | **48.15%** | +0.24% | 39.05% | 40.06% | 40.83% | 0.450 | 0.437 |
| `llama_3_1_8b_instruct` | **46.41%** | **46.64%** | +0.23% | 36.69% | 39.92% | 40.76% | 0.449 | 0.437 |
| `qwen2_5_7b_instruct` | **45.85%** | **46.00%** | +0.15% | 34.45% | 39.63% | 40.41% | 0.445 | 0.433 |
