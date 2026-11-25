# Attainable Coverage Ceiling & Normalized Macro Coverage (907 Claims)

Normalized coverage directly measures the proportion of attainable perspective clusters captured given the model's viewpoint budget: $\text{Cov}_{norm} = \frac{M_c}{\min(K_c, N_c)}$.

| Model Architecture | Viewpoint Budget | Raw Macro Coverage (%) | Attainable Ceiling (%) | Normalized Macro Coverage (Mean [95% CI]) | Argumentative Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `qwen2_5_32b_instruct` | 18630 | 81.41% | 98.96% | **82.14%** [80.53%, 83.72%] | **0.211** |
| `qwen2_5_1_5b_instruct` | 14209 | 80.39% | 98.30% | **81.74%** [80.11%, 83.31%] | **0.266** |
| `qwen2_5_3b_instruct` | 11021 | 78.99% | 97.14% | **81.35%** [79.71%, 82.96%] | **0.334** |
| `llama_3_2_3b_instruct` | 4535 | 54.90% | 85.32% | **67.13%** [65.13%, 69.12%] | **0.512** |
| `olmo_2_1124_7b_instruct` | 4535 | 52.79% | 85.32% | **65.16%** [63.09%, 67.25%] | **0.500** |
| `gemma_2_9b_it` | 4535 | 50.26% | 85.32% | **62.01%** [59.98%, 64.07%] | **0.475** |
| `olmo_2_1124_13b_instruct` | 4535 | 49.15% | 85.32% | **61.14%** [59.02%, 63.26%] | **0.471** |
| `gemma_2_27b_it` | 4535 | 47.90% | 85.32% | **59.24%** [57.14%, 61.30%] | **0.450** |
| `llama_3_1_8b_instruct` | 4535 | 46.41% | 85.32% | **58.00%** [55.84%, 60.17%] | **0.449** |
| `qwen2_5_7b_instruct` | 4536 | 45.85% | 85.33% | **57.43%** [55.34%, 59.54%] | **0.445** |
