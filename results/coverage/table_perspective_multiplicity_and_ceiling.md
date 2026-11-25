# Perspective Multiplicity Stratification & Attainable Ceilings (907 Claims)

Evaluates how fixed-budget generation models perform across low, moderate, and high perspective multiplicity claims.

| Perspective Multiplicity Tier | Claims ($N$) | Model Architecture | Raw Macro Coverage (%) | Attainable Ceiling (%) | Normalized Coverage (%) | Argumentative Efficiency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| Low Multiplicity (1-3 clusters) | 346 | `llama_3_2_3b_instruct` | 61.99% | 100.00% | **61.99%** | **0.249** |
| Low Multiplicity (1-3 clusters) | 346 | `olmo_2_1124_7b_instruct` | 60.16% | 100.00% | **60.16%** | **0.246** |
| Low Multiplicity (1-3 clusters) | 346 | `qwen2_5_7b_instruct` | 51.11% | 100.00% | **51.11%** | **0.208** |
| Moderate Multiplicity (4-6 clusters) | 313 | `llama_3_2_3b_instruct` | 57.34% | 95.37% | **60.38%** | **0.552** |
| Moderate Multiplicity (4-6 clusters) | 313 | `olmo_2_1124_7b_instruct` | 52.95% | 95.37% | **55.85%** | **0.512** |
| Moderate Multiplicity (4-6 clusters) | 313 | `qwen2_5_7b_instruct` | 45.72% | 95.37% | **48.21%** | **0.442** |
| High Multiplicity (>=7 clusters) | 248 | `llama_3_2_3b_instruct` | 41.92% | 52.17% | **82.82%** | **0.828** |
| High Multiplicity (>=7 clusters) | 248 | `olmo_2_1124_7b_instruct` | 42.31% | 52.17% | **83.87%** | **0.839** |
| High Multiplicity (>=7 clusters) | 248 | `qwen2_5_7b_instruct` | 38.68% | 52.18% | **77.90%** | **0.779** |
