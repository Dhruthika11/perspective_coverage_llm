# Common-Claim-Set Perspective Coverage Across All 10 Models (Exact Same 65 Claims)

> **Experimental Rigor:** All 10 models evaluated on the exact same 65 claims using the identical frozen verifier M0. Common denominator = 328 gold perspective clusters.

| Model | Claims | Viewpoints | Common Gold Clusters | M0 Matched | M0 Micro Cov | M0 Macro Cov | M_soft Macro Cov | Arg Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2_5_32b_instruct` | 65 | 1382 | 586 | 433 | 73.89% (433/586) | 82.34% | 79.16% | 0.313 (433/1382) |
| `qwen2_5_1_5b_instruct` | 65 | 1027 | 586 | 389 | 66.38% (389/586) | 78.46% | 75.36% | 0.379 (389/1027) |
| `qwen2_5_3b_instruct` | 65 | 844 | 586 | 375 | 63.99% (375/586) | 75.14% | 72.95% | 0.444 (375/844) |
| `olmo_2_1124_7b_instruct` | 65 | 325 | 586 | 222 | 37.88% (222/586) | 51.61% | 50.58% | 0.683 (222/325) |
| `llama_3_2_3b_instruct` | 65 | 325 | 586 | 219 | 37.37% (219/586) | 51.17% | 48.63% | 0.674 (219/325) |
| `olmo_2_1124_13b_instruct` | 65 | 325 | 586 | 208 | 35.49% (208/586) | 47.77% | 47.25% | 0.640 (208/325) |
| `gemma_2_9b_it` | 65 | 325 | 586 | 203 | 34.64% (203/586) | 46.63% | 44.95% | 0.625 (203/325) |
| `gemma_2_27b_it` | 65 | 325 | 586 | 200 | 34.13% (200/586) | 44.78% | 43.54% | 0.615 (200/325) |
| `qwen2_5_7b_instruct` | 65 | 325 | 586 | 197 | 33.62% (197/586) | 43.95% | 42.62% | 0.606 (197/325) |
| `llama_3_1_8b_instruct` | 65 | 325 | 586 | 196 | 33.45% (196/586) | 43.39% | 42.57% | 0.603 (196/325) |
