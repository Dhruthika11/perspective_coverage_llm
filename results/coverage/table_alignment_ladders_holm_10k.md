# Post-Training Alignment Ladders with Holm-Bonferroni Correction (m = 11)

Evaluates paired coverage gains across post-training alignment stages across all 907 claims with 10,000 paired bootstrap resamples.

| Model Family | Transition | Model From | Model To | Coverage Gain (%) | 95% Bootstrap CI of Gain | Paired t-stat | Unadj t-p | Holm-adj t-p | Wilcoxon W | Holm-adj W-p |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLMo 2 1B (0425)** | `BASE -> SFT` | `olmo_2_0425_1b` | `olmo_2_0425_1b_sft` | **+17.94%** | [+15.39%, +20.41%] | 13.985 | 2.20e-40 | **2.20e-39** (***) | 85615.5 | **2.29e-41** |
| **OLMo 2 13B** | `BASE -> SFT` | `olmo_2_1124_13b` | `olmo_2_1124_13b_sft` | **+18.02%** | [+15.48%, +20.55%] | 14.280 | 7.10e-42 | **7.81e-41** (***) | 87564.0 | **1.77e-40** |
| **OLMo 2 7B** | `BASE -> SFT` | `olmo_2_1124_7b` | `olmo_2_1124_7b_sft` | **+9.57%** | [+7.20%, +11.97%] | 7.780 | 1.98e-14 | **1.78e-13** (***) | 115826.0 | **1.66e-17** |
| **OLMo 2 32B (0325)** | `BASE -> INSTRUCT` | `olmo_2_0325_32b` | `olmo_2_0325_32b_instruct` | **-6.41%** | [-8.56%, -4.24%] | -5.831 | 7.67e-09 | **6.14e-08** (***) | 124482.5 | **2.86e-09** |
| **OLMo 2 1B (0425)** | `SFT -> INSTRUCT` | `olmo_2_0425_1b_sft` | `olmo_2_0425_1b_instruct` | **+3.28%** | [+1.73%, +4.85%] | 4.185 | 3.13e-05 | **2.18e-04** (***) | 82709.0 | **8.16e-08** |
| **OLMo 2 32B (0325)** | `INSTRUCT -> DPO` | `olmo_2_0325_32b_instruct` | `olmo_2_0325_32b_dpo` | **+2.72%** | [+1.48%, +4.00%] | 4.186 | 3.12e-05 | **2.18e-04** (***) | 73853.0 | **9.24e-05** |
| **OLMo 2 7B** | `INSTRUCT -> DPO` | `olmo_2_1124_7b_instruct` | `olmo_2_1124_7b_dpo` | **+2.38%** | [+0.93%, +3.86%] | 3.233 | 1.27e-03 | **6.35e-03** (**) | 98253.0 | **1.80e-03** |
| **OLMo 2 13B** | `SFT -> INSTRUCT` | `olmo_2_1124_13b_sft` | `olmo_2_1124_13b_instruct` | **-2.65%** | [-4.67%, -0.63%] | -2.591 | 9.73e-03 | **3.89e-02** (*) | 130891.0 | **2.90e-02** |
| **OLMo 2 7B** | `SFT -> INSTRUCT` | `olmo_2_1124_7b_sft` | `olmo_2_1124_7b_instruct` | **+1.46%** | [-0.36%, +3.25%] | 1.583 | 1.14e-01 | **3.41e-01** (ns) | 122563.5 | **3.32e-02** |
| **OLMo 2 1B (0425)** | `INSTRUCT -> DPO` | `olmo_2_0425_1b_instruct` | `olmo_2_0425_1b_dpo` | **-1.17%** | [-2.80%, +0.41%] | -1.430 | 1.53e-01 | **3.41e-01** (ns) | 107598.5 | **3.32e-02** |
| **OLMo 2 13B** | `INSTRUCT -> DPO` | `olmo_2_1124_13b_instruct` | `olmo_2_1124_13b_dpo` | **+0.94%** | [-0.74%, +2.70%] | 1.086 | 2.78e-01 | **3.41e-01** (ns) | 127420.5 | **5.86e-01** |
