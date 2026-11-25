# Post-Training Alignment Ladders: DPO vs. Instruct vs. SFT vs. Base (907 Claims)

Evaluates paired coverage gains across post-training alignment stages across all 907 claims.

| Model Family | Post-Training Transition | Model From | Model To | From Macro Cov (%) | To Macro Cov (%) | Coverage Gain (%) | 95% Bootstrap CI of Gain | p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLMo 2 7B** | `BASE -> SFT` | `olmo_2_1124_7b` | `olmo_2_1124_7b_sft` | 41.77% | 51.34% | **+9.57%** | [+7.11%, +11.92%] | 1.98e-14 (***) |
| **OLMo 2 7B** | `SFT -> INSTRUCT` | `olmo_2_1124_7b_sft` | `olmo_2_1124_7b_instruct` | 51.34% | 52.79% | **+1.46%** | [-0.39%, +3.31%] | 1.14e-01 (ns) |
| **OLMo 2 7B** | `INSTRUCT -> DPO` | `olmo_2_1124_7b_instruct` | `olmo_2_1124_7b_dpo` | 52.79% | 55.18% | **+2.38%** | [+0.85%, +3.83%] | 1.27e-03 (**) |
| **OLMo 2 13B** | `BASE -> SFT` | `olmo_2_1124_13b` | `olmo_2_1124_13b_sft` | 33.78% | 51.80% | **+18.02%** | [+15.50%, +20.52%] | 7.10e-42 (***) |
| **OLMo 2 13B** | `SFT -> INSTRUCT` | `olmo_2_1124_13b_sft` | `olmo_2_1124_13b_instruct` | 51.80% | 49.15% | **-2.65%** | [-4.60%, -0.57%] | 9.73e-03 (**) |
| **OLMo 2 13B** | `INSTRUCT -> DPO` | `olmo_2_1124_13b_instruct` | `olmo_2_1124_13b_dpo` | 49.15% | 50.10% | **+0.94%** | [-0.69%, +2.61%] | 2.78e-01 (ns) |
| **OLMo 2 1B (0425)** | `BASE -> SFT` | `olmo_2_0425_1b` | `olmo_2_0425_1b_sft` | 30.54% | 48.48% | **+17.94%** | [+15.42%, +20.47%] | 2.20e-40 (***) |
| **OLMo 2 1B (0425)** | `SFT -> INSTRUCT` | `olmo_2_0425_1b_sft` | `olmo_2_0425_1b_instruct` | 48.48% | 51.76% | **+3.28%** | [+1.78%, +4.84%] | 3.13e-05 (***) |
| **OLMo 2 1B (0425)** | `INSTRUCT -> DPO` | `olmo_2_0425_1b_instruct` | `olmo_2_0425_1b_dpo` | 51.76% | 50.59% | **-1.17%** | [-2.76%, +0.43%] | 1.53e-01 (ns) |
| **OLMo 2 32B (0325)** | `BASE -> INSTRUCT` | `olmo_2_0325_32b` | `olmo_2_0325_32b_instruct` | 55.01% | 48.60% | **-6.41%** | [-8.63%, -4.31%] | 7.67e-09 (***) |
| **OLMo 2 32B (0325)** | `INSTRUCT -> DPO` | `olmo_2_0325_32b_instruct` | `olmo_2_0325_32b_dpo` | 48.60% | 51.31% | **+2.72%** | [+1.45%, +4.06%] | 3.12e-05 (***) |
