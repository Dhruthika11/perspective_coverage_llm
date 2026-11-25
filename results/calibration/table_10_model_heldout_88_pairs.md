# Model-Level Verifier Performance on the 88 Held-Out Pairs (Version C Adjudicated)

Frozen $M_0$ ($T^* = 0.20$) audited across candidate pairs originating from all 10 models in the test split.

| Model Architecture | Pairs in Test | True C2 | TP | FP | Class 2 Recall | Class 2 Precision | Class 1 Rejection | Class 0a Rejection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma_2_27b_it` | 14 | 6 | 6 | 1 | 100.0% (6/6) | 85.7% (6/7) | 50.0% (1/2) | 100.0% (5/5) |
| `gemma_2_9b_it` | 10 | 5 | 5 | 0 | 100.0% (5/5) | 100.0% (5/5) | N/A | 100.0% (5/5) |
| `llama_3_1_8b_instruct` | 7 | 2 | 2 | 1 | 100.0% (2/2) | 66.7% (2/3) | 66.7% (2/3) | 100.0% (2/2) |
| `llama_3_2_3b_instruct` | 9 | 0 | 0 | 5 | N/A (0 C2) | 0.0% (0/5) | 0.0% (0/4) | 100.0% (3/3) |
| `olmo_2_1124_13b_instruct` | 8 | 3 | 3 | 1 | 100.0% (3/3) | 75.0% (3/4) | 50.0% (1/2) | 100.0% (1/1) |
| `olmo_2_1124_7b_instruct` | 13 | 3 | 3 | 5 | 100.0% (3/3) | 37.5% (3/8) | 37.5% (3/8) | 100.0% (2/2) |
| `qwen2_5_1_5b_instruct` | 4 | 2 | 2 | 0 | 100.0% (2/2) | 100.0% (2/2) | N/A | 100.0% (1/1) |
| `qwen2_5_32b_instruct` | 7 | 0 | 0 | 1 | N/A (0 C2) | 0.0% (0/1) | 100.0% (2/2) | 80.0% (4/5) |
| `qwen2_5_3b_instruct` | 2 | 0 | 0 | 0 | N/A (0 C2) | N/A (0 Acc) | 100.0% (1/1) | N/A |
| `qwen2_5_7b_instruct` | 14 | 2 | 2 | 5 | 100.0% (2/2) | 28.6% (2/7) | 37.5% (3/8) | 100.0% (2/2) |
| **OVERALL (All 10 Models)** | 88 | 23 | 23 | 19 | **100.0% (23/23)** | **54.8% (23/42)** | **43.3% (13/30)** | **96.2% (25/26)** |
