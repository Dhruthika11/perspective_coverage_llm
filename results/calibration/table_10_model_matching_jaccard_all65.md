# 10-Model Kuhn-Munkres Matching Robustness Table (Full Benchmark: 65 Claims)

| Model Architecture | Claims | Viewpoints | Gold Clusters (Denom) | Matched vA | Matched vC | Shared | Jaccard | Micro Cov vA | Micro Cov vC | Macro Cov vA | Macro Cov vC | Delta Macro | Arg Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemma_2_27b_it` | 29 | 36 | 42 | 10 | 10 | 10 | 100.00% | 23.81% (10/42) | 23.81% (10/42) | 23.10% | 23.10% | +0.00% | 0.278 (10/36) |
| `olmo_2_1124_7b_instruct` | 25 | 37 | 42 | 13 | 13 | 13 | 100.00% | 30.95% (13/42) | 30.95% (13/42) | 31.33% | 31.33% | +0.00% | 0.351 (13/37) |
| `qwen2_5_1_5b_instruct` | 15 | 18 | 20 | 4 | 4 | 4 | 100.00% | 20.00% (4/20) | 20.00% (4/20) | 18.33% | 18.33% | +0.00% | 0.222 (4/18) |
| `qwen2_5_3b_instruct` | 14 | 17 | 19 | 3 | 3 | 3 | 100.00% | 15.79% (3/19) | 15.79% (3/19) | 7.14% | 7.14% | +0.00% | 0.176 (3/17) |
| `qwen2_5_7b_instruct` | 30 | 40 | 41 | 13 | 12 | 12 | 92.31% | 31.71% (13/41) | 29.27% (12/41) | 37.78% | 36.11% | -1.67% | 0.300 (12/40) |
| `qwen2_5_32b_instruct` | 16 | 26 | 25 | 7 | 6 | 6 | 85.71% | 28.00% (7/25) | 24.00% (6/25) | 27.08% | 20.83% | -6.25% | 0.231 (6/26) |
| `gemma_2_9b_it` | 28 | 34 | 36 | 11 | 13 | 11 | 84.62% | 30.56% (11/36) | 36.11% (13/36) | 37.50% | 42.26% | +4.76% | 0.382 (13/34) |
| `llama_3_2_3b_instruct` | 25 | 33 | 33 | 9 | 7 | 7 | 77.78% | 27.27% (9/33) | 21.21% (7/33) | 25.33% | 17.33% | -8.00% | 0.212 (7/33) |
| `olmo_2_1124_13b_instruct` | 24 | 36 | 40 | 11 | 10 | 9 | 75.00% | 27.50% (11/40) | 25.00% (10/40) | 24.65% | 20.14% | -4.51% | 0.278 (10/36) |
| `llama_3_1_8b_instruct` | 23 | 26 | 26 | 9 | 5 | 5 | 55.56% | 34.62% (9/26) | 19.23% (5/26) | 36.96% | 21.74% | -15.22% | 0.192 (5/26) |

> [!NOTE]
> **Reconciliation of Matched Edge Counts across Scopes:**
> 1. **Model-Specific Sum ($N = 83$ edges):** Sum of independent matches across each of the 10 models (shown in the table).
> 2. **Pooled Multi-Model Matching ($N = 71$ clusters):** Total unique gold clusters covered when all viewpoints from all models are pooled into a single Kuhn-Munkres matching per claim. Multiple models often match the same gold cluster, collapsing duplicates.
> 3. **Raw Labeled Pairs ($N = 89$ Class 2 pairs):** Total pairs annotated as Class 2 in Version C benchmark.
> 
> **Micro vs. Macro Coverage Definitions:**
> - **Micro Coverage:** $\frac{\sum \text{Matched Clusters}}{\sum \text{Available Clusters}}$ (total matched clusters divided by total gold clusters).
> - **Macro Coverage:** $\frac{1}{|C|} \sum_{c \in C} \frac{\text{Matched Clusters}_c}{\text{Available Clusters}_c}$ (unweighted average of claim-level coverage scores).
> - **Argumentative Efficiency:** $\frac{\text{Matched Clusters}}{\text{Total Generated Viewpoints}}$ (yield of valid perspectives per generated viewpoint).
