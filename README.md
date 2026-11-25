# Calibrated Maximum Bipartite Matching for Perspective Coverage in Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Repository](https://img.shields.io/badge/GitHub-Dhruthika11%2Fperspective__coverage__llm-blue)](https://github.com/Dhruthika11/perspective_coverage_llm)

Official research code and reproducibility repository for:  
**"Calibrated Maximum Bipartite Matching for Perspective Coverage in Large Language Models"**  
*Dhruthika Thoutam — Indian Institute of Technology Gandhinagar*  
*Email:* `thoutam.dhruthika@iitgn.ac.in`

---

## 1. Abstract & Key Research Insights

When Large Language Models (LLMs) are deployed in deliberative, democratic, or decision-support environments, they must not collapse onto a single consensus stance; they must faithfully represent the full landscape of valid human perspectives. Evaluating perspective coverage requires determining whether a model's generated viewpoints capture the diverse reasons, values, and argumentative warrants established by human deliberators, rather than merely repeating superficial paraphrases.

In this work, we introduce the **Calibrated Maximum Bipartite Matching Framework** for evaluating generative deliberation. Our framework pairs generated viewpoints with ground-truth human equivalence clusters through a calibrated two-stage pipeline:
1. **Stage 1 (Dense Candidate Retrieval):** BGE-Large (`bge-large-en-v1.5`, 1024-d) operating at a calibrated safety threshold ($\tau^* = 0.55$), pruning 74.8% of non-candidate pairs on GPU while preserving $\ge 98\%$ candidate recall.
2. **Stage 2 (Stance Concordance & Warrant Verification):** Zero-shot NLI stance filtering coupled with a fine-tuned cross-encoder warrant probe (`DeBERTa-v3-large`, $T^* = 0.20$) to separate true perspective matches from topical overlap.
3. **Stage 3 (Bipartite Graph Optimization):** The **Kuhn–Munkres (Hungarian) algorithm** enforces strict 1-to-1 injection between generated viewpoints and gold perspective clusters, preventing models from gaming coverage scores through repetitive paraphrases.

### Key Empirical Findings (Evaluated on 34 LLMs, 205k+ Viewpoints, 907 Claims)
- **Budget-Bottlenecked vs. Parameter-Bottlenecked Coverage:** Perspective coverage under fixed generation budgets is **budget-bottlenecked rather than parameter-bottlenecked**. Merely scaling model weights without expanding the generation horizon ($K$) does not compel models to uncover new perspectives: scaling from Gemma-2-9B to 27B ($p = 0.633$), OLMo-2-7B to 13B ($p = 0.832$), and LLaMA-3.1-8B to 3.3-70B ($p = 0.389$) all result in statistical parity.
- **Compact Models Excel Under Controlled Budgets:** Under a strictly standardized budget ($K = 5$ viewpoints per claim), `llama_3_2_3b_instruct` achieves the highest coverage (**54.90%**, efficiency $0.512$), outperforming larger 8B, 13B, and 27B models.
- **Supervised Fine-Tuning Drives Diversity:** Across the OLMo 2 family, Supervised Fine-Tuning (SFT) yields a massive $+9.57\%$ to $+18.02\%$ coverage boost over base models ($p < 10^{-13}$), while Direct Preference Optimization (DPO) further refines deliberative balance without collapsing perspective diversity.
- **Resolution of Multiplicity Ceilings:** On high-complexity claims ($\ge 7$ clusters), raw coverage for $K=5$ models drops due to mathematical budget limits, but **Normalized Coverage ($NC_c$)** reaches **82.8%–83.9%**, demonstrating that models capture nearly all perspectives permitted by their generation budget.

---

## 2. Benchmark Summary (Standardized Budget $K=5$)

Evaluated across all 907 Perspectrum debate claims ($N_{\text{gold}} = 5,095$ equivalence clusters) using calibrated thresholds ($\tau^* = 0.55$, $T^* = 0.20$) and Kuhn–Munkres bipartite matching:

| Model Identifier | Architecture Family | Parameters | Raw Macro Coverage ($M_0$) | Normalized Coverage ($C_c^{\text{norm}}$) | Viewpoint Efficiency ($\eta$) | Operational Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `llama_3_2_3b_instruct` | LLaMA 3.2 | 3.0B | **0.4633** | **0.5489** | **0.5241** | **Peak efficiency across $K=5$** |
| `olmo_2_1124_7b_instruct` | OLMo 2 | 7.0B | 0.4611 | 0.5462 | 0.5208 | Frontier parity |
| `gemma_2_9b_it` | Gemma 2 | 9.0B | 0.4601 | 0.5451 | 0.5195 | Frontier parity |
| `olmo_2_1124_13b_instruct` | OLMo 2 | 13.0B | 0.4597 | 0.5448 | 0.5195 | Frontier parity |
| `gemma_2_27b_it` | Gemma 2 | 27.0B | 0.4578 | 0.5429 | 0.5166 | Parameter scaling plateau |
| `llama_3_1_8b_instruct` | LLaMA 3.1 | 8.0B | 0.4552 | 0.5398 | 0.5140 | High operational efficiency |
| `olmo_2_1124_7b_dpo` | OLMo 2 | 7.0B | 0.4431 | 0.5273 | 0.5003 | Post-training alignment |
| `olmo_2_0425_1b_instruct` | OLMo 2 | 1.0B | 0.3828 | 0.4538 | 0.4323 | Sub-3B parameter penalty |

---

## 3. Quickstart & Installation

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Dhruthika11/perspective_coverage_llm.git
cd perspective_coverage_llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducibility Commands
```bash
# 1. Run full test suite
pytest tests/

# 2. Execute end-to-end evaluation pipeline
python run_pipeline.py

# 3. Run two-stage calibrated benchmark on evaluated models
python run_calibrated_twostage_benchmark.py

# 4. Run statistical hypothesis testing (10,000 paired bootstrap resamples with Holm-Bonferroni correction)
python scripts/run_10k_bootstrap_and_holm.py
```

---

## 4. Repository Structure

```
perspective_coverage_llm/
├── configs/                     # YAML/JSON configurations for models and thresholds
├── data/
│   ├── calibration/             # Dataset B (1,700 pairs) & 341 human-LLM calibration splits
│   ├── fresh_claim_benchmark/   # Untouched generalization splits
│   └── processed/               # Clean Perspectrum claims, perspectives, and prompts
├── docs/                        # Formal scientific protocols and calibration methodology
├── results/
│   ├── calibration/             # Threshold sweep curves and precision-recall analyses
│   ├── coverage/                # Master coverage metrics across all 34 evaluated models
│   ├── paper_figures/           # High-resolution publication plots (300 DPI)
│   └── paper_tables/            # Master benchmark tables, planned comparisons, and CIs
├── scripts/                     # Automated runners for calibration, evaluation, and bootstrap audits
├── src/
│   ├── analysis/                # Bipartite matching (Kuhn-Munkres), bootstrap, and metrics
│   ├── data_prep/               # Ingestion adapters for Perspectrum and debate benchmarks
│   ├── embeddings/              # BGE-Large bi-encoder dense retrieval & pooling
│   ├── extraction/              # Viewpoint parsing, aspect filtering, and deduplication
│   ├── generation/              # Prompt templates and model generation harnesses
│   └── training/                # DeBERTa-v3 cross-encoder fine-tuning harnesses
├── tests/                       # Automated unit and integration test suite
├── overleaf_report.tex          # Complete Overleaf-ready LaTeX research paper
├── report.md                    # Comprehensive academic technical report
└── requirements.txt             # Environment dependencies
```

---

## 5. Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{thoutam2026perspective,
  title={Calibrated Maximum Bipartite Matching for Perspective Coverage in Large Language Models},
  author={Thoutam, Dhruthika},
  journal={Indian Institute of Technology Gandhinagar},
  year={2026}
}
```
