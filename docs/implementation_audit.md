# Perspective Coverage: Implementation Audit

**Audit Date:** November 2026  
**Auditor:** Senior ML Research Engineer & Research Scientist  
**Repository:** `perspective_coverage`

---

## 1. Executive Summary

This audit assesses the functional correctness, scientific validity, dataset integrity, and methodological soundess of the `perspective_coverage` codebase against the Master Research Implementation Plan. 

Every major component is classified into one of five categories:
- **A. Implemented and verified:** Methodologically sound, functionally tested, verified against ground truth.
- **B. Implemented but scientifically unvalidated:** Functional code exists, but validation is incomplete, narrow, or lacks statistical rigor.
- **C. Partially implemented:** Incomplete coverage of research goals or partial data pipelines.
- **D. Missing:** Required by the research plan but no code or artifacts exist.
- **E. Implemented but requiring correction:** Code exists and runs, but contains significant scientific, conceptual, or methodological errors.

---

## 2. Component-by-Component Classification

### 2.1 Dataset Ingestion & Adapters

| Component | Status | Source File | Findings & Methodological Issues |
|---|---|---|---|
| **Perspectrum Adapter** | **E** | `src/data_prep/parse_perspectrum.py` | Parses `claims_file` and `perspective_pool_file`. **Critical flaw:** Drops equivalence cluster IDs. Each perspective ID is stored as an independent row in `data/processed/perspectives.csv`. Evaluator treats paraphrases of the same perspective as distinct target perspectives. Raw JSON (`perspectrum_with_answers_v1.0.json`) has full cluster groupings that must be restored. |
| **IBM ArgQ-Rank-30k Adapter** | **E** | `src/data_prep/parse_ibm_debater.py` | Pulls from HuggingFace (`ibm-research/argument_quality_ranking_30k`). Filters by `WA >= 0.5`. **Flaws:** Does not preserve argument IDs, drops MACE-P quality score, does not cache raw data locally in `data/raw/ibm_debater/`. Unexecuted locally (`data/processed/ibm_debater/` missing). |
| **Kialo Corpus Adapter** | **E** | `src/data_prep/parse_kialo.py` | Downloads `timchen0618/Kialo`. **Critical Scientific Flaw:** Completely flattens argument trees, discards parent-child relationships, and **synthesizes fake stance labels** via `stance = "support" if p_idx % 2 == 0 else "undermine"`. Unexecuted locally. |
| **Unified Tri-Dataset Merger** | **E** | `run_tri_dataset_evaluation.py`, `inspect_tri_datasets.py` | Attempts to pool all three datasets into a single global table with arbitrary integer ID offsets. Conflates distinct argumentation topologies into an unjustified single benchmark score. |

---

### 2.2 Perspective Generation Methodology

| Component | Status | Source File | Findings & Methodological Issues |
|---|---|---|---|
| **Baseline 5-Aspect Prompt (P1)** | **C** | `src/generation/generate_llama.py`, `configs/config.yaml` | Implements sequential aspect prompting across 8 aspects (config default) with simple history string. Does not maintain a clean, frozen, versioned P1 baseline with 5 fixed aspects (Economic, Moral, Legal, Practical, Social). |
| **Reasoning-Criteria Prompt (P2)** | **D** | Missing | Research plan requires explicit underlying reasoning criteria (e.g., affordability, fairness, constitutional obligations) and semantic novelty constraints. Not implemented. |
| **Prompt Ablations (P3, P4)** | **D** | Missing | P3 (Aspect=Yes, History=No, Criteria=Yes) and P4 (Aspect=No, History=Yes, Criteria=Yes) do not exist. |
| **Abstention Mechanism** | **D** | Missing | Models are forced to output a JSON object regardless of argumentative plausibility. No explicit abstention state (`status: "abstain"`). |
| **Structured Output Parsing** | **B** | `src/generation/postprocess_generations.py` | Robust regex and JSON fallbacks, but performs "viewpoint explosion" (splitting one call into 1 to 4 rows). Models producing verbose bulleted lists generate 17+ viewpoints per claim, while concise models generate 5. This introduces a major candidate-count confounder for coverage. |
| **Generation Logging** | **C** | `src/generation/generate_llama.py` | Saves `{claim_id}_samples.json`, but omits prompt version, token budget metadata, latency, reasoning criteria, and abstention status. |

---

### 2.3 Evaluator: Stance Classification vs. Perspective Equivalence

| Component | Status | Source File | Findings & Methodological Issues |
|---|---|---|---|
| **NLI Stance Classifier** | **B** | `src/analysis/nli_stance_classifier.py` | Cross-encoder `cross-encoder/nli-deberta-v3-small` with Premise = Generation, Hypothesis = Claim. Direction is functionally plausible for stance detection, but accuracy on nuanced debate propositions needs standalone validation. |
| **Argument-NLI Data Preparation** | **E** | `src/training/prepare_multidataset_nli_data.py` | **Major Scientific Flaw:** Generates pseudo-training pairs by defining `label = "entailment"` if a candidate has the *same stance* as a human perspective on the same claim, and `"contradiction"` if it has the *opposite stance*. Two completely different arguments supporting the same claim (e.g., "reduces poverty" vs "simplifies tax administration") are labeled as *entailment*. |
| **DeBERTa Fine-Tuned Model** | **E** | `models/argument_nli_deberta_v3/` | Fine-tuned on the flawed data above. The resulting model is a **stance concordance classifier**, NOT a perspective equivalence classifier. |
| **Stage 2 Verification** | **E** | `src/analysis/evaluate_finetuned_two_stage.py`, `run_tri_dataset_evaluation.py` | Passes candidate matches through the stance-concordance DeBERTa. High entailment probability ($p_{\text{entail}} \ge 0.80$) simply confirms the generation shares the reference perspective's stance, not that it expresses equivalent reasoning. |
| **Equivalence Ground Truth Dataset** | **C** | `data/calibration/hard_calibration_pairs.csv` | 300 hand-crafted pairs. Lacks claim-level train/val/test splits (risking topic leakage). 300 pairs is insufficient for cross-dataset calibration. |
| **Probability Calibration** | **D** | Missing | No reliability diagrams, Brier score, ECE, or Platt scaling/temperature scaling for verification probabilities. |

---

### 2.4 Evaluator: Dense Embedding Retrieval & Matching

| Component | Status | Source File | Findings & Methodological Issues |
|---|---|---|---|
| **Embedding Generation** | **A** | `src/embeddings/embed_model.py` | Implements dense embeddings with BAAI/bge-large-en-v1.5, bge-base, nomic, mpnet. Caching and indexing work correctly. |
| **Stage 1 Threshold Calibration** | **B** | `src/analysis/threshold_calibration.py` | Evaluates within-cluster vs. across-cluster similarity in Perspectrum (finding $\tau \approx 0.744$). However, it optimizes F1 rather than enforcing the candidate recall constraint ($\tau^* = \max \tau \text{ s.t. CandidateRecall} \ge 0.98$). |
| **Bipartite Graph Matching** | **E** | `src/analysis/coverage_score.py` | Computes greedy maximum similarity per perspective: `sim_matrix.max(axis=0) >= threshold`. Does not construct a bipartite graph or run Maximum Bipartite Matching. Allows a single generated argument to cover unlimited reference perspectives. |
| **Cluster-Level Coverage** | **E** | `src/analysis/coverage_score.py` | Denominator is the raw count of perspective IDs ($N=11,805$), not the number of equivalence clusters. Violates the benchmark specification. |

---

### 2.5 Experiments & Analysis

| Component | Status | Source File / Output | Findings & Methodological Issues |
|---|---|---|---|
| **Main Scaling Experiment (Qwen2.5)** | **C** | `results/two_stage_cross_model_comparison.csv` | Summary metrics exist for 0.5B, 1.5B, 3B, 7B, 14B, 32B. However, raw generation files on local disk exist only for 1.5B, 3B, 32B. Inverted-U scaling curve is heavily confounded by parsing failures (1.5B had 0.04% parse rate) and viewpoint explosion. |
| **Post-Training Alignment (OLMo2)** | **D** | Missing | Base, SFT, DPO, Instruct comparison for OLMo2 (1B, 7B, 13B) is completely unexecuted and unscripted. |
| **Teacher-Student Preservation** | **D** | Missing | Comparison between Llama 3.1 70B teacher and 8B/1B/3B students is unexecuted. |
| **Perspective Blind Spots** | **C** | `src/analysis/analyze_perspective_blind_spots.py`, `results/paper_tables/` | Produces categorization (1,920 always covered, 2,368 partial, 7,517 never). However, these numbers reflect the flawed single-stage greedy matching, not bipartite cluster coverage. |
| **Sycophancy Experiment** | **E** | `paper_draft.tex` (Section 4.3) | The manuscript claims models exhibit "Universal Sycophancy" based purely on the ratio of Support vs. Undermine coverage. No controlled user-framing experiment (Neutral vs. User-Support vs. User-Undermine) exists in the codebase. Conflates stance bias with sycophancy. |
| **Tri-Dataset Experiment** | **C** | `run_missing_generations.py`, `run_tri_dataset_evaluation.py` | Script exists, but cannot run locally because IBM and Kialo processed files are absent. |

---

### 2.6 Manuscript & Paper Artifacts

| Component | Status | Source File | Findings & Methodological Issues |
|---|---|---|---|
| **LaTeX Draft** | **B** | `paper_draft.tex` | Well-structured manuscript, but contains claims derived from the flawed stance-concordance DeBERTa, unvalidated sycophancy claims, and uncorrected scaling figures. |
| **Paper Tables** | **B** | `results/paper_tables/*.csv` | Pre-computed tables match manuscript text, but must be regenerated once the evaluator and adapters are scientifically corrected. |

