# Next Execution Steps: Roadmap & Immediate Actions

This document specifies the exact, step-by-step technical plan to execute the research project according to the Master Research Implementation Prompt.

---

## 1. Phase-by-Phase Roadmap

### Phase 1: Validate Existing Data & Evaluation Methodology (Immediate Next Step)
1. **Fix Perspectrum Parser (`src/data_prep/parse_perspectrum.py`):**
   - Extract and preserve the `cluster_id` from raw `perspectrum_with_answers_v1.0.json`.
   - Update `data/processed/perspectives.csv` to include `cluster_id`.
   - Add automated test `tests/test_dataset_perspectrum.py` verifying cluster preservation and ID joins.
2. **Implement Scientifically Valid IBM Debater Adapter (`src/data_prep/parse_ibm_debater.py`):**
   - Download `ibm-research/argument_quality_ranking_30k`.
   - Preserve original topic IDs, argument IDs, authentic stance, WA quality scores, and MACE-P scores.
   - Cache raw data in `data/raw/ibm_debater/`.
3. **Implement Tree-Preserving Kialo Adapter (`src/data_prep/parse_kialo.py`):**
   - Remove fake alternating stance logic (`stance = "support" if p_idx % 2 == 0...`).
   - Parse debate tree structure: root-level propositions vs child arguments, parent IDs, and genuine support/attack relations.
   - Cache raw data in `data/raw/kialo/`.
4. **Audit and Separate Stance vs. Perspective Matching:**
   - Formalize Task 1: Stance Classification ($\text{Premise} = \text{Argument}, \text{Hypothesis} = \text{Claim}$).
   - Formalize Task 2: Perspective Equivalence ($\text{Premise} = \text{Perspective A}, \text{Hypothesis} = \text{Perspective B}$).

---

### Phase 2: Improve Perspective Generation
1. **Freeze Versioned Baseline Prompt (P1):**
   - 5 fixed aspects: Economic, Moral, Legal, Practical, Social.
   - Cumulative history with atomic argument requirement.
2. **Implement Improved Reasoning-Criteria Prompt (P2):**
   - Explicit reasoning criterion elicitation (e.g. affordability, equality, constitutional authority).
   - Explicit semantic novelty constraint preventing near-paraphrases.
   - Explicit abstention capability (`"status": "abstain"`).
   - Structured JSON schema: `{"status": ..., "aspect": ..., "reasoning_criterion": ..., "stance": ..., "argument": ...}`.
3. **Implement Controlled Prompt Ablations (P1, P2, P3, P4):**
   - P1: Aspect=Yes, History=Yes, Reasoning Criterion=No (Baseline).
   - P2: Aspect=Yes, History=Yes, Reasoning Criterion=Yes (Improved Main Prompt).
   - P3: Aspect=Yes, History=No, Reasoning Criterion=Yes.
   - P4: Aspect=No, History=Yes, Reasoning Criterion=Yes.
4. **Implement Marginal Coverage and Diversity Metrics:**
   - Incremental coverage: $\Delta C_k = C_k - C_{k-1}$.
   - Abstention rate logging.

---

### Phase 3: Calibrate Dense Retrieval & Perspective Equivalence
1. **Construct Expanded Ground Truth Benchmark (600–900 Pairs):**
   - Three classes: 0 = Different, 1 = Related, 2 = Equivalent.
   - Split strictly by claim (train / calibration / test) to prevent topical data leakage.
2. **Stage 1 Embedding Calibration:**
   - Sweep cosine threshold $\tau \in [0.55, 0.90]$ across BGE-large, BGE-base, MPNet, Nomic.
   - Select $\tau^* = \max \tau \text{ s.t. CandidateRecall}(\tau) \ge 0.98$.
3. **Stage 2 Equivalence Verification:**
   - Replace the stance-concordance DeBERTa with a validated equivalence model.
   - Threshold selection rule: maximize recall subject to precision $\ge 0.90$.
4. **Implement Maximum Bipartite Matching:**
   - Construct bipartite graph between generated viewpoints and reference clusters.
   - Solve using `scipy.optimize.linear_sum_assignment` to ensure 1-to-1 matching.

---

### Phase 4: Complete Three-Dataset Experiments
1. Ingest and parse IBM Debater and Kialo locally.
2. Run generations for missing combinations (`gemma_2_2b_it`, `gemma_3_12b_it`).
3. Execute tri-dataset evaluation with independent per-dataset reporting (no artificial pooling).

---

### Phase 5 & 6: Core Model Experiments & Supporting Investigations
1. **Scaling Analysis:** Audit Qwen2.5 (0.5B to 32B) outputs; remove parsing and candidate-count confounders; fit log-parameter scaling curves.
2. **Sycophancy Experiment:** Implement 3-condition user-framing prompt (Neutral, User-Support, User-Undermine) to measure genuine stance drift.
3. **Blind-Spot Analysis:** Recompute blind spots using bipartite cluster coverage.
4. **Quality-Weighted Coverage:** Compute IBM coverage weighted by WA and MACE-P.

---

### Phase 7 & 8: Statistical Analysis & Manuscript Revision
1. Claim-level bootstrap confidence intervals (1,000 resamples).
2. Paired Wilcoxon signed-rank tests for model comparisons.
3. Update `paper_draft.tex` and generate publication-quality figures programmatically.

---

## 2. Immediate Next Command

To begin Phase 1 immediately without modifying any generation results:

```powershell
python -m pytest tests/
```
Followed by executing the corrected Perspectrum cluster-preserving adapter:
```powershell
python src/data_prep/parse_perspectrum.py
```

