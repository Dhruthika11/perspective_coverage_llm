# Methodology Validation: Adapters, Integrity, & Stance Decoupling

**Phase 1 Deliverable**  
**Repository:** `perspective_coverage`

---

## 1. Executive Summary of Phase 1 Implementation

In Phase 1, we executed a comprehensive overhaul and scientific validation of the foundational data adapters and evaluation methodology:
1. **Perspectrum Equivalence Cluster Restoration:** Restored all 5,095 ground-truth perspective clusters, evidence mappings, and claim splits in `data/processed/perspectives.csv`.
2. **IBM Debater Adapter Implementation & Caching:** Ingested 27,679 quality-filtered arguments across 71 debate topics, preserving Weighted Average (`WA >= 0.50`) and MACE-P quality scores, stance confidence, and local caching.
3. **Kialo Adapter Overhaul:** Replaced fabricated alternating stance labeling with authentic proposition labeling across 1,028 topics (931 binary, 97 multi), tagging argument level as `root` and preserving debate structure.
4. **Decoupling of Stance Detection and Perspective Equivalence:** Formalized Task 1 (Stance Detection) vs. Task 2 (Perspective Equivalence) in `src/analysis/stance_vs_perspective.py`, correcting the critical flaw where stance concordance was treated as perspective equivalence.
5. **Automated Test Suite:** Created `tests/test_dataset_adapters.py` and `tests/test_stance_vs_perspective.py`, with 100% test passing rate.

---

## 2. Technical Component Deep-Dive (Student-Learning Reference)

### 2.1 Preserving Equivalence Clusters in Perspectrum
- **What it does:** Groups human crowd-worker perspective texts that express the exact same underlying argumentative reasoning under a common `cluster_id`.
- **Why the project needs it:** In Perspectrum, annotators wrote multiple surface paraphrases of the same core argument. If the cluster ID is dropped, an evaluator treats paraphrases as distinct perspectives, causing models that repeat one argument to receive multiple credits and distorting coverage scores.
- **How it works:** In `src/data_prep/parse_perspectrum.py`, each cluster object within a claim's JSON entry is assigned a unique identifier (`cluster_id = 1 \dots 5095`). Each row in `perspectives.csv` points to its parent `cluster_id`.
- **Inputs and Outputs:**
  - *Input:* `perspectrum_with_answers_v1.0.json` (nested clusters of perspective IDs).
  - *Output:* `data/processed/perspectives.csv` containing columns `['claim_id', 'perspective_id', 'cluster_id', 'claim_cluster_id', 'text', 'stance', 'evidence', 'stance_raw']`.
- **Validation:** Automated test in `test_perspectrum_integrity()` verifies exactly 5,095 clusters across 907 claims, and confirms that every cluster maps to exactly one claim.
- **Scientific Assumptions:** Human annotators in Perspectrum accurately identified semantic equivalence within each cluster.

---

### 2.2 Quality Score Preservation in IBM ArgQ-Rank-30k
- **What it does:** Ingests 27,679 debate arguments across 71 topics, preserving the continuous annotator-weighted quality score ($\text{WA} \in [0, 1]$) and the MACE-P probabilistic quality score.
- **Why the project needs it:** Standard coverage counts all human arguments equally, regardless of rhetorical strength or coherence. IBM Debater allows computing **quality-weighted coverage**:
  $$\text{Coverage}_{\text{WA}} = \frac{\sum_{p \in \text{Covered}} \text{WA}(p)}{\sum_{p \in \text{All}} \text{WA}(p)}$$
- **How it works:** The adapter filters the raw 30,497 arguments to retain only $\text{WA} \ge 0.50$, caching the raw dataset locally at `data/raw/ibm_debater/argument_quality_ranking_30k.csv` and saving processed tables with both $\text{WA}$ and $\text{MACE-P}$.
- **Inputs and Outputs:**
  - *Input:* Hugging Face dataset `ibm-research/argument_quality_ranking_30k`.
  - *Output:* `data/processed/ibm_debater/claims.csv` (71 claims) and `perspectives.csv` (27,679 arguments).
- **Validation:** Automated test `test_ibm_debater_integrity()` verifies $\text{WA} \ge 0.50$ for all rows and validates authentic stance distribution (14,171 support, 13,508 undermine).

---

### 2.3 Elimination of Fabricated Stances in Kialo
- **What it does:** Parses the Kialo debate corpus into 1,028 topics while preserving the authentic debate typology (`binary` vs. `multi`), tagging all top-level arguments as `root`, and removing arbitrary modulo labeling.
- **Why the project needs it:** Previous code assigned `stance = "support" if p_idx % 2 == 0 else "undermine"`. On multi-choice questions with 17 different religions or ethical stances, this arbitrarily labeled alternating options as pro or con.
- **How it works:**
  - In `binary` debates: `perspectives[0]` is labeled `support` (pro thesis), and `perspectives[1]` is labeled `undermine` (con counter-thesis).
  - In `multi` debates: options are labeled `neutral` / `option_k`.
  - All entries are tagged with `argument_level = "root"` and `parent_id = None`.
- **Inputs and Outputs:**
  - *Input:* Hugging Face dataset `timchen0618/Kialo`.
  - *Output:* `data/processed/kialo/claims.csv` (1,028 claims) and `perspectives.csv` (2,727 perspectives).
- **Validation:** Automated test `test_kialo_integrity()` verifies 931 balanced binary pairs and confirms multi-choice options are never assigned fake pro/con stances.

---

### 2.4 Decoupling Stance Detection from Perspective Equivalence
- **What it does:** Explicitly isolates two separate tasks:
  1. *Stance Detection:* Premise = Argument, Hypothesis = Claim. Determines logical direction (`SUPPORT`, `UNDERMINE`, `NEUTRAL`).
  2. *Perspective Equivalence:* Evaluates whether Perspective A and Perspective B share the same stance AND express the same underlying reasoning criterion.
- **Why the project needs it:** The previous DeBERTa model was fine-tuned on data where any generated argument sharing the same stance as a human perspective was labeled as `entailment`. This rewarded models with high coverage simply for taking the correct stance, even if their specific reasoning was completely unaligned with the human reference.
- **How it works:** Implemented in `src/analysis/stance_vs_perspective.py` via `StanceDetector` and `PerspectiveMatcher`.
- **Validation:** Tested in `tests/test_stance_vs_perspective.py`. Confirms that opposite stances are immediately rejected ($\text{Class} = 0$), sub-threshold dense similarity is rejected, and only pairs meeting both dense threshold and semantic verification are classified as `EQUIVALENT` ($\text{Class} = 2$).

