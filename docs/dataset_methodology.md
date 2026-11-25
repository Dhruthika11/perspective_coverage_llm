# Dataset Methodology & Adapter Architecture

**Phase 1 Deliverable**  
**Repository:** `perspective_coverage`  
**Datasets Covered:** Perspectrum, IBM ArgQ-Rank-30k (IBM Debater), and Kialo Debate Corpus

---

## 1. Overview and Design Principles

The research requires evaluating large language models across three distinct argumentation topologies. A core finding of the Phase 0 audit is that previous scripts conflated these topologies into an unjustified flattened benchmark and discarded essential structural relationships.

The corrected Phase 1 architecture enforces the following principles:
1. **Topological Preservation:** Equivalence clusters (Perspectrum), argument quality distributions (IBM Debater), and debate proposition types (Kialo) must be authentically preserved.
2. **Dataset Independence:** Each dataset must be processed and evaluated independently with its own dataset-specific metrics (e.g., quality-weighted coverage for IBM, cluster-level bipartite coverage for Perspectrum, root vs. child coverage for Kialo).
3. **Local Offline Reproducibility:** Raw datasets are cached locally in `data/raw/<dataset>/` so downstream experiments do not depend on external network access or volatile remote repository states.

---

## 2. Dataset Specifications

### 2.1 Perspectrum (Cluster-Preserving Deliberation)
- **Source:** Cognitive Computation Group, University of Illinois at Urbana-Champaign (Chen et al., 2019).
- **Raw Location:** `data/raw/perspectrum/` (`perspectrum_with_answers_v1.0.json`, `perspective_pool_v1.0.json`, `dataset_split_v1.0.json`).
- **Adapter:** `src/data_prep/parse_perspectrum.py`.
- **Benchmark Dimensions:**
  - **907 controversial claims** spanning 9 thematic categories (politics, bioethics, law, society).
  - **5,095 ground-truth equivalence clusters** ($\mu = 5.62$ clusters per claim).
  - **11,805 total perspective instances** ($\mu = 2.32$ crowd-worker paraphrases per cluster).
- **Preserved Schema (`data/processed/perspectives.csv`):**
  - `claim_id` (int): Claim identifier.
  - `perspective_id` (int): Unique crowd-worker paraphrase identifier.
  - `cluster_id` (int): Globally unique equivalence cluster identifier ($1 \dots 5095$).
  - `claim_cluster_id` (int): Index of the cluster within the claim ($0 \dots N-1$).
  - `text` (str): Perspective text statement.
  - `stance` (str): Normalized stance (`support`, `undermine`, or `unclear`).
  - `evidence` (str / json): Associated evidence identifiers.
  - `stance_raw` (str): Original crowd annotation (`SUPPORT`, `UNDERMINE`).

---

### 2.2 IBM ArgQ-Rank-30k / IBM Project Debater
- **Source:** IBM Research (Gretz et al., 2020; Toledo et al., 2019).
- **Raw Location:** `data/raw/ibm_debater/argument_quality_ranking_30k.csv`.
- **Adapter:** `src/data_prep/parse_ibm_debater.py`.
- **Benchmark Dimensions:**
  - **71 controversial debate topics**.
  - **27,679 quality-filtered arguments** satisfying $\text{WA} \ge 0.50$ (from an original unfiltered pool of 30,497 arguments).
  - Stance distribution: 14,171 Support ($51.2\%$) and 13,508 Undermine ($48.8\%$).
- **Preserved Schema (`data/processed/ibm_debater/perspectives.csv`):**
  - `claim_id` (int): Unique topic identifier ($1 \dots 71$).
  - `perspective_id` (int): Unique argument identifier.
  - `text` (str): High-quality argument text.
  - `stance` (str): Ground-truth stance (`support` or `undermine`).
  - `wa_quality` (float): Weighted Average quality score ($\ge 0.50$, primary quality metric).
  - `mace_p_quality` (float): Multi-Annotator Competence Estimation probability score (for sensitivity analysis).
  - `stance_conf` (float): Annotator stance agreement confidence.
  - `original_topic` (str): Raw debate proposition.
  - `original_set` (str): Dataset split (`train`, `validation`, `test`).

---

### 2.3 Kialo Debate Corpus
- **Source:** Kialo debate platform, diverse subjective retrieval benchmark (Chen et al., 2023).
- **Raw Location:** `data/raw/kialo/kialo_raw.json`.
- **Adapter:** `src/data_prep/parse_kialo.py`.
- **Benchmark Dimensions:**
  - **1,028 unique controversial debate questions**.
  - **931 Binary debates** (questions with explicit pro vs. con propositions).
  - **97 Multi-perspective debates** (questions with multiple categorical positions).
  - **2,727 total root perspectives** (931 binary support, 931 binary undermine, 865 multi-choice options).
- **Preserved Schema (`data/processed/kialo/perspectives.csv`):**
  - `claim_id` (int): Debate identifier ($1 \dots 1028$).
  - `perspective_id` (int): Unique perspective identifier.
  - `kialo_id` (str): Original platform discussion identifier.
  - `debate_type` (str): `'binary'` or `'multi'`.
  - `text` (str): Proposition statement.
  - `stance` (str): `'support'`, `'undermine'`, or `'neutral'` (categorical option).
  - `relation` (str): `'pro'`, `'con'`, or `'option_k'`.
  - `argument_level` (str): `'root'` for top-level propositions; extensible for hierarchical debate trees.
  - `parent_id`: `None` for root perspectives.
  - `option_index` (int): Index within the debate's perspectives.

---

## 3. Data Integrity & Validation Summary

All three adapters were validated with automated tests in `tests/test_dataset_adapters.py`:
- **0 duplicate records** across claim-perspective pairs.
- **0 empty or whitespace-only texts**.
- **100% valid stance assignments** (eliminated all fabricated alternating labels).
- **Cluster integrity confirmed**: Every cluster in Perspectrum maps to exactly one parent claim with zero cross-claim contamination.

