# Tri-Dataset Independent Evaluation Protocol (Phase 4)

## 1. Executive Summary

This protocol defines the independent, scientifically valid evaluation methodology across the three core research datasets:
1. **Perspectrum** (Crowd-annotated multi-perspective debates with semantic equivalence clusters)
2. **IBM ArgQ-Rank-30k** (Expert & crowd-ranked argument quality dataset with WA and MACE-P scores)
3. **Kialo Debate Corpus** (Structured multi-party debate trees with binary and multi-choice topics)

---

## 2. Why Cross-Dataset "Pooled" Coverage is Invalid

In legacy evaluation pipelines (`run_tri_dataset_evaluation.py`), claims and perspectives from all three datasets were concatenated into an artificial single table:
$$\text{Flawed Pooled Coverage} = \frac{\text{Covered}_{\text{perspectrum}} + \text{Covered}_{\text{ibm}} + \text{Covered}_{\text{kialo}}}{N_{\text{perspectrum}} + N_{\text{ibm}} + N_{\text{kialo}}}$$

This pooling creates severe scientific artifacts:
* **Incompatible Denominators:** Perspectrum evaluates against equivalence clusters ($M_{\text{clusters}} \approx 5.6$ per claim), whereas IBM Debater contains hundreds of quality-ranked sentence-level arguments per topic ($N \approx 390$ per topic). Pooling causes IBM Debater to mathematically overwhelm Perspectrum by $70:1$, distorting the model rankings.
* **Structural Incompatibility:** Kialo features parent-child thesis/rebuttal hierarchies and multi-choice options, which cannot be modeled as a flat list of independent sentences.
* **Metric Incongruity:** An argument matching an IBM Debater perspective should be weighted by its empirical quality ($\text{WA}$), whereas a Perspectrum match represents covering a semantic cluster.

**Core Principle:** *Each dataset must be evaluated and reported independently under its native ground-truth formulation.*

---

## 3. Dataset-Specific Mathematical Formulations

### A. Perspectrum (Cluster-Level Maximum Bipartite Matching)

* **Ground Truth:** Each claim has $M_{\text{clusters}}$ distinct semantic viewpoint clusters $C_1, \dots, C_M$.
* **Cluster Similarity:**
  $$S(g, C) = \max_{p \in C} S(g, p) \quad \text{or} \quad S_{\text{top-}k}(g, C) = \frac{1}{\min(k, |C|)} \sum_{p \in \text{Top}K(C)} S(g, p)$$
* **Bipartite Assignment:** Solved using the Kuhn-Munkres algorithm on bipartite graph $G = (V_{\text{gen}}, V_{\text{clusters}}, E)$.
* **Coverage Metric:**
  $$C_{\text{perspectrum}} = \frac{|\mathcal{M}^*|}{M_{\text{clusters}}} \in [0.0, 1.0]$$
* **Stance Balance Gap:**
  $$\Delta_{\text{stance}} = |C_{\text{support}} - C_{\text{undermine}}|$$

---

### B. IBM ArgQ-Rank-30k (Quality-Weighted Perspective Coverage)

* **Ground Truth:** 71 controversial debate topics with 27,679 filtered arguments satisfying Weighted-Average Quality $\text{WA} \ge 0.50$.
* **Quality-Weighted Coverage:**
  $$C_{\text{quality}} = \frac{\sum_{p \in \mathcal{M}^*} \text{WA}(p)}{\sum_{p \in \mathcal{P}} \text{WA}(p)} \in [0.0, 1.0]$$
* **High-Quality Coverage ($C_{\text{HQ}}$):**
  Evaluates coverage restricted strictly to the highest-quality human arguments ($\text{WA} \ge 0.70$).

---

### C. Kialo Debate Corpus (Tree-Aware & Multi-Choice Coverage)

* **Ground Truth:** 931 binary debates and 97 categorical multi-choice debates (total 1,028 debate claims).
* **Binary Proposition Coverage:** Coverage across pro/con proposition trees.
* **Categorical Multi-Choice Coverage:**
  $$\text{Coverage}_{\text{multi}} = \frac{|\text{Covered Options}|}{N_{\text{options}}}$$
  Measures whether models authentically cover distinct categorical options ($0 \dots N-1$) in non-binary debates (e.g. types of renewable energy, philosophical systems).
* **Root Proposition Coverage:** Coverage restricted to root-level central theses.

---

## 4. Execution Workflow

### Step 1: Audit Generation Status
```powershell
python -m src.generation.tri_dataset_generator --audit_status
```

### Step 2: Dry-Run Validation (Zero LLM / Token Cost)
```powershell
python -m src.generation.tri_dataset_generator --dataset ibm_debater --prompt p2 --k 5 --dry_run
python -m src.generation.tri_dataset_generator --dataset kialo --prompt p2 --k 5 --dry_run
```

### Step 3: Run Live Generation (Requires Explicit User Approval)
```powershell
python -m src.generation.tri_dataset_generator --dataset ibm_debater --model <model_name> --prompt p2 --k 5
```

### Step 4: Run Independent Tri-Dataset Evaluation
```powershell
python -m src.analysis.tri_dataset_evaluator --tau 0.75 --theta 0.80
```

