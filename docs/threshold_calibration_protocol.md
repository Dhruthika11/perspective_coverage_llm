# Threshold Calibration & Evaluator Validation Protocol (Phase 3)

**Phase 3 Implementation Deliverable**  
**Repository:** `perspective_coverage`

---

## 1. Executive Summary

Phase 3 establishes an empirically calibrated, mathematically principled two-stage evaluation pipeline to determine whether LLM-generated arguments cover human reference perspectives.

### Key Corrections Over Previous Baseline:
1. **Replaced Synthetic Artifacts:** Historical calibration files (such as `build_hard_calibration.py`) manufactured cosine similarities using random numbers (`np.random.uniform(0.04, 0.10)`). Phase 3 uses authentic pairwise distances derived from ground-truth human deliberation clusters.
2. **Strict Claim-Level Splits:** Eliminated pair-level cross-validation which leaked identical debate claims across train and test sets. Phase 3 enforces strict **claim-level splitting** ($60\%$ train, $20\%$ calibration, $20\%$ test) with **zero claim overlap**.
3. **Decoupled Candidate Retrieval from Verification:** Calibrates Stage 1 dense retrieval under a **Candidate Recall Constraint** ($\tau^* \ge 0.98$) to prevent premature exclusion of valid arguments before Stage 2 cross-encoder verification.
4. **Maximum Bipartite Matching:** Replaced greedy maximum similarity with the Kuhn-Munkres algorithm (`scipy.optimize.linear_sum_assignment`), mathematically guaranteeing that no generated viewpoint covers multiple reference clusters and no cluster is counted more than once.

---

## 2. The 3-Class Ground Truth Benchmark Dataset

Located at `data/calibration/tri_dataset_750_pairs.csv` ($N = 749$ pairs across 331 unique debate topics).

### The 3 Annotation Classes:
* **Class 2 (Equivalent Underlying Perspective, $N = 330$):**
  - Both statements express the same core argumentative reason and share the same stance polarity toward the claim.
  - Sourced from within the same ground-truth Perspectrum equivalence cluster and audited human paraphrase matches.
* **Class 1 (Related but Distinct Reasoning, $N = 191$):**
  - Both statements share the same topic and stance, but invoke fundamentally different reasoning criteria (e.g. an economic cost argument vs. an environmental sustainability argument supporting renewable energy).
  - Sourced from cross-cluster pairs within Perspectrum and same-topic arguments from IBM Debater.
* **Class 0 (Different / Opposing Perspective, $N = 228$):**
  - Statements advocating opposing stances on the same topic, or addressing completely unrelated debate propositions.
  - Sourced from same-claim pro vs. con pairs in Kialo and IBM Debater.

### Claim-Level Split Partitioning:
* **Train Split ($60\%$, 198 claims, 478 pairs):** Used for fitting cross-encoders and initial parameter searches.
* **Calibration Split ($20\%$, 66 claims, 124 pairs):** Used strictly for selecting thresholds $\tau^*$ and $\theta^*$ and fitting probability calibrators.
* **Test Split ($20\%$, 67 claims, 147 pairs):** Held-out partition for unbiased final generalization reporting.
* **Cross-Split Claim Leakage:** Strictly **$0.0\%$**.

---

## 3. Stage 1: Dense Retrieval Calibration

Dense bi-encoders compress long propositions into single dense vectors. Stage 1 serves strictly as a high-recall candidate retrieval filter.

### The Primary Selection Rule:
$$\tau^* = \max \tau \quad \text{subject to} \quad \text{CandidateRecall}(\tau) \ge 0.98$$

Where candidate recall evaluates Class 2 (Equivalent) pairs:
$$\text{CandidateRecall}(\tau) = \frac{|\{(g_i, p_j) \in \mathcal{E}_{\text{equivalent}} : \cos(\mathbf{e}_{g_i}, \mathbf{e}_{p_j}) \ge \tau\}|}{|\mathcal{E}_{\text{equivalent}}|}$$

### Empirical Calibration Results (Calibration Split $\to$ Held-Out Test Split):
* **$\tau^*(98\% \text{ target}) = 0.77$:**
  - Calibration: Recall = $100.0\%$, Negative Rejection = $100.0\%$
  - Held-out Test Generalization: Recall = $100.0\%$, Negative Rejection = $100.0\%$
* **$\tau^*(95\% \text{ target}) = 0.78$:**
  - Calibration: Recall = $96.55\%$, Negative Rejection = $100.0\%$

---

## 4. Stage 2: Perspective Verification & Probability Calibration

For candidate pairs passing Stage 1, a cross-encoder evaluates token-level cross-attention:
$$\mathcal{M}_{\text{equiv}}(g_i, p_j) \to p_{\text{equiv}} \in [0, 1]$$

### 4.1 Precision-Constrained Selection Rule:
$$\theta^* = \arg\max_\theta \text{Recall}(\theta) \quad \text{subject to} \quad \text{Precision}(\theta) \ge 0.90$$

Only Class 2 (Equivalent) is credited as a positive match. Class 1 (Related) and Class 0 (Different) are strictly non-equivalent.

### 4.2 Probability Calibration:
Fitted strictly on the Calibration split via Platt scaling:
* **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* Calibration Split ECE: $0.1479$ | Brier Score: $0.0642$
* Held-Out Test Split ECE: $0.1494$ | Brier Score: $0.0813$

---

## 5. Stage 3: Maximum Bipartite Matching on Equivalence Clusters

### The Greedy Fallacy:
Under the historical greedy metric:
$$\text{Covered}_j = \mathbb{I}\left(\max_i S_{i, j} \ge \tau\right)$$
A model that repeats a single argument 5 times can cover multiple distinct human reference clusters simultaneously.

### The Corrected Bipartite Matching Formulation:
Implemented in `src/analysis/bipartite_matching.py`:
1. Constructs bipartite graph $\mathcal{G} = (\mathcal{V}_{\text{gen}}, \mathcal{V}_{\text{clusters}}, \mathcal{E})$.
2. Weighted cost matrix: $C_{i, j} = -1000.0 - S_{i, j}$ for valid edges ($S_{i, j} > 0$), $0$ otherwise.
3. Solves optimal assignment via `scipy.optimize.linear_sum_assignment`.

### Invariance Properties:
1. $\forall g_i \in \mathcal{V}_{\text{gen}}, \ \text{deg}_{\mathcal{M}^*}(g_i) \le 1$ (Each generation covers at most one cluster).
2. $\forall C_j \in \mathcal{V}_{\text{clusters}}, \ \text{deg}_{\mathcal{M}^*}(C_j) \le 1$ (Each cluster is credited at most once).
3. $\text{Coverage}(\mathcal{C}) = \frac{|\mathcal{M}^*|}{M_{\text{clusters}}} \in [0.0, 1.0]$.

