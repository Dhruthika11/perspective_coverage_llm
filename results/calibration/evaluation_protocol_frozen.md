# Frozen Perspective Coverage & Verification Protocol (Final)

This document formalizes the frozen evaluation and verification protocol for Perspective Coverage on the Perspectrum benchmark, specifying the annotation standard, verifier architecture, thresholds, matching algorithm, and metric calculations.

---

## 1. Primary Annotation Reference Standard
* **Primary Reference Standard**: **Version C (Adjudicated Reference)**.
* **Corpus Size**: $N = 341$ candidate pairs across 65 claims (153 Train, 100 Calibration, 88 Held-Out Test).
* **Provenance**:
  - Incorporates the blinded independent re-audit of 90 pairs (55 human vs. LLM jury disagreements + 35 consensus controls).
  - Test-retest consistency on controls: **82.86% agreement (29/35)**, Cohen's $\kappa = 0.7328$. Zero crossing between Class 2 matches and Class 0 polarity/noise.
  - The remaining 251 pairs retain their original verified annotations.
* **Class Distribution (Version C)**:
  - Class 2 (True Semantic Match): 89 pairs (26.10%)
  - Class 1 (Hard Negative / Distinct Reasoning): 136 pairs (39.88%)
  - Class 0a (Opposite Stance): 83 pairs (24.34%)
  - Class 0b (Cross-Claim Noise): 33 pairs (9.68%)

---

## 2. Frozen Verifier Architecture ($M_0$)
* **Model ID**: Reasoning-Aware Feature Cascade ($M_0$).
* **Components**:
  1. **Stage 1 (Topical Retrieval Gate)**: Bi-Encoder BGE-Large cosine similarity $\ge 0.550$.
  2. **Stage 2a (Stance Consistency Gate)**: Contradictory polarities (`Support` $\leftrightarrow$ `Undermine`) are strictly pruned. Viewpoints labeled `Unknown` stance are permitted if semantic reasoning passes.
  3. **Stage 2b (Warrant Probe)**: Calibrated logistic probe over DeBERTa-v3-large NLI probabilities (`p_entail`, `p_neutral`) and BGE-Large cosine similarity:
     $$z = w_{\text{cos}} \cdot \tilde{z}_{\text{cos}} + w_{\text{ent}} \cdot \tilde{z}_{\text{ent}} + w_{\text{neu}} \cdot \tilde{z}_{\text{neu}} + b$$
     $$P(M_0) = \frac{1}{1 + e^{-z}}$$
* **Operating Threshold ($T^*$)**: **$T^* = 0.20$**.
  - **Calibration Provenance**: Selected strictly on the 100-pair Calibration partition (16 claims) to enforce Candidate Recall $\ge 95\%$ on Class 2 matches (achieving 100% on Version C calibration pairs).
  - **Held-Out Test Enforcement**: Frozen permanently before evaluating the 88 held-out test pairs. **Zero threshold tuning was performed on the 17 held-out test claims.**

---

## 3. Matching Algorithm & Eligibility Threshold Reconciliation
* **Algorithm**: Kuhn–Munkres Maximum Weight Bipartite Matching (`scipy.optimize.linear_sum_assignment`).
* **Pre-Filtering vs. Post-Filtering**:
  - **Threshold Application**: Applied **strictly BEFORE bipartite matching** (sparse edge admission).
  - Any candidate edge $(v_i, c_j)$ failing Stage 1 ($\text{cosine} < 0.55$), failing Stage 2a (stance contradiction), or having probe probability $P(M_0) < T^* = 0.20$ is pruned (edge weight set to 0.0 and removed from the active bipartite graph).
* **Reconciliation of $0.20$ vs. $0.50$ Thresholds**:
  - **$T^* = 0.20$**: The continuous probabilistic threshold used by $M_0$ to admit candidate edges into bipartite matching.
  - **$0.50$**: A discrete threshold used strictly in prototype scripts when evaluating binary $\{0.0, 1.0\}$ ground-truth labels (where $\text{weight} \ge 0.50$ selects labeled Class 2 edges). In production, $M_0$ uses $T^* = 0.20$ continuously.
* **Mathematical Invariance**:
  - Injective: Each generated viewpoint covers at most one human perspective cluster.
  - Non-redundant: Each gold perspective cluster is counted at most once per evaluation.
  - Strict Bounding: Coverage is bounded in $[0.0, 1.0]$.

---

## 4. Metric Levels: Candidate-Pair Level vs. Matched-Cluster Level
* **Candidate-Pair Level Metrics (Diagnostic Verifier Evaluation)**:
  - Unit of analysis: Candidate pairs $(g, h)$.
  - Metrics: Class 2 Recall ($\frac{\text{TP}}{\text{True C2}}$), Class 2 Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$), Class 1 Rejection Rate, Class 0a Rejection Rate.
  - Scope: Evaluates intrinsic discrimination across the 88 held-out annotated pairs.
* **Matched-Cluster Level Metrics (Downstream Perspective Coverage)**:
  - Unit of analysis: Per-claim gold perspective clusters and generated viewpoints.
  - **Micro Coverage**:
    $$\text{Micro Coverage} = \frac{\sum_{c=1}^{|C|} \text{Matched Clusters}_c}{\sum_{c=1}^{|C|} \text{Gold Clusters}_c}$$
  - **Macro Coverage**:
    $$\text{Macro Coverage} = \frac{1}{|C|} \sum_{c=1}^{|C|} \frac{\text{Matched Clusters}_c}{\text{Gold Clusters}_c}$$
  - **Argumentative Efficiency**:
    $$\text{Argumentative Efficiency} = \frac{\sum_{c=1}^{|C|} \text{Matched Clusters}_c}{\sum_{c=1}^{|C|} \text{Viewpoints Generated}_c}$$
  - Scope: Evaluates end-to-end downstream coverage across the 17 common held-out claims ($N_{\text{gold}} = 173$).
