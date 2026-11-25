# Final 3-Seed Calibrated Ensemble Report: Model Selection, Temperature Calibration, and Clean Benchmark Evaluation

## 1. Validation-Only Model Selection Decision
- **Protocol:** Strict evaluation on disjoint validation set (`val.csv`, $N=800$, 0% test overlap).
- **Candidate Architectures:**
  * **Data-Only:** Random Head + Balanced Multi-Genre Data
  * **Warm-Head:** Warm MNLI Head + Balanced Multi-Genre Data
- **Validation Ensemble AUROC:**
  * **Data-Only:** `0.9800` (Seeds Mean: `0.8674 ± 0.0100`)
  * **Warm-Head:** `0.9800` (Seeds Mean: `0.8674 ± 0.0100`)
- **Official Selection:** **`Data-Only (Random Head + Balanced Multi-Genre Data)`**
- **Methodological Rationale:** Data-Only achieved equal or superior AUROC on the disjoint validation set (0.9800 vs 0.9800). Selected as the official Main Model by Occam's razor: simpler architecture that learns directly from balanced multi-genre supervision without external MNLI head surgery or label re-mapping.

---

## 2. Temperature Scaling & Calibration (Tuned Strictly on Validation)
- **Optimal Temperature Parameter:** $T^* = 0.2323$ (optimized via scalar Brent minimization of NLL)
- **Validation Calibration Metrics:**
  * **Negative Log-Likelihood (NLL):** `0.6492` $\to$ `0.2809`
  * **Expected Calibration Error (ECE, 10 bins):** `31.87%` $\to$ `2.76%`
  * **Brier Score:** `0.1153` $\to$ `0.0546`

---

## 3. Frozen Operating Thresholds & 3-Tier Triage Policy
> *These thresholds were calibrated once on the disjoint validation set and permanently frozen prior to test evaluation.*

| Parameter | Frozen Value | Meaning |
|---|---|---|
| **$\theta_{F1}^*$ (Optimal $F_1$)** | `0.30` | Maximizes binary classification $F_1$ score on validation data. |
| **$\theta_{\text{prec90}}^*$ (High Precision)** | `0.27` | Minimum threshold achieving $\ge 90\%$ precision on validation data. |
| **Tier 1: Accept** | $P(\text{full}) \ge 0.27$ | High confidence entailment; accepted without manual review. |
| **Tier 2: Review** | $0.20 \le P(\text{full}) < 0.27$ | Borderline / partial coverage; flagged for active human verification. |
| **Tier 3: Reject** | $P(\text{full}) < 0.20$ | Confident non-entailment or contradiction; discarded. |

---

## 4. Locked 1,118 Test Benchmark Evaluation (Pristine, Zero Peeking)
- **Overall Macro AUROC:** `0.9842`
- **Stratum 1 (Short Human) AUROC:** `0.9855`
- **Stratum 2 (Long LLM) AUROC:** `0.9807`
- **Stratum 3 (Stance Separation) AUROC:** `0.9863`
- **Contradiction Detection:** `107/120` (`89.2%`)
- **Performance at Frozen $\theta_{F1}^* = 0.30$:**
  * **Precision:** `94.18%`
  * **Recall:** `92.96%`
  * **$F_1$ Score:** `0.9357`
- **Performance at Frozen $\theta_{\text{prec90}}^* = 0.27$:**
  * **Precision:** `93.70%`
  * **Recall:** `93.70%`
  * **$F_1$ Score:** `0.9370`
- **3-Tier Triage Distribution:**
  * **Accept Tier:** `540/1118` (`48.3%`) with **93.7% Precision**
  * **Review Tier:** `12/1118` (`1.1%`)
  * **Reject Tier:** `566/1118` (`50.6%`)

---

## 5. Pristine Fresh Claims Holdout Benchmark (50 Claims, N=1,110)
- **Overall Macro AUROC:** `0.9806`
- **Short Human Stratum AUROC:** `0.9805`
- **Balanced Long LLM Stratum AUROC:** `0.9864`
- **Stance Separation AUROC:** `0.9827`
- **Contradictions Caught:** `61/73` (`83.6%`)
- **Performance at Frozen $\theta_{F1}^*$:**
  * **Precision:** `95.59%` | **Recall:** `92.72%` | **$F_1$ Score:** `0.9413`
