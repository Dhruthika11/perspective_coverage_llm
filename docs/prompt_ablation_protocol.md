# Controlled Prompt Ablation Protocol (P1 – P4)

**Phase 2 Implementation Deliverable**  
**Repository:** `perspective_coverage`

---

## 1. Research Objective

To isolate the causal contributions of **aspect conditioning**, **cumulative history**, and **explicit reasoning criteria** on perspective generation quality and reference coverage.

---

## 2. The 4 Experimental Conditions

| Condition | Aspect Conditioning | Cumulative History | Reasoning Criterion | Abstention | Scientific Role |
|---|:---:|:---:|:---:|:---:|---|
| **P1** | **Yes** (5 aspects) | **Yes** | **No** | **No** | **Historical Baseline:** Replicates the original pipeline prompt (atomic argument with history). |
| **P2** | **Yes** (5 aspects) | **Yes** | **Yes** | **Yes** | **Improved Main Prompt:** Full framework with semantic novelty constraints and abstention. |
| **P3** | **Yes** (5 aspects) | **No** | **Yes** | **Yes** | **Ablation (No Memory):** Tests whether reasoning criteria alone suffice without history context. |
| **P4** | **No** (Open aspect) | **Yes** | **Yes** | **Yes** | **Ablation (No Aspects):** Tests whether aspect prompts are necessary or if open sampling with criteria works equally well. |

---

## 3. Controlled Experimental Constants

To guarantee internal validity across all 4 conditions, the following hyperparameters remain strictly invariant:
1. **Model Checkpoint:** Identical model weights and revision.
2. **Claim Set:** Identical sample of controversial claims.
3. **Generation Budget:** Exactly $K = 5$ attempts per claim ($K=8$ for sensitivity analysis on subset).
4. **Sampling Parameters:**
   - Temperature: $T = 0.80$
   - Top-$p$: $0.95$
   - Max New Tokens: $512$
   - Base Seed: $42$ (per-step seed: $42 + \text{step}$)
5. **Aspect Order:** Economic $\to$ Moral $\to$ Legal $\to$ Practical $\to$ Social (with optional Latin-square counterbalancing for sensitivity).
6. **Parsing Rules:** Handled by `src/generation/prompt_templates.py` with identical fallback rules.

---

## 4. Evaluated Metrics for Prompt Comparison

| Metric | Scientific Question Answered | Formula / Definition |
|---|---|---|
| **Cluster Reference Coverage ($C_K$)** | Does explicit prompting increase human perspective coverage? | $\frac{\|\mathcal{M}^*\|}{M_{\text{clusters}}}$ |
| **Marginal Coverage ($\Delta C_k$)** | Does history prevent diminishing returns across steps? | $C_k - C_{k-1}$ |
| **Redundancy Rate** | Does semantic novelty instruction reduce repetition? | $\frac{N_{\text{duplicate\_cluster}}}{K \times N_{\text{claims}}}$ |
| **Abstention Rate** | Does the model accurately recognize empty aspect domains? | $\frac{N_{\text{abstentions}}}{K \times N_{\text{claims}}}$ |
| **Stance Balance (Support / Undermine)** | Does prompting alter the intrinsic stance bias? | $\frac{N_{\text{support}}}{N_{\text{undermine}}}$ |
| **Output Validity Rate** | Does structured reasoning hurt JSON compliance? | $\frac{N_{\text{valid}}}{N_{\text{attempts}}}$ |

---

## 5. Execution Interface

Generation for any ablation condition is invoked via `src/generation/perspective_generator.py`:

```bash
# Example: Running P2 condition on a 20-claim pilot sample
python -c "
from src.generation.perspective_generator import generate_perspectives_for_dataset
generate_perspectives_for_dataset(prompt_version='p2', max_claims=20)
"
```

All results are automatically segregated into:
- `data/llm_outputs/<model>/p1/`
- `data/llm_outputs/<model>/p2/`
- `data/llm_outputs/<model>/p3/`
- `data/llm_outputs/<model>/p4/`
ensuring that historical baseline data is never overwritten.

