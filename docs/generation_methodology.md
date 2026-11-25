# Perspective Generation Methodology (Phase 2)

**Phase 2 Implementation Deliverable**  
**Repository:** `perspective_coverage`

---

## 1. Overview and Scientific Motivation

Standard generative evaluation prompts ask large language models to "list viewpoints on a topic" or "argue both sides". This approach suffers from three fundamental failure modes:
1. **Superficial Paraphrasing:** Models generate repetitive statements that share the same underlying logic under different phrasing (e.g. *"reduces transit costs"* and *"saves money on train tickets"*).
2. **Forced Hallucinated Arguments:** When constrained to an aspect that has no compelling arguments (e.g., an Economic argument for a purely aesthetic or metaphysical claim), models hallucinate contrived reasoning rather than recognizing that no valid perspective exists under that aspect.
3. **Loss of Provenance:** Unversioned prompts make it impossible to disentangle whether coverage gains stem from model scaling or from prompt design.

The Phase 2 architecture resolves these limitations by introducing **explicit reasoning criteria**, **semantic novelty constraints**, **structured abstention**, and **versioned prompt ablations**.

---

## 2. Core Methodological Components

### 2.1 The 5 Canonical Deliberative Aspects
The generation framework uses 5 fixed deliberative aspects:
1. **Economic:** Resource allocation, government expenditure, affordability, market efficiency, employment.
2. **Moral:** Fairness, equality, human rights, individual responsibility, bodily autonomy, dignity.
3. **Legal:** Constitutional obligations, statutory authority, regulatory compliance, judicial precedent.
4. **Practical:** Implementation feasibility, operational logistics, accessibility, infrastructure, enforcement.
5. **Social:** Community cohesion, public trust, social inequality, cultural cohesion, demographic behavior.

### 2.2 Explicit Reasoning Criteria
Rather than asking for free-form prose, the prompt explicitly requires the model to state the underlying **reasoning criterion** (the fundamental value or consideration) that justifies the argument.

*Example:*
- Aspect: **Moral**
- Stance: **Support**
- Reasoning Criterion: **Equality of Opportunity**
- Argument: *"Universal broadband access ensures students in low-income districts have equal access to educational resources, fulfilling the moral obligation to provide fair opportunity."*

*Example:*
- Aspect: **Economic**
- Stance: **Undermine**
- Reasoning Criterion: **Fiscal Deficit**
- Argument: *"Subsidizing universal higher education imposes severe fiscal strain on municipal budgets, necessitating deficit spending that triggers broader inflation."*

### 2.3 Semantic Novelty Constraint
The model is explicitly instructed that a paraphrase does NOT qualify as a distinct perspective:
> *"A mere paraphrase or rephrasing of an existing idea is NOT a new perspective. Your argument must introduce a DIFFERENT underlying reasoning criterion or value."*

### 2.4 Structured Abstention
When an LLM cannot produce a genuine, meaningful argument with distinct reasoning under the assigned aspect, it must output an explicit abstention object:
```json
{
  "status": "abstain",
  "aspect": "Legal",
  "reasoning_criterion": "none",
  "stance": "none",
  "argument": "No distinct legal argument remains uncovered."
}
```
Abstentions are logged explicitly as `is_abstention = True`. They are **never** silently discarded or replaced with retry attempts.

---

## 3. Machine-Readable Schema

Output JSON schema enforced across all generations:
```json
{
  "status": "argument",
  "aspect": "Economic",
  "reasoning_criterion": "affordability",
  "stance": "support",
  "argument": "1-3 sentences stating the perspective and its underlying justification"
}
```

---

## 4. Full Generation Logging Specification

For every generation step, the engine records 15 explicit metadata fields in `raw_generations.jsonl`:
1. `claim_id` (int): Claim identifier.
2. `model_id` (str): Full Hugging Face model identifier and revision.
3. `prompt_version` (str): `'p1'`, `'p2'`, `'p3'`, or `'p4'`.
4. `aspect` (str): Assigned deliberative aspect.
5. `generation_step` (int): Sequential index ($0 \dots K-1$).
6. `previous_history` (list[str]): Text of prior arguments presented in the prompt context.
7. `temperature` (float): Sampling temperature (e.g., $0.80$).
8. `random_seed` (int): Seed for deterministic reproducibility ($\text{seed} + \text{step}$).
9. `token_budget` (int): Maximum new tokens (e.g., $512$).
10. `raw_model_output` (str): Exact, unstripped string returned by the model.
11. `parsed_output` (dict): Extracted JSON fields.
12. `is_valid` (bool): True if schema and stance constraints were satisfied.
13. `stance` (str): Extracted stance (`support`, `undermine`, `unclear`, `none`).
14. `reasoning_criterion` (str): Extracted value criterion.
15. `is_abstention` (bool): True if model abstained.
16. `generation_time_sec` (float): Exact inference latency.
17. `error_information` (str or null): Exception message if inference failed.

---

## 5. Generation Metrics & Marginal Coverage

To evaluate generation dynamics, we compute:
1. **Cumulative Reference Coverage ($C_k$):** Proportion of reference clusters covered by step $k$.
2. **Marginal Coverage ($\Delta C_k$):** Additional reference coverage gained at step $k$:
   $$\Delta C_k = C_k - C_{k-1}, \quad \text{with } C_0 = 0$$
3. **Redundancy Rate:** Fraction of non-abstained arguments that merely re-cover reference clusters already satisfied by prior steps on the same claim.
4. **Abstention Rate:** Proportion of generation attempts yielding explicit abstention:
   $$\text{Rate}_{\text{abstain}} = \frac{N_{\text{abstain}}}{K \times N_{\text{claims}}}$$
5. **Validity Rate:** Proportion of outputs conforming to the required JSON schema.
6. **Average Argument Length:** Mean word count of generated arguments.

