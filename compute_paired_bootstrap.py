import pandas as pd
import numpy as np
from scipy import stats

models = [
    ('qwen2_5_7b_instruct', 'Qwen2.5-7B'),
    ('llama_3_1_8b_instruct', 'Llama-3.1-8B'),
    ('gemma_2_9b_it', 'Gemma-2-9B'),
    ('olmo_2_1124_7b_instruct', 'OLMo-2-7B')
]

dfs = {}
for m_key, m_name in models:
    path = f"results/{m_key}/metrics/two_stage_per_claim_coverage.csv"
    dfs[m_key] = pd.read_csv(path).set_index('claim_id')

shared_cids = sorted(list(set.intersection(*[set(dfs[m_key].index) for m_key, _ in models])))
print(f"Number of shared claims: {len(shared_cids)}")

# Run pairwise claim-clustered bootstrap tests
np.random.seed(42)
B = 10000

print("\n" + "="*85)
print("PAIRED CLAIM-CLUSTERED BOOTSTRAP TEST RESULTS (STAGE 2 TWO-STAGE COVERAGE)")
print("="*85)
print(f"{'Comparison':<30} | {'Mean Diff':<10} | {'95% CI of Difference':<22} | {'p-value (Boot)':<14} | {'p (Wilcoxon)':<12}")
print("-" * 85)

results = []
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        m1_key, m1_name = models[i]
        m2_key, m2_name = models[j]

        cov1 = dfs[m1_key].loc[shared_cids, 'stage2_v5_coverage_f1'].to_numpy()
        cov2 = dfs[m2_key].loc[shared_cids, 'stage2_v5_coverage_f1'].to_numpy()

        diffs = cov1 - cov2
        obs_mean_diff = float(np.mean(diffs))

        # Bootstrap
        n = len(diffs)
        boot_diffs = []
        for _ in range(B):
            idx = np.random.choice(n, size=n, replace=True)
            boot_diffs.append(np.mean(diffs[idx]))
        boot_diffs = np.array(boot_diffs)

        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)

        # Empirical two-tailed p-value: H0: mean_diff = 0
        p_boot = 2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
        p_boot = min(1.0, max(1.0 / B, p_boot))

        try:
            stat_w, p_wilcoxon = stats.wilcoxon(cov1, cov2)
        except Exception:
            p_wilcoxon = np.nan

        comp_label = f"{m1_name} vs {m2_name}"
        ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        print(f"{comp_label:<30} | {obs_mean_diff:+.4f}    | {ci_str:<22} | {p_boot:<14.4f} | {p_wilcoxon:<12.4f}")
        results.append({
            'comparison': comp_label,
            'mean_diff': obs_mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_boot': p_boot,
            'p_wilcoxon': p_wilcoxon
        })

print("="*85)

