import json
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

test_df = pd.read_csv('data/calibration/human_ground_truth_test_partition_114.csv')
test_claims = sorted(test_df['claim_id'].unique())
n_test_claims = len(test_claims)
print(f"Held-out test partition: {len(test_df)} pairs across {n_test_claims} unique claims.")

# 1. Wilson Score Intervals (treating pairs as independent reference)
n_c2 = (test_df['human_ground_truth_label'] == 'class_2').sum()
pass_c2_55 = ((test_df['human_ground_truth_label'] == 'class_2') & (test_df['cosine_similarity'] >= 0.55)).sum()
pass_c2_60 = ((test_df['human_ground_truth_label'] == 'class_2') & (test_df['cosine_similarity'] >= 0.60)).sum()

w_low_55, w_high_55 = proportion_confint(pass_c2_55, n_c2, alpha=0.05, method='wilson')
w_low_60, w_high_60 = proportion_confint(pass_c2_60, n_c2, alpha=0.05, method='wilson')

print("\n--- 1. Wilson Score 95% Confidence Intervals (N = 26 Class 2 pairs) ---")
print(f"tau = 0.55: {pass_c2_55}/{n_c2} = {pass_c2_55/n_c2*100:.2f}% | 95% Wilson CI: [{w_low_55*100:.2f}%, {w_high_55*100:.2f}%]")
print(f"tau = 0.60: {pass_c2_60}/{n_c2} = {pass_c2_60/n_c2*100:.2f}% | 95% Wilson CI: [{w_low_60*100:.2f}%, {w_high_60*100:.2f}%]")

# 2. Claim-Clustered Bootstrap (Resampling 20 Claims with replacement, 2,000 resamples)
rng = np.random.RandomState(42)
claim_indices = {cid: np.where(test_df['claim_id'].values == cid)[0] for cid in test_claims}

c2_mask = (test_df['human_ground_truth_label'] == 'class_2').values
c1_mask = (test_df['human_ground_truth_label'] == 'class_1').values
c0a_mask = (test_df['human_ground_truth_label'] == 'class_0a').values
c0b_mask = (test_df['human_ground_truth_label'] == 'class_0b').values
sims = test_df['cosine_similarity'].values

def run_cluster_boot(tau, n_boot=2000):
    rec_c2_list = []
    rej_c0b_list = []
    rej_c0a_list = []
    rej_c1_list = []
    pass_rate_list = []
    
    for _ in range(n_boot):
        sampled_cids = rng.choice(test_claims, size=n_test_claims, replace=True)
        idx = np.concatenate([claim_indices[cid] for cid in sampled_cids])
        
        sub_sims = sims[idx]
        sub_c2 = c2_mask[idx]
        sub_c1 = c1_mask[idx]
        sub_c0a = c0a_mask[idx]
        sub_c0b = c0b_mask[idx]
        
        passed = (sub_sims >= tau)
        
        # Metric computations
        if np.sum(sub_c2) > 0:
            rec_c2 = np.sum(passed & sub_c2) / np.sum(sub_c2)
            rec_c2_list.append(rec_c2)
        if np.sum(sub_c0b) > 0:
            rej_c0b = np.sum((~passed) & sub_c0b) / np.sum(sub_c0b)
            rej_c0b_list.append(rej_c0b)
        if np.sum(sub_c0a) > 0:
            rej_c0a = np.sum((~passed) & sub_c0a) / np.sum(sub_c0a)
            rej_c0a_list.append(rej_c0a)
        if np.sum(sub_c1) > 0:
            rej_c1 = np.sum((~passed) & sub_c1) / np.sum(sub_c1)
            rej_c1_list.append(rej_c1)
        pass_rate_list.append(np.sum(passed) / len(idx))
        
    return {
        "tau": tau,
        "c2_recall_ci": (round(float(np.percentile(rec_c2_list, 2.5)), 4), round(float(np.percentile(rec_c2_list, 97.5)), 4)),
        "c0b_rejection_ci": (round(float(np.percentile(rej_c0b_list, 2.5)), 4), round(float(np.percentile(rej_c0b_list, 97.5)), 4)),
        "c0a_rejection_ci": (round(float(np.percentile(rej_c0a_list, 2.5)), 4), round(float(np.percentile(rej_c0a_list, 97.5)), 4)),
        "c1_rejection_ci": (round(float(np.percentile(rej_c1_list, 2.5)), 4), round(float(np.percentile(rej_c1_list, 97.5)), 4)),
        "pass_rate_ci": (round(float(np.percentile(pass_rate_list, 2.5)), 4), round(float(np.percentile(pass_rate_list, 97.5)), 4))
    }

boot_55 = run_cluster_boot(0.55)
boot_60 = run_cluster_boot(0.60)
boot_75 = run_cluster_boot(0.75)

print("\n--- 2. Claim-Clustered Bootstrap 95% Confidence Intervals (2,000 Claim Resamples) ---")
print(f"tau = 0.55:")
print(f"  Class 2 Recall 95% CI:        [{boot_55['c2_recall_ci'][0]*100:.2f}%, {boot_55['c2_recall_ci'][1]*100:.2f}%]")
print(f"  Class 0B Rejection 95% CI:     [{boot_55['c0b_rejection_ci'][0]*100:.2f}%, {boot_55['c0b_rejection_ci'][1]*100:.2f}%]")
print(f"  Class 0A Rejection 95% CI:     [{boot_55['c0a_rejection_ci'][0]*100:.2f}%, {boot_55['c0a_rejection_ci'][1]*100:.2f}%]")
print(f"  Class 1 Rejection 95% CI:      [{boot_55['c1_rejection_ci'][0]*100:.2f}%, {boot_55['c1_rejection_ci'][1]*100:.2f}%]")
print(f"  Stage 1 Pass Rate 95% CI:      [{boot_55['pass_rate_ci'][0]*100:.2f}%, {boot_55['pass_rate_ci'][1]*100:.2f}%]")

print(f"\ntau = 0.60:")
print(f"  Class 2 Recall 95% CI:        [{boot_60['c2_recall_ci'][0]*100:.2f}%, {boot_60['c2_recall_ci'][1]*100:.2f}%]")
print(f"  Class 0B Rejection 95% CI:     [{boot_60['c0b_rejection_ci'][0]*100:.2f}%, {boot_60['c0b_rejection_ci'][1]*100:.2f}%]")
print(f"  Class 0A Rejection 95% CI:     [{boot_60['c0a_rejection_ci'][0]*100:.2f}%, {boot_60['c0a_rejection_ci'][1]*100:.2f}%]")
print(f"  Class 1 Rejection 95% CI:      [{boot_60['c1_rejection_ci'][0]*100:.2f}%, {boot_60['c1_rejection_ci'][1]*100:.2f}%]")
print(f"  Stage 1 Pass Rate 95% CI:      [{boot_60['pass_rate_ci'][0]*100:.2f}%, {boot_60['pass_rate_ci'][1]*100:.2f}%]")

print(f"\ntau = 0.75 (Human-Human):")
print(f"  Class 2 Recall 95% CI:        [{boot_75['c2_recall_ci'][0]*100:.2f}%, {boot_75['c2_recall_ci'][1]*100:.2f}%]")
print(f"  Class 0B Rejection 95% CI:     [{boot_75['c0b_rejection_ci'][0]*100:.2f}%, {boot_75['c0b_rejection_ci'][1]*100:.2f}%]")

# Save to json
out_ci = {
    "wilson_ci_95": {
        "tau_0.55": {"recall": 1.0, "ci_lower": w_low_55, "ci_upper": w_high_55},
        "tau_0.60": {"recall": 1.0, "ci_lower": w_low_60, "ci_upper": w_high_60}
    },
    "claim_clustered_bootstrap_ci_95": {
        "tau_0.55": boot_55,
        "tau_0.60": boot_60,
        "tau_0.75": boot_75
    }
}
with open('results/calibration/held_out_test_claim_clustered_ci.json', 'w') as f:
    json.dump(out_ci, f, indent=2)
