import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv('results/calibration/e3_pairwise_features_and_predictions_341.csv')
test_df = df[df['partition'] == 'test'].copy()

# Filters
is_contra = (
    ((test_df['stance_gold'] == 'support') & (test_df['stance_generated'] == 'undermine')) |
    ((test_df['stance_gold'] == 'undermine') & (test_df['stance_generated'] == 'support'))
)
s1_pass = test_df['cosine_similarity'] >= 0.55
s2a_pass = s1_pass & (~is_contra)

test_claims = sorted(test_df['claim_id'].unique())
n_claims = len(test_claims)
print(f"Number of held-out test claims: {n_claims}")

# Precompute per-claim data
claim_data = {}
for cid in test_claims:
    sub = test_df[test_df['claim_id'] == cid]
    claim_data[cid] = {
        'labels': sub['human_ground_truth_label'].values,
        's2a_pass': s2a_pass.loc[sub.index].values,
        'prob_m0': sub['prob_M0_baseline'].values,
        'prob_m2': sub['prob_M2_reasoning'].values,
        'y_c2': (sub['human_ground_truth_label'] == 'class_2').values,
        'y_c1': (sub['human_ground_truth_label'] == 'class_1').values,
    }

def run_paired_bootstrap(t_m0, t_m2, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    
    delta_f1 = []
    delta_rec = []
    delta_prec = []
    delta_c1_rej = []
    delta_auc = []
    
    for _ in range(n_boot):
        sampled_cids = rng.choice(test_claims, size=n_claims, replace=True)
        
        # Assemble bootstrap sample
        b_labels = np.concatenate([claim_data[c]['labels'] for c in sampled_cids])
        b_s2a = np.concatenate([claim_data[c]['s2a_pass'] for c in sampled_cids])
        b_prob_m0 = np.concatenate([claim_data[c]['prob_m0'] for c in sampled_cids])
        b_prob_m2 = np.concatenate([claim_data[c]['prob_m2'] for c in sampled_cids])
        
        b_c2 = (b_labels == 'class_2')
        b_c1 = (b_labels == 'class_1')
        b_non_c2 = (~b_c2)
        
        tot_c2 = b_c2.sum()
        tot_c1 = b_c1.sum()
        if tot_c2 == 0 or tot_c1 == 0:
            continue
            
        # M0 evaluation
        m0_pass = b_s2a & (b_prob_m0 >= t_m0)
        tp_m0 = (m0_pass & b_c2).sum()
        fp_m0 = (m0_pass & b_non_c2).sum()
        fp_c1_m0 = (m0_pass & b_c1).sum()
        
        rec_m0 = tp_m0 / tot_c2
        prec_m0 = tp_m0 / max(tp_m0 + fp_m0, 1)
        f1_m0 = 2 * prec_m0 * rec_m0 / max(prec_m0 + rec_m0, 1e-8)
        c1_rej_m0 = (tot_c1 - fp_c1_m0) / tot_c1
        
        # M2 evaluation
        m2_pass = b_s2a & (b_prob_m2 >= t_m2)
        tp_m2 = (m2_pass & b_c2).sum()
        fp_m2 = (m2_pass & b_non_c2).sum()
        fp_c1_m2 = (m2_pass & b_c1).sum()
        
        rec_m2 = tp_m2 / tot_c2
        prec_m2 = tp_m2 / max(tp_m2 + fp_m2, 1)
        f1_m2 = 2 * prec_m2 * rec_m2 / max(prec_m2 + rec_m2, 1e-8)
        c1_rej_m2 = (tot_c1 - fp_c1_m2) / tot_c1
        
        # C2 vs C1 AUC
        mask_c21 = b_c2 | b_c1
        try:
            auc_m0 = roc_auc_score(b_c2[mask_c21].astype(int), b_prob_m0[mask_c21])
            auc_m2 = roc_auc_score(b_c2[mask_c21].astype(int), b_prob_m2[mask_c21])
            d_auc = auc_m2 - auc_m0
        except Exception:
            d_auc = 0.0
            
        delta_f1.append(f1_m2 - f1_m0)
        delta_rec.append(rec_m2 - rec_m0)
        delta_prec.append(prec_m2 - prec_m0)
        delta_c1_rej.append(c1_rej_m2 - c1_rej_m0)
        delta_auc.append(d_auc)
        
    def stats(arr):
        arr = np.array(arr)
        mean = float(np.mean(arr))
        ci_l = float(np.percentile(arr, 2.5))
        ci_h = float(np.percentile(arr, 97.5))
        p = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        p = min(1.0, max(1.0 / len(arr), p))
        return mean, ci_l, ci_h, p
        
    return {
        'F1': stats(delta_f1),
        'Recall': stats(delta_rec),
        'Precision': stats(delta_prec),
        'C1_Rejection': stats(delta_c1_rej),
        'AUC': stats(delta_auc)
    }

print("=" * 85)
print("1. PAIRED CLAIM-CLUSTERED BOOTSTRAP: M2 (T*=0.31, Cal F1-Opt) vs M0 (T*=0.20)")
print("=" * 85)
res_31 = run_paired_bootstrap(0.20, 0.31)
for k, (mean, l, h, p) in res_31.items():
    print(f"  Delta {k:<14}: Mean = {mean:+.4f} | 95% Clustered CI = [{l:+.4f}, {h:+.4f}] | p-value = {p:.4f}")

print("\n" + "=" * 85)
print("2. PAIRED CLAIM-CLUSTERED BOOTSTRAP: M2 (T*=0.18, Cal Recall>=90%) vs M0 (T*=0.20)")
print("=" * 85)
res_18 = run_paired_bootstrap(0.20, 0.18)
for k, (mean, l, h, p) in res_18.items():
    print(f"  Delta {k:<14}: Mean = {mean:+.4f} | 95% Clustered CI = [{l:+.4f}, {h:+.4f}] | p-value = {p:.4f}")
