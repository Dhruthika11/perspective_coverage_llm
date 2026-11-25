import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

# Filter to Class 2 (Positive = 1) vs Class 1 (Hard Negative = 0)
c21 = df[df['human_ground_truth_label'].isin(['class_2', 'class_1'])].copy()

train_df = c21[c21['partition'] == 'train'].copy()
cal_df = c21[c21['partition'] == 'calibration'].copy()
test_df = c21[c21['partition'] == 'test'].copy()

y_train = (train_df['human_ground_truth_label'] == 'class_2').astype(int).values
y_cal = (cal_df['human_ground_truth_label'] == 'class_2').astype(int).values
y_test = (test_df['human_ground_truth_label'] == 'class_2').astype(int).values

test_claims = sorted(test_df['claim_id'].unique())
n_test_claims = len(test_claims)
claim_indices = {cid: np.where(test_df['claim_id'].values == cid)[0] for cid in test_claims}

feature_sets = {
    "1. Entailment only": ['p_entail'],
    "2. Cosine only": ['cosine_similarity'],
    "3. Cosine + Entailment": ['cosine_similarity', 'p_entail'],
    "4. Cosine + Entail + Neutral": ['cosine_similarity', 'p_entail', 'p_neutral'],
    "5. All Four Features": ['cosine_similarity', 'p_entail', 'p_neutral', 'p_contra']
}

print("=" * 95)
print("EXPERIMENT B1: SYSTEMATIC FEATURE ABLATION (CLASS 2 VS. CLASS 1)")
print("=" * 95)
print(f"Train Partition (32 Claims):       {len(train_df)} pairs (51 Class 2, 53 Class 1)")
print(f"Calibration Partition (16 Claims): {len(cal_df)} pairs (24 Class 2, 45 Class 1)")
print(f"Held-Out Test Partition (17 Claims): {len(test_df)} pairs (24 Class 2, 29 Class 1)")

results = []
trained_models = {}
test_prob_preds = {}

for name, feats in feature_sets.items():
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[feats].values)
    X_ca = scaler.transform(cal_df[feats].values)
    X_te = scaler.transform(test_df[feats].values)
    
    # Train regularized logistic regression (C=1.0)
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_tr, y_train)
    
    tr_probs = clf.predict_proba(X_tr)[:, 1]
    ca_probs = clf.predict_proba(X_ca)[:, 1]
    te_probs = clf.predict_proba(X_te)[:, 1]
    
    tr_auc = roc_auc_score(y_train, tr_probs)
    ca_auc = roc_auc_score(y_cal, ca_probs)
    te_auc = roc_auc_score(y_test, te_probs)
    te_ap = average_precision_score(y_test, te_probs)
    
    trained_models[name] = clf
    test_prob_preds[name] = te_probs
    
    # Coefficients formatted
    coef_str = ", ".join([f"{f}: {c:+.3f}" for f, c in zip(feats, clf.coef_[0])])
    
    results.append({
        "Model": name,
        "Features": feats,
        "Train_AUC": round(float(tr_auc), 4),
        "Cal_AUC": round(float(ca_auc), 4),
        "Test_AUC": round(float(te_auc), 4),
        "Test_PR_AUC": round(float(te_ap), 4),
        "Standardized_Weights": coef_str
    })

res_df = pd.DataFrame(results)
print("\nAblation Results Summary Table:")
print(res_df[['Model', 'Train_AUC', 'Cal_AUC', 'Test_AUC', 'Test_PR_AUC']].to_string(index=False))

print("\nStandardized Feature Weights (beta coefficients after StandardScaler):")
for r in results:
    print(f"  {r['Model']:30s} -> {r['Standardized_Weights']}")

# Claim-Clustered Bootstrap for Paired Delta AUC vs Model 1 (Entailment Only)
print("\n" + "-" * 95)
print("CLAIM-CLUSTERED PAIRED BOOTSTRAP CI FOR DELTA-AUC (VS. ENTAILMENT ONLY, 2,000 RESAMPLES)")
print("-" * 95)

rng = np.random.RandomState(42)
base_preds = test_prob_preds["1. Entailment only"]

delta_cis = {}
for name in feature_sets.keys():
    if name == "1. Entailment only":
        continue
    cand_preds = test_prob_preds[name]
    
    delta_list = []
    for _ in range(2000):
        sampled_cids = rng.choice(test_claims, size=n_test_claims, replace=True)
        idx = np.concatenate([claim_indices[cid] for cid in sampled_cids])
        
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
            
        auc_base = roc_auc_score(y_b, base_preds[idx])
        auc_cand = roc_auc_score(y_b, cand_preds[idx])
        delta_list.append(auc_cand - auc_base)
        
    ci_low = float(np.percentile(delta_list, 2.5))
    ci_high = float(np.percentile(delta_list, 97.5))
    mean_delta = float(np.mean(delta_list))
    p_val = float(np.mean(np.array(delta_list) <= 0))
    
    delta_cis[name] = {
        "mean_delta_auc": round(mean_delta, 4),
        "ci_95": (round(ci_low, 4), round(ci_high, 4)),
        "p_value_one_tailed": round(p_val, 4)
    }
    
    print(f"{name:30s}: Delta AUC = {mean_delta:+.4f} | 95% Claim-Clustered CI: [{ci_low:+.4f}, {ci_high:+.4f}] | p = {p_val:.4f}")

# Save results
out_data = {
    "ablation_results": results,
    "delta_auc_bootstraps": delta_cis
}
with open('results/calibration/experiment_b1_ablation_report.json', 'w') as f:
    json.dump(out_data, f, indent=2)
