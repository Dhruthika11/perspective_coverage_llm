import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def audit():
    df = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    
    train = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))]
    test = df[df['partition'] == 'test'].copy()
    
    y_tr = (train['human_ground_truth_label'] == 'class_2').astype(int).values
    
    # Train M0
    feats_m0 = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc_m0 = StandardScaler()
    X_tr_m0 = sc_m0.fit_transform(train[feats_m0].values)
    clf_m0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m0, y_tr)
    
    # Train M2 v1.1
    feats_m2 = ['cosine_similarity', 'p_entail', 'p_neutral', 'reasoning_cosine_v1_1']
    sc_m2 = StandardScaler()
    X_tr_m2 = sc_m2.fit_transform(train[feats_m2].values)
    clf_m2 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m2, y_tr)
    
    test['prob_m0'] = clf_m0.predict_proba(sc_m0.transform(test[feats_m0].values))[:, 1]
    test['prob_m2'] = clf_m2.predict_proba(sc_m2.transform(test[feats_m2].values))[:, 1]
    
    # Cascade filters
    is_contra = (
        ((test['stance_gold'] == 'support') & (test['stance_generated'] == 'undermine')) |
        ((test['stance_gold'] == 'undermine') & (test['stance_generated'] == 'support'))
    )
    s1_pass = test['cosine_similarity'] >= 0.55
    s2a_pass = s1_pass & (~is_contra)
    
    test['pass_s2a'] = s2a_pass
    test['pass_m0_020'] = s2a_pass & (test['prob_m0'] >= 0.20)
    test['pass_m2_018'] = s2a_pass & (test['prob_m2'] >= 0.18)
    test['pass_m2_031'] = s2a_pass & (test['prob_m2'] >= 0.31)
    
    # Load extracted v1.1 texts for auditing
    with open("data/calibration/e3_1_extracted_components_cache.json", "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    def get_comp(txt):
        import hashlib
        tid = hashlib.sha256(txt.strip().encode('utf-8')).hexdigest()[:16]
        return cache.get(tid, {})
        
    print("=" * 100)
    print("AUDIT E3.2: CLASS 2 FALSE NEGATIVE ANALYSIS ON TEST PARTITION (24 Class 2 Pairs)")
    print("=" * 100)
    
    c2_test = test[test['human_ground_truth_label'] == 'class_2'].copy()
    
    fn_m0 = c2_test[~c2_test['pass_m0_020']]
    fn_m2_018 = c2_test[~c2_test['pass_m2_018']]
    fn_m2_031 = c2_test[~c2_test['pass_m2_031']]
    
    print(f"Total C2 test pairs: {len(c2_test)}")
    print(f"M0 (T*=0.20) False Negatives: {len(fn_m0)} pairs (Recall: {(len(c2_test)-len(fn_m0))/len(c2_test)*100:.1f}%)")
    print(f"M2 (T*=0.18) False Negatives: {len(fn_m2_018)} pairs (Recall: {(len(c2_test)-len(fn_m2_018))/len(c2_test)*100:.1f}%)")
    print(f"M2 (T*=0.31) False Negatives: {len(fn_m2_031)} pairs (Recall: {(len(c2_test)-len(fn_m2_031))/len(c2_test)*100:.1f}%)")
    
    print("\nDetailed inspection of M2 @ 0.18 False Negatives:")
    for _, row in fn_m2_018.iterrows():
        g_c = get_comp(row['gold_perspective_text'])
        h_c = get_comp(row['generated_viewpoint_text'])
        print(f"\n--- Pair ID: {row['pair_id']} (Claim ID {row['claim_id']}: '{row['claim_text']}') ---")
        print(f"  Gold Text:       {row['gold_perspective_text']}")
        print(f"  Gold Reasoning:  {g_c.get('core_reasoning', 'N/A')}")
        print(f"  Gen Text:        {row['generated_viewpoint_text']}")
        print(f"  Gen Reasoning:   {h_c.get('core_reasoning', 'N/A')}")
        print(f"  Stage 1 BGE Cos: {row['cosine_similarity']:.4f} (pass={row['cosine_similarity']>=0.55})")
        print(f"  Stage 2a Stance: gold={row['stance_gold']}, gen={row['stance_generated']} (pass={row['pass_s2a']})")
        print(f"  Reasoning Cos:   {row['reasoning_cosine_v1_1']:.4f}")
        print(f"  Probe M0 Prob:   {row['prob_m0']:.4f} (pass M0@0.20: {row['pass_m0_020']})")
        print(f"  Probe M2 Prob:   {row['prob_m2']:.4f} (pass M2@0.18: {row['pass_m2_018']})")
        
    print("\n" + "=" * 100)
    print("AUDIT E3.3: CLASS 1 ERROR ANALYSIS ON TEST PARTITION (29 Class 1 Pairs)")
    print("=" * 100)
    
    c1_test = test[test['human_ground_truth_label'] == 'class_1'].copy()
    
    # Class 1 pairs should be REJECTED.
    # A true rejection is when pass == False (i.e. model correctly identifies non-equivalence)
    # A false positive is when pass == True.
    # "Recovered" means rejected by M2 but accepted by M0.
    rec_c1_018 = c1_test[c1_test['pass_m0_020'] & (~c1_test['pass_m2_018'])]
    rec_c1_031 = c1_test[c1_test['pass_m0_020'] & (~c1_test['pass_m2_031'])]
    lost_c1_018 = c1_test[(~c1_test['pass_m0_020']) & c1_test['pass_m2_018']]
    lost_c1_031 = c1_test[(~c1_test['pass_m0_020']) & c1_test['pass_m2_031']]
    
    print(f"Total C1 test pairs: {len(c1_test)}")
    print(f"M0 @ 0.20 C1 rejections: {(~c1_test['pass_m0_020']).sum()} / {len(c1_test)} ({((~c1_test['pass_m0_020']).sum())/len(c1_test)*100:.1f}%)")
    print(f"M2 @ 0.18 C1 rejections: {(~c1_test['pass_m2_018']).sum()} / {len(c1_test)} ({((~c1_test['pass_m2_018']).sum())/len(c1_test)*100:.1f}%)")
    print(f"M2 @ 0.31 C1 rejections: {(~c1_test['pass_m2_031']).sum()} / {len(c1_test)} ({((~c1_test['pass_m2_031']).sum())/len(c1_test)*100:.1f}%)")
    print(f"\nPairs newly rejected by M2 @ 0.18 (Recovered True Negatives): {len(rec_c1_018)}")
    print(f"Pairs newly accepted by M2 @ 0.18 (New False Positives): {len(lost_c1_018)}")
    print(f"Pairs newly rejected by M2 @ 0.31 (Recovered True Negatives): {len(rec_c1_031)}")
    print(f"Pairs newly accepted by M2 @ 0.31 (New False Positives): {len(lost_c1_031)}")
    
    print("\n--- Detailed Audit of Recovered Class 1 Pairs at T*=0.18 ---")
    for _, row in rec_c1_018.iterrows():
        g_c = get_comp(row['gold_perspective_text'])
        h_c = get_comp(row['generated_viewpoint_text'])
        print(f"\nPair ID: {row['pair_id']} | Claim {row['claim_id']}: '{row['claim_text']}'")
        print(f"  Gold Text:       {row['gold_perspective_text']}")
        print(f"  Gold Reasoning:  {g_c.get('core_reasoning', 'N/A')}")
        print(f"  Gen Text:        {row['generated_viewpoint_text']}")
        print(f"  Gen Reasoning:   {h_c.get('core_reasoning', 'N/A')}")
        print(f"  BGE Cosine:      {row['cosine_similarity']:.4f}")
        print(f"  Reasoning Cos:   {row['reasoning_cosine_v1_1']:.4f}")
        print(f"  M0 Prob:         {row['prob_m0']:.4f} (Accepted by M0)")
        print(f"  M2 Prob:         {row['prob_m2']:.4f} (Correctly Rejected by M2)")
        
    print("\n--- Detailed Audit of All 7 Recovered Class 1 Pairs at T*=0.31 ---")
    for _, row in rec_c1_031.iterrows():
        g_c = get_comp(row['gold_perspective_text'])
        h_c = get_comp(row['generated_viewpoint_text'])
        print(f"\nPair ID: {row['pair_id']} | Claim {row['claim_id']}: '{row['claim_text']}'")
        print(f"  Gold Text:       {row['gold_perspective_text']}")
        print(f"  Gold Reasoning:  {g_c.get('core_reasoning', 'N/A')}")
        print(f"  Gen Text:        {row['generated_viewpoint_text']}")
        print(f"  Gen Reasoning:   {h_c.get('core_reasoning', 'N/A')}")
        print(f"  BGE Cosine:      {row['cosine_similarity']:.4f}")
        print(f"  Reasoning Cos:   {row['reasoning_cosine_v1_1']:.4f}")
        print(f"  M0 Prob:         {row['prob_m0']:.4f} (Accepted by M0)")
        print(f"  M2 Prob:         {row['prob_m2']:.4f} (Correctly Rejected by M2)")

if __name__ == '__main__':
    audit()
