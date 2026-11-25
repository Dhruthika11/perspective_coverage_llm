import json
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 90)
print("COMPREHENSIVE INVESTIGATION OF PRODUCTION AUDIT ERRORS (PRIORITIES 1-5)")
print("=" * 90)

# Load frozen probe configuration
with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
    config = json.load(f)

scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
w_cos = config['coefficients']['cosine_similarity']
w_ent = config['coefficients']['p_entail']
w_neu = config['coefficients']['p_neutral']
intercept = config['intercept']
t_probe = config['calibrated_threshold']

def calc_probe_prob(cos_val, ent_val, neu_val):
    z_cos = (cos_val - scaler_mean[0]) / scaler_scale[0]
    z_ent = (ent_val - scaler_mean[1]) / scaler_scale[1]
    z_neu = (neu_val - scaler_mean[2]) / scaler_scale[2]
    linear = z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept
    return 1.0 / (1.0 + np.exp(-linear))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/argument_nli_deberta_v3_v3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

def run_nli_pair(p, h):
    inputs = tokenizer([p], [h], return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs[0], probs[1], probs[2] # contra, entail, neutral

# ==============================================================================
# PRIORITY 1: RESOLVE PAIR 403's UNKNOWN STANCE FAILURE
# ==============================================================================
print("\n" + "=" * 90)
print("PRIORITY 1: RESOLVING PAIR 403 (UNKNOWN STANCE REBUTTAL)")
print("=" * 90)

df_audit = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')
p403 = df_audit[df_audit['pair_id'] == 403].iloc[0]

claim_403 = p403['claim_text'] # "Music that glorifies violence against women should be banned"
gen_403 = p403['generated_viewpoint_text']
gold_403 = p403['gold_perspective_text']

# Test Claim-Grounded Stance Verification
# Hypothesis A: Claim is true
# Hypothesis B: Claim is false (negated claim)
neg_claim_403 = "Music that glorifies violence against women should NOT be banned"

p_c_contra, p_c_entail, p_c_neutral = run_nli_pair(gen_403, claim_403)
p_nc_contra, p_nc_entail, p_nc_neutral = run_nli_pair(gen_403, neg_claim_403)
p_gold_contra, p_gold_entail, p_gold_neutral = run_nli_pair(gen_403, gold_403)
p_rev_contra, p_rev_entail, p_rev_neutral = run_nli_pair(gold_403, gen_403)

print(f"Claim: \"{claim_403}\"")
print(f"Gen Viewpoint: \"{gen_403[:120]}...\"")
print(f"\n1. Direct Perspective Entailment (Gen -> Gold):")
print(f"   Contra: {p_gold_contra:.4f} | Entail: {p_gold_entail:.4f} | Neutral: {p_gold_neutral:.4f}")
print(f"2. Reverse Perspective Entailment (Gold -> Gen):")
print(f"   Contra: {p_rev_contra:.4f} | Entail: {p_rev_entail:.4f} | Neutral: {p_rev_neutral:.4f}")
print(f"3. Claim-Grounded Stance: Gen -> Claim (\"Banned\"):")
print(f"   Contra: {p_c_contra:.4f} | Entail: {p_c_entail:.4f} | Neutral: {p_c_neutral:.4f}")
print(f"4. Claim-Grounded Stance: Gen -> Negated Claim (\"NOT Banned\"):")
print(f"   Contra: {p_nc_contra:.4f} | Entail: {p_nc_entail:.4f} | Neutral: {p_nc_neutral:.4f}")

# Check all UNKNOWN stance pairs in the audit sample (N=8)
unknown_pairs = df_audit[df_audit['stance_generated'].astype(str).str.lower() == 'unknown']
print(f"\nAudit of All {len(unknown_pairs)} Pairs with UNKNOWN Generated Stance in Audit Sample:")
for idx, r in unknown_pairs.iterrows():
    c_contra, c_entail, c_neutral = run_nli_pair(r['generated_viewpoint_text'], r['claim_text'])
    nc_contra, nc_entail, nc_neutral = run_nli_pair(r['generated_viewpoint_text'], f"It is false that {r['claim_text']}")
    # Stance classification rule: if c_contra > 0.50 or nc_entail > 0.50, flag as undermine
    print(f"  PID {r['pair_id']:4d} | Claim: \"{r['claim_text'][:30]}...\" | Human: {r['human_audit_decision']:8s} | Dec: {r['final_decision']:30s} | Gen->Claim Contra: {c_contra:.4f}, Entail: {c_entail:.4f}")

# ==============================================================================
# PRIORITY 2: PREFIX STRIPPING & NORMALIZATION ON 4 FALSE NEGATIVES
# ==============================================================================
print("\n" + "=" * 90)
print("PRIORITY 2: PREFIX STRIPPING ON 4 CLASS 2 FALSE NEGATIVES")
print("=" * 90)

fn_pids = [734, 344, 44, 1273]
fn_rows = df_audit[df_audit['pair_id'].isin(fn_pids)].copy()

prefix_patterns = [
    r'^Practical Perspective:\s*Health Implications:\s*',
    r'^From a Legal Perspective::\s*',
    r'^Post-Invasion Governance:\s*',
    r'^[A-Za-z\s]+Perspective:\s*',
    r'^[A-Za-z\s]+Perspective::\s*'
]

for idx, r in fn_rows.iterrows():
    pid = r['pair_id']
    raw_gen = r['generated_viewpoint_text']
    clean_gen = raw_gen
    for pat in prefix_patterns:
        clean_gen = re.sub(pat, '', clean_gen, flags=re.IGNORECASE).strip()
        
    gold = r['gold_perspective_text']
    
    # Old scores
    old_contra, old_ent, old_neu = run_nli_pair(raw_gen, gold)
    old_probe = calc_probe_prob(r['cosine_similarity'], old_ent, old_neu)
    
    # New scores with clean text
    new_contra, new_ent, new_neu = run_nli_pair(clean_gen, gold)
    new_probe = calc_probe_prob(r['cosine_similarity'], new_ent, new_neu)
    
    print(f"\nPID {pid} (Claim {r['claim_id']}):")
    print(f"  Raw Gen:   \"{raw_gen[:80]}...\"")
    print(f"  Clean Gen: \"{clean_gen[:80]}...\"")
    print(f"  Before Stripping -> Entail: {old_ent:.4f} | Neutral: {old_neu:.4f} | Probe: {old_probe:.4f} (Threshold T*={t_probe})")
    print(f"  After Stripping  -> Entail: {new_ent:.4f} | Neutral: {new_neu:.4f} | Probe: {new_probe:.4f} (Delta: {new_probe - old_probe:+.4f})")

# ==============================================================================
# PRIORITY 3: SECOND-STAGE DISAGREEMENT CHECK ON 3 FALSE POSITIVES
# ==============================================================================
print("\n" + "=" * 90)
print("PRIORITY 3: SECOND-STAGE DISAGREEMENT CHECK ON FALSE POSITIVES")
print("=" * 90)

fp_pids = [1442, 403, 180]
fp_rows = df_audit[df_audit['pair_id'].isin(fp_pids)].copy()

for idx, r in fp_rows.iterrows():
    pid = r['pair_id']
    gen = r['generated_viewpoint_text']
    gold = r['gold_perspective_text']
    
    fwd_c, fwd_e, fwd_n = run_nli_pair(gen, gold)      # Gen -> Gold
    bwd_c, bwd_e, bwd_n = run_nli_pair(gold, gen)      # Gold -> Gen
    
    # Asymmetry metrics
    entail_diff = abs(fwd_e - bwd_e)
    bidirectional_min = min(fwd_e, bwd_e)
    bidirectional_harmonic = 2 * fwd_e * bwd_e / max(fwd_e + bwd_e, 1e-8)
    
    print(f"\nPID {pid} ({r['human_audit_decision']}):")
    print(f"  Gold: \"{gold[:75]}...\"")
    print(f"  Gen:  \"{gen[:75]}...\"")
    print(f"  Forward (Gen -> Gold): Contra: {fwd_c:.4f} | Entail: {fwd_e:.4f} | Neutral: {fwd_n:.4f}")
    print(f"  Backward(Gold -> Gen): Contra: {bwd_c:.4f} | Entail: {bwd_e:.4f} | Neutral: {bwd_n:.4f}")
    print(f"  Bidirectional Min Entailment: {bidirectional_min:.4f} | Harmonic: {bidirectional_harmonic:.4f} | Asymmetry: {entail_diff:.4f}")

# Also compare against True Positives from Stratum 1
print("\nComparison with Sample True Positives (Class 2):")
for pid in [1524, 808, 977]:
    r = df_audit[df_audit['pair_id'] == pid].iloc[0]
    fwd_c, fwd_e, fwd_n = run_nli_pair(r['generated_viewpoint_text'], r['gold_perspective_text'])
    bwd_c, bwd_e, bwd_n = run_nli_pair(r['gold_perspective_text'], r['generated_viewpoint_text'])
    b_min = min(fwd_e, bwd_e)
    print(f"  TP PID {pid:4d} | Fwd Entail: {fwd_e:.4f} | Bwd Entail: {bwd_e:.4f} | Bidirectional Min: {b_min:.4f}")
