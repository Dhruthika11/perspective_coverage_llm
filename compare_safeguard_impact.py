import pandas as pd

df = pd.read_csv('data/audit/unified_production_audit_benchmark_100.csv')

# Baseline: No Safeguard
df['pred_no_safe'] = df['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT'
c2 = df['human_audit_decision'] == 'class_2'

tp_base = (df['pred_no_safe'] & c2).sum()
fp_base = (df['pred_no_safe'] & ~c2).sum()
fn_base = (~df['pred_no_safe'] & c2).sum()
tn_base = (~df['pred_no_safe'] & ~c2).sum()

p_base = tp_base / (tp_base + fp_base)
r_base = tp_base / (tp_base + fn_base)
f1_base = 2 * p_base * r_base / (p_base + r_base)

# With Safeguard
# Fails safeguard if stance == 'unknown' and not (p_entail >= 0.10 or p_neutral <= 0.90)
is_unk = df['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
fails_safe = is_unk & ~((df['p_entail'] >= 0.10) | (df['p_neutral'] <= 0.90))

df['pred_with_safe'] = df['pred_no_safe'] & (~fails_safe)

tp_safe = (df['pred_with_safe'] & c2).sum()
fp_safe = (df['pred_with_safe'] & ~c2).sum()
fn_safe = (~df['pred_with_safe'] & c2).sum()
tn_safe = (~df['pred_with_safe'] & ~c2).sum()

p_safe = tp_safe / (tp_safe + fp_safe)
r_safe = tp_safe / (tp_safe + fn_safe)
f1_safe = 2 * p_safe * r_safe / (p_safe + r_safe)

print("COMPARISON ON 100 AUDITED PRODUCTION PAIRS:")
print(f"{'Metric':<25} | {'WITHOUT Safeguard (Clean)':<28} | {'WITH Safeguard':<20} | {'Delta'}")
print("-" * 85)
print(f"{'True Positives (TP)':<25} | {tp_base:<28} | {tp_safe:<20} | {tp_safe - tp_base:+d}")
print(f"{'False Positives (FP)':<25} | {fp_base:<28} | {fp_safe:<20} | {fp_safe - fp_base:+d}")
print(f"{'False Negatives (FN)':<25} | {fn_base:<28} | {fn_safe:<20} | {fn_safe - fn_base:+d}")
print(f"{'True Negatives (TN)':<25} | {tn_base:<28} | {tn_safe:<20} | {tn_safe - tn_base:+d}")
print(f"{'Class 2 Precision':<25} | {p_base*100:6.2f}% ({tp_base}/{tp_base+fp_base}){'':<12} | {p_safe*100:6.2f}% ({tp_safe}/{tp_safe+fp_safe}){'':<4} | {(p_safe - p_base)*100:+5.2f}%")
print(f"{'Class 2 Recall':<25} | {r_base*100:6.2f}% ({tp_base}/{tp_base+fn_base}){'':<12} | {r_safe*100:6.2f}% ({tp_safe}/{tp_safe+fn_safe}){'':<4} | {(r_safe - r_base)*100:+5.2f}%")
print(f"{'Class 2 F1-Score':<25} | {f1_base:6.4f}{'':<22} | {f1_safe:6.4f}{'':<14} | {f1_safe - f1_base:+6.4f}")
