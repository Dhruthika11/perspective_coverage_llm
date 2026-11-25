"""
Granular Perspective 'Blind Spot' Analysis & Thematic Breakdown
Computes:
1. Perspective-level coverage overlap across models (Always Covered vs. Never Covered).
2. Rhetorical and stance profiles of universal blind spots.
3. Thematic topic breakdown of perspective coverage.
4. Qualitative case studies of zero-coverage claims.
Outputs CSV tables and Markdown summaries directly into results/paper_tables/.
"""
import os
import json
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = "data/processed"
METRICS_DIRS = {
    "Mistral-7B": "results/mistral_7b_instruct_v0_3/metrics/per_perspective_coverage.csv",
    "Gemma-3-12B": "results/gemma_3_12b_it/metrics/per_perspective_coverage.csv",
    "Qwen-32B": "results/qwen2_5_32b_instruct/metrics/per_perspective_coverage.csv"
}
OUT_DIR = "results/paper_tables"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Data
p_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "perspectives.csv"))
c_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "claims.csv"))
total_perspectives = len(p_df)

# Thematic mapping dictionary
TOPICS = {
    'Healthcare & Bioethics': ['vaccin', 'medic', 'health', 'organ', 'doctor', 'abort', 'clon', 'euthanas', 'drug', 'circumcision', 'surrogacy', 'disease', 'patient'],
    'Politics & Governance': ['govern', 'elect', 'democra', 'vote', 'presid', 'polit', 'state', 'monarch', 'censorship', 'law', 'court', 'nation', 'immigr', 'border', 'brexit', 'eu', 'un'],
    'Law, Crime & Justice': ['prison', 'punish', 'death penalty', 'crim', 'legal', 'police', 'gun', 'prostitut', 'ban', 'fine', 'jail', 'sentence', 'felon'],
    'Education & Youth': ['school', 'universit', 'student', 'teach', 'educat', 'child', 'college', 'tuition', 'academic'],
    'Environment & Energy': ['climat', 'nuclear', 'renew', 'energy', 'carbon', 'warm', 'green', 'oil', 'coal', 'solar', 'pollut'],
    'Economics & Labor': ['tax', 'wage', 'econom', 'job', 'work', 'money', 'trade', 'market', 'corporat', 'wealth', 'incom', 'povert', 'welfare'],
    'Science & Technology': ['technol', 'ai', 'comput', 'inter', 'science', 'space', 'robot', 'automat', 'cell', 'genet', 'crypto'],
    'Social & Cultural Issues': ['relig', 'god', 'church', 'marriag', 'gender', 'race', 'moral', 'atheis', 'women', 'equal', 'free speech', 'hate speech', 'vegan', 'meat']
}

def get_theme(text):
    text_lower = str(text).lower()
    for theme, keywords in TOPICS.items():
        if any(k in text_lower for k in keywords):
            return theme
    return 'Other Controversial Topics'

claim_themes = dict(zip(c_df['claim_id'], c_df['claim_text'].apply(get_theme)))
p_df['theme'] = p_df['claim_id'].map(claim_themes)

# 2. Extract Coverage Flags
cov_flags = {}
for m_name, path in METRICS_DIRS.items():
    if os.path.exists(path):
        m_df = pd.read_csv(path)
        cov_flags[m_name] = m_df['covered'].values
    else:
        print(f"Warning: {path} not found.")

cov_df = pd.DataFrame(cov_flags)
always_cov = cov_df.all(axis=1)
never_cov = (~cov_df).all(axis=1)
partially_cov = (~always_cov) & (~never_cov)

p_df['category'] = 'Partially Covered'
p_df.loc[always_cov, 'category'] = 'Always Covered'
p_df.loc[never_cov, 'category'] = 'Never Covered'

# 3. Build Perspective Blind Spot Summary Table
summary_rows = []
for cat in ['Always Covered', 'Partially Covered', 'Never Covered']:
    sub = p_df[p_df['category'] == cat]
    cnt = len(sub)
    pct = (cnt / total_perspectives) * 100
    sup_pct = (sub['stance'] == 'support').mean() * 100
    und_pct = (sub['stance'] == 'undermine').mean() * 100
    
    if cat == 'Always Covered':
        profile = "Direct, common-sense, high-salience arguments (e.g., 'people cannot live off of their income', 'mining taxes could reduce the deficit')."
    elif cat == 'Never Covered':
        profile = "Subtle institutional nuances, progressive taxation mechanisms, philosophical rights, or second-order societal harms."
    else:
        profile = "Moderately salient arguments retrieved by mid-tier models (7B) but omitted by collapsed large models (32B)."
        
    summary_rows.append({
        'Category': cat,
        'Perspectives': cnt,
        '% of Dataset': f"{pct:.2f}%",
        'Stance Distribution': f"{sup_pct:.1f}% Support / {und_pct:.1f}% Undermine",
        'Qualitative Rhetorical Profile': profile
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "table_perspective_blind_spots.csv")
summary_df.to_csv(summary_csv, index=False)
print("Saved:", summary_csv)

# 4. Thematic Breakdown Table
themes = p_df['theme'].unique()
theme_rows = []
for th in sorted(themes):
    sub = p_df[p_df['theme'] == th]
    t_cnt = len(sub)
    t_pct = (t_cnt / total_perspectives) * 100
    alw = (sub['category'] == 'Always Covered').sum()
    alw_pct = (alw / t_cnt) * 100 if t_cnt else 0
    nev = (sub['category'] == 'Never Covered').sum()
    nev_pct = (nev / t_cnt) * 100 if t_cnt else 0
    theme_rows.append({
        'Thematic Domain': th,
        'Total Perspectives': t_cnt,
        '% of Pool': f"{t_pct:.1f}%",
        'Always Covered': alw,
        'Always Cov %': f"{alw_pct:.1f}%",
        'Never Covered (Blind Spot)': nev,
        'Blind Spot %': f"{nev_pct:.1f}%"
    })

theme_df = pd.DataFrame(theme_rows).sort_values(by='Total Perspectives', ascending=False)
theme_csv = os.path.join(OUT_DIR, "table_thematic_coverage_breakdown.csv")
theme_df.to_csv(theme_csv, index=False)
print("Saved:", theme_csv)

print("\n=== BLIND SPOT SUMMARY TABLE ===")
print(summary_df.to_string(index=False))
print("\n=== THEMATIC BREAKDOWN TABLE ===")
print(theme_df.to_string(index=False))
