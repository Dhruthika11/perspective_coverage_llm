import json
import re
import pandas as pd

with open("data/calibration/e3_extracted_components_cache.json") as f:
    cache = json.load(f)

records = list(cache.values())
df_ext = pd.DataFrame(records)

print(f"Total extracted records audited: {len(df_ext)}")
print("Sources:", df_ext['metadata'].apply(lambda x: x['source']).value_counts().to_dict())

# 1. Identity rate: claim == reasoning
df_ext['source'] = df_ext['metadata'].apply(lambda x: x['source'])
df_ext['claim_eq_reason'] = df_ext['central_claim'] == df_ext['core_reasoning']
df_ext['orig_word_count'] = df_ext['original_text'].str.split().str.len()
df_ext['claim_word_count'] = df_ext['central_claim'].str.split().str.len()
df_ext['reason_word_count'] = df_ext['core_reasoning'].str.split().str.len()

print("\n--- 1. IDENTITY RATE (Claim == Reasoning) ---")
for src in ['gold', 'generated']:
    sub = df_ext[df_ext['source'] == src]
    n_same = sub['claim_eq_reason'].sum()
    print(f"  {src.upper()}: {n_same} / {len(sub)} ({n_same/len(sub):.2%}) have claim == reasoning")

print("\n--- 2. LENGTH DISTRIBUTION (Word Counts) ---")
for src in ['gold', 'generated']:
    sub = df_ext[df_ext['source'] == src]
    print(f"  {src.upper()}:")
    print(f"    Original : mean = {sub['orig_word_count'].mean():.1f}, min = {sub['orig_word_count'].min()}, max = {sub['orig_word_count'].max()}")
    print(f"    Claim    : mean = {sub['claim_word_count'].mean():.1f}, min = {sub['claim_word_count'].min()}, max = {sub['claim_word_count'].max()}")
    print(f"    Reasoning: mean = {sub['reason_word_count'].mean():.1f}, min = {sub['reason_word_count'].min()}, max = {sub['reason_word_count'].max()}")

# 3. Check for anomalies / degenerate cases
print("\n--- 3. POTENTIAL ANOMALIES ---")
# Very short claims (< 3 words)
short_claims = df_ext[df_ext['claim_word_count'] < 3]
print(f"Claims with < 3 words: {len(short_claims)}")
for _, r in short_claims.iterrows():
    print(f"  [{r['source'].upper()}] Original: \"{r['original_text']}\" -> Claim: \"{r['central_claim']}\"")

# Check for trailing punctuation or connective fragments
dangling = df_ext[df_ext['central_claim'].str.endswith((',', ';', ':', 'because', 'as', 'since', 'by'))]
print(f"\nClaims ending in connective/punctuation: {len(dangling)}")

# Check cases where 'as' caused a split
as_splits = df_ext[df_ext['core_reasoning'].str.startswith(('as ', 'as\t'))]
print(f"\nReasoning starting with 'as': {len(as_splits)}")
for _, r in as_splits.head(5).iterrows():
    print(f"  Orig: \"{r['original_text'][:70]}...\"")
    print(f"    -> Claim:  \"{r['central_claim']}\"")
    print(f"    -> Reason: \"{r['core_reasoning'][:70]}...\"")
