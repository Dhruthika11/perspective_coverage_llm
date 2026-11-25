import pandas as pd

pre_smol = pd.read_csv('data/pilot_validation_prefix/smollm2_135m_instruct_pilot_viewpoints.csv')
post_smol = pd.read_csv('data/pilot_validation/smollm2_135m_instruct_pilot_viewpoints.csv')

pre_counts = pre_smol.groupby(['claim_id', 'sample_id']).size()
post_counts = post_smol.groupby(['claim_id', 'sample_id']).size()

all_keys = set(pre_counts.index).union(set(post_counts.index))
diffs = []
for k in sorted(all_keys):
    cp = pre_counts.get(k, 0)
    cpost = post_counts.get(k, 0)
    if cp != cpost:
        diffs.append((k[0], k[1], cp, cpost, cpost - cp))

df_diff = pd.DataFrame(diffs, columns=['claim_id', 'sample_id', 'pre_count', 'post_count', 'diff'])
print(f"Total diff samples in SmolLM2: {len(df_diff)}")
print(df_diff)
