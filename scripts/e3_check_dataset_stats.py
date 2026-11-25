import pandas as pd

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
print(f"Total pairs: {len(df)}")
n_gold = df['gold_perspective_text'].nunique()
n_gen = df['generated_viewpoint_text'].nunique()
print(f"Unique gold perspectives: {n_gold}")
print(f"Unique generated viewpoints: {n_gen}")
print(f"Unique claims: {df['claim_id'].nunique()}")

all_texts = set(df['gold_perspective_text'].dropna()).union(set(df['generated_viewpoint_text'].dropna()))
print(f"Total unique texts to extract: {len(all_texts)}")

gold_lens = df['gold_perspective_text'].dropna().str.split().str.len()
gen_lens = df['generated_viewpoint_text'].dropna().str.split().str.len()
print(f"Gold text word count: mean={gold_lens.mean():.1f}, min={gold_lens.min()}, max={gold_lens.max()}")
print(f"Gen text word count: mean={gen_lens.mean():.1f}, min={gen_lens.min()}, max={gen_lens.max()}")
