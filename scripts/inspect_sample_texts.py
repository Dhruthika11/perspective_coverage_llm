import pandas as pd

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
print("Sample generated viewpoints:")
for i, row in df.sample(8, random_state=42).iterrows():
    print(f"--- Pair {row['pair_id']} (Claim {row['claim_id']}: {row['claim_text']}) ---")
    print(f"Gold: {row['gold_perspective_text']}")
    print(f"Gen : {row['generated_viewpoint_text']}")
    print()
