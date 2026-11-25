import pandas as pd

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

print("--- GOLD TEXTS (10 samples) ---")
for t in df['gold_perspective_text'].drop_duplicates().sample(10, random_state=42):
    print("GOLD:", repr(t))

print("\n--- GENERATED TEXTS (10 samples) ---")
for t in df['generated_viewpoint_text'].drop_duplicates().sample(10, random_state=42):
    print("GEN :", repr(t))
