import pandas as pd

raw_qwen = pd.read_csv('data/llm_outputs/raw_generations/qwen2_5_32b_instruct/raw_generations.csv')
raw_text = raw_qwen[(raw_qwen['claim_id']==4) & (raw_qwen['sample_id']==0)]['generated_text'].values[0]

prefix_vp = pd.read_csv('data/pilot_validation_prefix/qwen2_5_32b_instruct_pilot_viewpoints.csv')
pre_vps = prefix_vp[(prefix_vp['claim_id']==4) & (prefix_vp['sample_id']==0)]

postfix_vp = pd.read_csv('data/pilot_validation/qwen2_5_32b_instruct_pilot_viewpoints.csv')
post_vps = postfix_vp[(postfix_vp['claim_id']==4) & (postfix_vp['sample_id']==0)]

print("=== RAW TEXT ===")
print(raw_text)
print("\n=== PRE-FIX VIEWPOINTS (" + str(len(pre_vps)) + ") ===")
for idx, r in pre_vps.iterrows():
    print(str(r['viewpoint_id']) + ". [" + str(r['extraction_method']) + "] " + str(r['viewpoint_text']))
print("\n=== POST-FIX VIEWPOINTS (" + str(len(post_vps)) + ") ===")
for idx, r in post_vps.iterrows():
    print(str(r['viewpoint_id']) + ". [" + str(r['extraction_method']) + "] " + str(r['viewpoint_text']))
