import sys, os
sys.path.insert(0, '/home/srujan.sai/perspective_coverage')
import pandas as pd
from src.extraction.extraction_pipeline import ExtractionPipeline
from src.extraction.schemas import QuarantineReason

df_raw = pd.read_csv('/home/srujan.sai/perspective_coverage/data/llm_outputs/raw_generations/olmo_2_1124_7b/raw_generations.csv')
t = df_raw[(df_raw['claim_id']==477) & (df_raw['sample_id']==4)]['generated_text'].iloc[0]
p = ExtractionPipeline()
res = p.process_raw_generation('olmo_2_1124_7b', 477, 4, t)
print('res.viewpoints:', len(res.viewpoints))
print('res.quarantined:', res.quarantined)
print('res.parsing_outcome:', res.parsing_outcome)
print('res.notes:', res.notes)

# Also let's inspect what candidate_segments was
cand = p.segmenter.segment_text(t)
print('segmenter output count:', len(cand))
for c in cand:
    print('  cand text:', repr(c.text))
    is_valid, q_reason, expl = p.quarantine.validate_viewpoint_text(c.text)
    print('  is_valid:', is_valid, q_reason, expl)
