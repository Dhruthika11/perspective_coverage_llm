import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

def clean_text(text):
    text = re.sub(r'^(?:####\s*)?(?:[\w\s\-]+Perspective|Viewpoint|Justification|Argument)\s*::?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\*\*(?:Stance|Perspective|Summary|Reasoning)\*\*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_toulmin_components(text, claim_topic=""):
    cleaned = clean_text(text)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]
    if not sentences:
        return cleaned, cleaned, cleaned
    
    # 1. Central Claim:
    lead = sentences[0]
    # Check for causal split
    c_split = re.search(r'\b(because|since|as|by\s+[a-z]+ing|in order to|so that|which leads to|resulting in)\b', lead, re.IGNORECASE)
    if c_split and c_split.start() > 15:
        claim_text = lead[:c_split.start()].strip().rstrip(',;')
        intra_reason = lead[c_split.start():].strip()
    else:
        claim_text = lead
        intra_reason = ""
        
    # 2. Core Reasoning:
    if len(sentences) > 1:
        tail = " ".join(sentences[1:])
        reasoning_text = f"{intra_reason} {tail}".strip() if intra_reason else tail
    else:
        reasoning_text = intra_reason if intra_reason else lead

    # 3. Underlying Warrant:
    # Toulmin warrant: The general normative rule or value bridge that justifies the claim based on the reasoning.
    # Extract the evaluative nexus: the core value judgment or functional bridge
    eval_match = re.search(r'\b(lead to|leads to|results in|causes|promotes|fosters|enhances|undermines|prevents|violates|ensures|reduces|increases|is harmful|is beneficial|is crucial|is essential|is necessary|infringes|protects)\b\s+(.*)', cleaned, re.IGNORECASE)
    if eval_match:
        warrant_phrase = eval_match.group(0).rstrip('.')
        warrant_text = f"The underlying principle that {warrant_phrase} justifies the position on {claim_topic}."
    else:
        warrant_text = f"The principle that {reasoning_text} justifies {claim_text} regarding {claim_topic}."
        
    return claim_text, reasoning_text, warrant_text

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
sample_c2 = df[df['human_ground_truth_label'] == 'class_2'].head(5)
sample_c1 = df[df['human_ground_truth_label'] == 'class_1'].head(5)

print("Computing cosine similarities on extracted components for sample pairs:")
for label, sample_df in [('CLASS 2 (Matches)', sample_c2), ('CLASS 1 (Hard Negatives)', sample_c1)]:
    print("\n" + "=" * 90)
    print(f"--- {label} ---")
    print("=" * 90)
    for idx, r in sample_df.iterrows():
        c_gold, r_gold, w_gold = extract_toulmin_components(r['gold_perspective_text'], r['claim_text'])
        c_gen, r_gen, w_gen = extract_toulmin_components(r['generated_viewpoint_text'], r['claim_text'])
        
        orig_cos = float(r['cosine_similarity'])
        
        # Embed components
        embs = embedder.encode([c_gold, c_gen, r_gold, r_gen, w_gold, w_gen], normalize_embeddings=True)
        cos_c = float(np.dot(embs[0], embs[1]))
        cos_r = float(np.dot(embs[2], embs[3]))
        cos_w = float(np.dot(embs[4], embs[5]))
        
        print(f"PID {r['pair_id']:<4d} | Orig={orig_cos:.3f} | Claim_Sim={cos_c:.3f} | Reason_Sim={cos_r:.3f} | Warrant_Sim={cos_w:.3f}")
        print(f"  Gold: {r['gold_perspective_text'][:70]}...")
        print(f"  Gen : {r['generated_viewpoint_text'][:70]}...")
