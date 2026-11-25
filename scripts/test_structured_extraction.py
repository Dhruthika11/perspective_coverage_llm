import re
import pandas as pd

# Let's inspect 15 diverse pairs from train, calibration, and test
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

def clean_prefix(text):
    text = re.sub(r'^(?:####\s*)?(?:[\w\s\-]+Perspective|Viewpoint|Justification|Argument)\s*::?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\*\*(?:Stance|Perspective|Summary|Reasoning)\*\*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_argument_components(text, claim_topic=""):
    """
    Decomposes an argumentative viewpoint into:
      1. Central Claim (the core assertion/stance)
      2. Core Reasoning (the causal mechanism, empirical grounds, or explanation)
      3. Underlying Warrant (the normative principle or inferential bridge)
    """
    clean = clean_prefix(text)
    
    # Sentence splitting
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if s.strip()]
    if not sentences:
        return clean, clean, clean
    
    # 1. Central Claim Extraction
    # In argumentative text, the first sentence (or main clause) is the claim thesis.
    lead_sentence = sentences[0]
    
    # If the lead sentence contains internal connective markers like "because", "by", "as", "since",
    # split into claim clause and reasoning clause
    connective_match = re.search(r'\b(because|since|as|by\s+[a-z]+ing|in order to|so that|which leads to|resulting in)\b', lead_sentence, re.IGNORECASE)
    
    if connective_match and connective_match.start() > 15:
        claim_part = lead_sentence[:connective_match.start()].strip().rstrip(',;')
        intra_reasoning = lead_sentence[connective_match.start():].strip()
    else:
        claim_part = lead_sentence
        intra_reasoning = ""
        
    central_claim = claim_part
    
    # 2. Core Reasoning Extraction
    # If there are subsequent sentences (sentences[1:]), they contain the grounds/evidence/mechanisms.
    if len(sentences) > 1:
        elaboration = " ".join(sentences[1:])
        if intra_reasoning:
            core_reasoning = intra_reasoning + " " + elaboration
        else:
            core_reasoning = elaboration
    else:
        core_reasoning = intra_reasoning if intra_reasoning else lead_sentence

    # 3. Underlying Warrant Extraction
    # The warrant is the inferential principle linking the reason/mechanism to the claim topic.
    # It captures the evaluative criterion: why does this reason matter for the claim?
    # We formulate the warrant representation by isolating the evaluative criterion and inferential bridge:
    # "Reasoning justifies stance on claim: [Core evaluative principle]"
    eval_match = re.search(r'\b(can lead to|leads to|results in|causes|promotes|fosters|enhances|undermines|prevents|violates|ensures|reduces|increases|is essential|is harmful|is beneficial|is crucial|is necessary|infringes|protects)\b\s+(.*)', clean, re.IGNORECASE)
    if eval_match:
        eval_phrase = eval_match.group(0).rstrip('.')
        warrant_repr = f"Principle: {eval_phrase}"
    else:
        warrant_repr = f"Principle that {core_reasoning} warrants {central_claim}"
        
    return central_claim, core_reasoning, warrant_repr

print("Testing Structured Argument Component Extraction on Sample Benchmark Pairs:")
print("=" * 80)
for idx, r in df.sample(6, random_state=123).iterrows():
    print(f"\n[Pair {r['pair_id']} - {r['human_ground_truth_label'].upper()}] Claim Topic: {r['claim_text']}")
    print(f"Original Gen: {r['generated_viewpoint_text']}")
    c_gen, r_gen, w_gen = extract_argument_components(r['generated_viewpoint_text'], r['claim_text'])
    print(f"  -> Extracted Claim:     {c_gen}")
    print(f"  -> Extracted Reasoning: {r_gen}")
    print(f"  -> Extracted Warrant:   {w_gen}")
    
    print(f"Original Gold: {r['gold_perspective_text']}")
    c_gold, r_gold, w_gold = extract_argument_components(r['gold_perspective_text'], r['claim_text'])
    print(f"  -> Extracted Claim:     {c_gold}")
    print(f"  -> Extracted Reasoning: {r_gold}")
    print(f"  -> Extracted Warrant:   {w_gold}")
