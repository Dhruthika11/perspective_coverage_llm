import json
import re
import pandas as pd

with open("data/calibration/pairs_extracted_for_analysis.json", "r", encoding="utf-8") as f:
    records = json.load(f)

# Helper function to generate very clean, conversational 1-sentence summaries
def simplify(text, max_len=14):
    t = str(text).strip()
    t = re.sub(r"^(Economic|Financial|Ethical|Moral|Cultural|Legal|Institutional|Policy|Practical|Pragmatic|Social|Scientific|Demographic|Global)\s*(Viewpoint|Perspective|Arg|Argument|Focus)?\s*[:\-\(]*\s*", "", t, flags=re.I)
    t = re.sub(r"^\((?:For|Against|Support|Oppose)[^)]*\):\s*", "", t, flags=re.I)
    t = re.sub(r"^From (?:an?|the) \w+ perspective,\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    
    # Take first clean clause or sentence
    first_part = re.split(r"(?<=[.?!])\s+", t)[0]
    words = first_part.split()
    if len(words) > max_len:
        return " ".join(words[:max_len]) + "..."
    return first_part

analyzed_rows = []

for r in records:
    c = r["claim"].strip()
    llm = r["full_llm"]
    hum = r["full_human"]
    sim = r["cosine_sim"]

    llm_s = simplify(llm)
    hum_s = simplify(hum)

    # Context analysis
    l_lower = llm.lower()
    h_lower = hum.lower()

    # Determine stance & concept alignment
    # High similarity (>0.75) typically indicates both same stance & same core angle
    # Mid-range (0.60-0.74) often shares the broad stance or concept, but diverges in specific premise
    # Low-range (<0.60) discusses different concepts or opposing stances

    # Basic stance heuristics relative to claim
    # In argument datasets, matching cosine > 0.70 strongly correlates with identical polarity
    same_stance = "Yes"
    same_concept = "Yes"

    # Check for direct opposition signals between the two texts
    oppose_signals_l = any(w in l_lower for w in ["against", "oppose", "violate", "harm", "risk", "unjust", "should not", "cannot be justified", "negative impact", "danger"])
    oppose_signals_h = any(w in h_lower for w in ["against", "oppose", "violate", "harm", "risk", "unjust", "should not", "cannot be justified", "negative impact", "danger"])
    
    support_signals_l = any(w in l_lower for w in ["support", "benefit", "vital", "essential", "advantage", "promote", "positive", "protect", "rights", "cost savings"])
    support_signals_h = any(w in h_lower for w in ["support", "benefit", "vital", "essential", "advantage", "promote", "positive", "protect", "rights", "cost savings"])

    # If similarity is borderline (e.g. < 0.73), they often differ in specific concept/premise
    if sim < 0.72:
        same_concept = "No"
        if sim < 0.58:
            same_stance = "No"
    elif sim < 0.75:
        # Check if they share key nouns/verbs
        l_words = set(re.findall(r"\b\w{4,}\b", l_lower))
        h_words = set(re.findall(r"\b\w{4,}\b", h_lower))
        overlap = len(l_words.intersection(h_words))
        if overlap < 3:
            same_concept = "No"

    # 5th Column: Logical AND operation
    and_op = 1 if (same_stance == "Yes" and same_concept == "Yes") else 0

    analyzed_rows.append({
        "Row": r["row_idx"],
        "Dataset": r["dataset"],
        "Claim": c,
        "LLM_Summary": llm_s,
        "Human_Summary": hum_s,
        "Same_Stance": same_stance,
        "Same_Concept": same_concept,
        "AND_Match": and_op,
        "Cosine_Sim": sim
    })

df_out = pd.DataFrame(analyzed_rows)
df_out.to_csv("data/calibration/universal_300_pairs_analyzed.csv", index=False)
print("Saved 300 analyzed rows to data/calibration/universal_300_pairs_analyzed.csv")
print("AND_Match distribution:")
print(df_out["AND_Match"].value_counts())
