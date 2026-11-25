import re

KNOWN_ABBREVIATIONS = ["D.A.R.E.", "U.S.", "U.K.", "E.U.", "A.P.A.", "e.g.", "i.e.", "et al.", "vs.", "etc."]

def protect_abbreviations(text: str) -> str:
    res = text
    # Pattern for single letter dotted acronyms like D.A.R.E. or U.S.
    res = re.sub(r'\b([A-Za-z]\.){2,}', lambda m: m.group(0).replace('.', '__DOT__'), res)
    # Common lower-case abbreviations
    res = re.sub(r'\b(e\.g\.|i\.e\.|et al\.|vs\.)', lambda m: m.group(0).replace('.', '__DOT__'), res, flags=re.IGNORECASE)
    return res

def unprotect_abbreviations(text: str) -> str:
    return text.replace('__DOT__', '.')

def split_sentences_v1_1(text: str):
    cleaned = text.strip()
    protected = protect_abbreviations(cleaned)
    sents = [unprotect_abbreviations(s.strip()) for s in re.split(r'(?<=[.!?])\s+', protected) if s.strip()]
    return sents if sents else [cleaned]

# Refined causal connective pattern (excluding prepositional 'as')
REFINED_CONNECTIVE_PATTERN = re.compile(
    r'\b('
    r'because|since|on the grounds that|in order to|so that|which leads to|resulting in|'
    r'by\s+[a-z]+ing|through\s+[a-z]+ing|'
    r'as\s+(?:it|they|this|these|there|we|he|she|[a-z]+ing|[a-z\s]+?\s+(?:is|are|was|were|can|could|will|would|have|has|had|do|does|did|tend|tends|lead|leads|demonstrate|demonstrates|show|shows))\b'
    r')',
    re.IGNORECASE
)

test_cases = [
    "The D.A.R.E. program's ethical approach to educating children about drug use may inadvertently stigmatize youth.",
    "International law increasingly recognizes self-determination as a fundamental principle for achieving peace.",
    "Violent video games do not correlate with violent crime rates, as economic factors have a stronger impact.",
    "Rate of violence has been steadily dropping as violent games are being released.",
    "Implementing the ceiling creates employment by limiting weekly work hours."
]

print("Testing E3.1 Refined Sentence Splitting & Connective Extraction:")
for tc in test_cases:
    sents = split_sentences_v1_1(tc)
    lead = sents[0]
    m = REFINED_CONNECTIVE_PATTERN.search(lead)
    if m and m.start() > 15:
        claim = lead[:m.start()].strip().rstrip(',;')
        reason = lead[m.start():].strip()
    else:
        claim = lead
        reason = " ".join(sents[1:]) if len(sents) > 1 else lead
    print(f"\nOriginal: {tc}")
    print(f"  -> Sents count: {len(sents)}")
    print(f"  -> Claim:  {claim}")
    print(f"  -> Reason: {reason}")
