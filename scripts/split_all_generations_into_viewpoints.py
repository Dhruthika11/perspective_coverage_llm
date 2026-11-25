"""
Production Viewpoint Separation Engine.

Takes raw/processed LLM generations across all model families (JSON, Multi-Viewpoint lists, Plain prose)
and explodes them so that EACH ROW in the resulting CSV contains EXACTLY ONE discrete viewpoint (~35-55 words).

Outputs:
  - data/viewpoint_separated/<model_key>/viewpoints.csv
  - data/viewpoint_separated/master_viewpoints_all_models.csv

Usage on HPC (Flair2):
    python scripts/split_all_generations_into_viewpoints.py
"""

import os
import re
import glob
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.expanduser("~/perspective_coverage/data/llm_outputs")
if not os.path.exists(BASE_DIR):
    BASE_DIR = "data/llm_outputs"

OUTPUT_DIR = os.path.expanduser("~/perspective_coverage/data/viewpoint_separated")
if not os.path.exists(os.path.dirname(OUTPUT_DIR)):
    OUTPUT_DIR = "data/viewpoint_separated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_invalid_viewpoint(text: str) -> tuple[bool, str]:
    """Detects if text is degenerate, prompt leakage, hallucinated exam task, or meta-commentary."""
    t = str(text).strip()
    if not t or t == "nan" or len(t.split()) < 6:
        return True, "too_short"
        
    t_lower = t.lower()
    
    # 1. Prompt regurgitation / instruction leakage
    prompt_leakage_patterns = [
        r"you are analyzing the controversial claim",
        r"respond with only a single json",
        r"matching this schema",
        r"1 concise sentence stating the core claim",
        r"1-2 sentences explaining the supporting logic",
        r"take one clear, atomic stance",
        r"strictly from an? [a-z, ]+ perspective",
        r"what do you think about the following claim",
        r"do not provide a balanced overview"
    ]
    for pat in prompt_leakage_patterns:
        if re.search(pat, t_lower):
            return True, "prompt_leakage"
            
    # 2. Hallucinated exam/task structures
    meta_task_patterns = [
        r"^\s*task:\s*(?:create a list|for each key term|provide|enter how many)",
        r"\bquestion:\s*what is the most authoritative source",
        r"\banswer:\s*\"[^\"]+\",\s*\"[^\"]+\"",
        r"^\s*task:\s+",
        r"\bkey terms that will allow you to quickly find\b"
    ]
    for pat in meta_task_patterns:
        if re.search(pat, t_lower):
            return True, "hallucinated_task"
            
    # 3. Non-argumentative meta-commentary about the generation task
    meta_commentary_patterns = [
        r"\ba list of (?:the most )?.*?arguments (?:about the claim )?would be\b",
        r"\bwould be a great contribution for the future\b",
        r"\bit would save people (?:a lot of )?time\b",
        r"\b(?:i am|as) an ai\b",
        r"^(?:here is|below is|this is) a (?:list|summary|breakdown)\b"
    ]
    for pat in meta_commentary_patterns:
        if re.search(pat, t_lower):
            return True, "meta_commentary"
            
    return False, "valid"

def extract_pseudo_json_fields(text: str) -> str:
    """Recovers summary and reasoning from unparsed pseudo-JSON like:
    Example: "perspective_summary": "...", "core_reasoning": "..."
    """
    t = str(text).strip()
    p_sum_match = re.search(r'["\']?(?:perspective_)?summary["\']?\s*:\s*["\']([^"\']{15,})["\']', t, re.I)
    c_reas_match = re.search(r'["\']?(?:core_)?reasoning["\']?\s*:\s*["\']([^"\']{15,})["\']', t, re.I)
    
    if p_sum_match:
        p_sum = p_sum_match.group(1).strip()
        c_reas = c_reas_match.group(1).strip() if c_reas_match else ""
        comb = f"{p_sum} {c_reas}".strip() if c_reas else p_sum
        return comb
    return None

def is_multiviewpoint_or_unparsed(text: str, row: pd.Series) -> bool:
    """Detects if text or row represents an unparsed dump or contains multi-viewpoint structures."""
    # Check postprocessor status flags
    parse_ok = row.get("parse_ok")
    if parse_ok is False or str(parse_ok).lower() == "false":
        return True
    if str(row.get("parse_method", "")).lower() == "unparsed":
        return True
        
    t = str(text).strip()
    if not t or t == "nan":
        return False
        
    # Check for bulleted lists, numbered lists, or section headers
    if re.search(r'(?:\n+\s*(?:###|(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?))|\A(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?))\s+', t, re.I):
        return True
    if re.search(r'\n+\s*(?:[-*•]|\d+[\.\)])\s+\*\*[^*]+\*\*', t):
        return True
    if "### " in t or "**Supportive Viewpoint**" in t or "**Critical Viewpoint**" in t:
        return True
    if len(t.split()) > 75 and ("\n1." in t or "\n2." in t or "\n-" in t or "\n*" in t):
        return True
        
    return False

def parse_single_viewpoint_json(row: pd.Series) -> list[dict]:
    """Extracts single viewpoint from genuine JSON fields or JSON raw string."""
    p_sum = str(row.get("perspective_summary", "")).strip()
    c_reas = str(row.get("core_reasoning", "")).strip()
    stance = str(row.get("llm_reported_stance", "")).strip()
    
    # 1. Check if postprocessor already parsed clean non-JSON text into summary
    if p_sum and p_sum != "nan" and not p_sum.startswith("{"):
        comb = f"{p_sum} {c_reas}".strip() if (c_reas and c_reas != "nan") else p_sum
        # If it is NOT multi-viewpoint and NOT unparsed dump, accept it as single atomic viewpoint
        if not is_multiviewpoint_or_unparsed(comb, row):
            return [{
                "text": comb,
                "stance": stance if stance != "nan" else "",
                "source_type": "parsed_json_fields"
            }]
        
    # 2. Try parsing raw text as JSON string
    text_col = "generated_text" if "generated_text" in row else ("argument" if "argument" in row else "combined_text")
    raw_text = str(row.get(text_col, "")).strip()
    if not raw_text or raw_text == "nan":
        raw_text = p_sum
    
    t_clean = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.I).strip()
    t_clean = re.sub(r'\s*```$', '', t_clean)
    start = t_clean.find('{')
    end = t_clean.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        candidate = t_clean[start:end+1]
        try:
            candidate_clean = re.sub(r',\s*\}', '}', candidate)
            obj = json.loads(candidate_clean)
            ps = obj.get("perspective_summary") or obj.get("summary") or obj.get("argument") or ""
            cr = obj.get("core_reasoning") or obj.get("reasoning") or ""
            st = obj.get("stance") or stance or ""
            comb = f"{ps} {cr}".strip() if cr else str(ps).strip()
            if comb and len(comb.split()) >= 6 and not is_multiviewpoint_or_unparsed(comb, pd.Series()):
                return [{
                    "text": comb,
                    "stance": str(st).lower().strip(),
                    "source_type": "raw_json_parse"
                }]
        except Exception:
            pass
            
    return []

def clean_viewpoint_text(s: str) -> str:
    """Thoroughly cleans markdown headers, bullet prefixes, and bold title keys."""
    s = str(s).strip()
    s = re.sub(r'^###\s+[^\n]+\n*', '', s).strip()
    # Strip leading bullet/numbers: '1. ', '2) ', '- ', '* ', without stripping '**'
    s = re.sub(r'^\s*(?:\d+[\.\)]|[-•*](?!\*))\s*', '', s).strip()
    # Strip leading bold title: '**Cost-Saving Perspective**:' or '**Title** -'
    s = re.sub(r'^\*{1,2}[^*]+\*{1,2}:?\s*[-–—:]?\s*', '', s).strip()
    # Strip secondary sub-bullet if present: '   - **Supportive Viewpoint**:'
    s = re.sub(r'^\s*(?:\d+[\.\)]|[-•*](?!\*))\s*', '', s).strip()
    s = re.sub(r'^\*{1,2}[^*]+\*{1,2}:?\s*[-–—:]?\s*', '', s).strip()
    # Strip leftover 'Supportive Viewpoint:' prefix
    s = re.sub(r'^(?:Supportive|Critical)\s+Viewpoint:?\s*', '', s, flags=re.I).strip()
    return s.strip()

def split_multiviewpoint_text(raw_text: str) -> list[dict]:
    """Splits multi-viewpoint markdown lists, numbered bullets, or section headers."""
    text = str(raw_text).strip()
    if not text or text == "nan":
        return []
        
    # Discard non-argumentative preamble preceding the first numbered bullet / header
    m_start = re.search(r'(?:\A|\n+)\s*(?:###\s+[^\n]+|(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?))\s+', text, flags=re.I)
    text_body = text[m_start.start():] if m_start else text
    
    # 1. Numbered items & headers: '### Header\n\n1. ', '1. ', '2. ', '1) ', 'Viewpoint 1:', 'Perspective 1:'
    pattern_num = r'(?:\n+\s*(?:###\s+[^\n]+(?:\n+\s*(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?))?|(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?))\s+|\A(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?)\s+)'
    splits = re.split(pattern_num, text_body, flags=re.I)
    
    cleaned_vps = []
    for s in splits:
        st = ""
        s_lower = s.lower()
        if "supportive viewpoint" in s_lower or "supportive" in s_lower[:35]:
            st = "support"
        elif "critical viewpoint" in s_lower or "undermine" in s_lower[:35]:
            st = "undermine"
            
        c = clean_viewpoint_text(s)
        if len(c.split()) >= 10:
            cleaned_vps.append({
                "text": c,
                "stance": st,
                "source_type": "numbered_bullet_split"
            })
            
    if len(cleaned_vps) >= 2 or (len(cleaned_vps) == 1 and m_start is not None):
        return cleaned_vps
        
    # 2. Markdown bold headers: '**Perspective 1:**', '**Economic Viewpoint:**'
    m_hdr = re.search(r'(?:\A|\n+)\s*\*\*(?:Viewpoint|Perspective)\s+1:?\*\*', text, flags=re.I)
    text_body_hdr = text[m_hdr.start():] if m_hdr else text
    pattern_hdr = r'\n+\s*\*\*[^*]+(?::\*\*|\*\*:?)\s*|\A\*\*[^*]+(?::\*\*|\*\*:?)\s*'
    splits_hdr = re.split(pattern_hdr, text_body_hdr)
    
    cleaned_vps_hdr = []
    for s in splits_hdr:
        c = clean_viewpoint_text(s)
        if len(c.split()) >= 10:
            cleaned_vps_hdr.append({
                "text": c,
                "stance": "",
                "source_type": "markdown_bold_split"
            })
    if len(cleaned_vps_hdr) >= 2 or (len(cleaned_vps_hdr) == 1 and m_hdr is not None):
        return cleaned_vps_hdr
        
    # 3. Double-newline paragraphs
    paras = [clean_viewpoint_text(p) for p in text.split('\n\n') if len(clean_viewpoint_text(p).split()) >= 15]
    if len(paras) >= 2:
        return [{
            "text": p,
            "stance": "",
            "source_type": "paragraph_split"
        } for p in paras]
        
    # Fallback: single unit (cleaned)
    fallback = clean_viewpoint_text(text)
    return [{
        "text": fallback if fallback else text,
        "stance": "",
        "source_type": "single_unit_fallback"
    }]

def process_all_models():
    print("=" * 115)
    print("PRODUCTION VIEWPOINT SEPARATION ENGINE: 1 ROW = 1 ATOMIC VIEWPOINT")
    print(f"Scanning models in: {BASE_DIR}")
    print(f"Saving to:          {OUTPUT_DIR}")
    print("=" * 115)
    
    subdirs = sorted([
        d for d in glob.glob(os.path.join(BASE_DIR, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in ["raw_generations", "ibm_debater", "kialo"]
    ])
    
    if not subdirs:
        print(f"No model folders found in {BASE_DIR}!")
        return
        
    all_master_records = []
    audit_summary = []
    
    for sdir in subdirs:
        model_key = os.path.basename(sdir)
        csv_path = os.path.join(sdir, "processed_generations.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(sdir, "raw_generations.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        text_col = "generated_text" if "generated_text" in df.columns else ("argument" if "argument" in df.columns else "combined_text")
        
        model_records = []
        raw_count = len(df)
        
        for idx, row in df.iterrows():
            cid = row.get("claim_id")
            sid = row.get("sample_id", idx)
            raw_text = str(row.get(text_col, "")).strip()
            if not raw_text or raw_text == "nan":
                raw_text = str(row.get("perspective_summary", "")).strip()
            if not raw_text or raw_text == "nan":
                raw_text = str(row.get("combined_text", "")).strip()
            
            # 1. Try JSON extraction first
            vps = parse_single_viewpoint_json(row)
            
            # 2. Try pseudo-JSON field extraction (for models that emit unbraced fields)
            if not vps:
                pseudo = extract_pseudo_json_fields(raw_text)
                if pseudo:
                    is_junk, _ = is_invalid_viewpoint(pseudo)
                    if not is_junk:
                        vps = [{"text": pseudo, "stance": "", "source_type": "pseudo_json_fields"}]
            
            # 3. If not JSON, split into multi-viewpoint units
            if not vps:
                vps = split_multiviewpoint_text(raw_text)
                
            clean_vp_idx = 0
            for item in vps:
                vp_text = item["text"]
                is_junk, reason = is_invalid_viewpoint(vp_text)
                if is_junk:
                    continue  # Filter out prompt regurgitations, hallucinated tasks, and meta-commentary
                    
                clean_vp_idx += 1
                w_count = len(vp_text.split())
                
                rec = {
                    "model_key": model_key,
                    "claim_id": cid,
                    "sample_id": sid,
                    "viewpoint_id": clean_vp_idx,
                    "viewpoint_text": vp_text,
                    "word_count": w_count,
                    "stance": item.get("stance", ""),
                    "source_type": item.get("source_type", "")
                }
                model_records.append(rec)
                all_master_records.append(rec)
                
        df_model = pd.DataFrame(model_records)
        model_out_dir = os.path.join(OUTPUT_DIR, model_key)
        os.makedirs(model_out_dir, exist_ok=True)
        model_csv = os.path.join(model_out_dir, "viewpoints.csv")
        df_model.to_csv(model_csv, index=False)
        
        # Compute stats
        w_counts = df_model["word_count"]
        med_w = w_counts.median() if len(w_counts) else 0
        mean_w = w_counts.mean() if len(w_counts) else 0
        pct_sweet = ((w_counts >= 30) & (w_counts <= 65)).mean() * 100 if len(w_counts) else 0
        claims_cnt = df_model["claim_id"].nunique() if "claim_id" in df_model.columns else 0
        vps_per_claim = len(df_model) / claims_cnt if claims_cnt else 0
        
        audit_summary.append({
            "Model": model_key,
            "Claims": claims_cnt,
            "Raw Gens": raw_count,
            "Exploded VPs": len(df_model),
            "VPs / Claim": f"{vps_per_claim:.1f}",
            "Median Words": f"{med_w:.0f}w",
            "Mean Words": f"{mean_w:.1f}w",
            "30-65w %": f"{pct_sweet:.1f}%"
        })
        print(f"  + [{model_key:25s}]: {raw_count:5d} raw gens -> {len(df_model):6d} viewpoints (Median: {med_w:.0f}w, SweetSpot: {pct_sweet:.1f}%)")
        
    master_df = pd.DataFrame(all_master_records)
    master_csv = os.path.join(OUTPUT_DIR, "master_viewpoints_all_models.csv")
    master_df.to_csv(master_csv, index=False)
    
    print("\n" + "=" * 115)
    print("MASTER VIEWPOINT SEPARATION AUDIT TABLE across ALL MODELS")
    print("=" * 115)
    print(pd.DataFrame(audit_summary).to_string(index=False))
    print("=" * 115)
    print(f"\n[SUCCESS] Master Dataset Saved: {master_csv} (Total Viewpoints: {len(master_df)})")

if __name__ == "__main__":
    process_all_models()
