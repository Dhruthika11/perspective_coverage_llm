"""
Export Viewpoint-Separated Datasets Across All LLM Generations.

Parses multi-viewpoint generations into discrete, normalized viewpoint rows (~35-60 words).
Handles:
  1. Single-perspective JSON models (Gemma, Llama, OLMo, Qwen-7B): parses markdown code blocks,
     combining perspective_summary + core_reasoning (normalized to ~45-59 words).
  2. Multi-perspective list models (Qwen-1.5B, 3B, 32B): discards preamble intros,
     extracts each individual viewpoint argument (~45-65 words).

Outputs clean, structured tables for each model and a unified master table:
  - data/viewpoint_separated/<model_key>/viewpoints.csv
  - data/viewpoint_separated/master_viewpoints_all_models.csv
"""

import os
import re
import glob
import json
import pandas as pd
import numpy as np

OUTPUT_DIR = "data/viewpoint_separated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_json_text(text: str):
    """Attempts to parse JSON object from markdown or raw text."""
    text_clean = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text_clean = re.sub(r'\s*```$', '', text_clean.strip())
    start = text_clean.find('{')
    end = text_clean.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text_clean[start:end+1]
        try:
            obj = json.loads(candidate)
            ps = obj.get('perspective_summary') or obj.get('argument') or ''
            cr = obj.get('core_reasoning') or ''
            stance = obj.get('stance') or ''
            comb = f"{ps} {cr}".strip() if cr else str(ps).strip()
            if comb:
                return [{
                    'viewpoint_text': comb,
                    'stance': stance,
                    'perspective_summary': str(ps).strip(),
                    'core_reasoning': str(cr).strip()
                }]
        except Exception:
            pass
    return None

def split_multiviewpoint(text: str) -> list[str]:
    """Splits multi-viewpoint text into individual viewpoint strings, discarding preambles."""
    text = text.strip()
    
    # Discard non-argumentative preamble preceding bullet 1
    m = re.search(r'(?:\A|\n+)\s*(?:1[\.\)]|\bViewpoint\s+1:?|\bPerspective\s+1:?)\s+', text, flags=re.IGNORECASE)
    if m:
        text_body = text[m.start():]
    else:
        text_body = text
        
    # Split on numbered markers
    pattern_num = r'\n+\s*(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?)\s+|\A(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?)\s+'
    splits = re.split(pattern_num, text_body, flags=re.IGNORECASE)
    splits = [s.strip() for s in splits if len(s.strip().split()) >= 10]
    if len(splits) >= 2:
        return splits
        
    # Markdown bold headers
    m_hdr = re.search(r'(?:\A|\n+)\s*\*\*(?:Viewpoint|Perspective)\s+1:?\*\*', text, flags=re.IGNORECASE)
    if m_hdr:
        text_body = text[m_hdr.start():]
    else:
        text_body = text
    pattern_hdr = r'\n+\s*\*\*[^*]+(?::\*\*|\*\*:?)\s*|\A\*\*[^*]+(?::\*\*|\*\*:?)\s*'
    splits = re.split(pattern_hdr, text_body)
    splits = [s.strip() for s in splits if len(s.strip().split()) >= 10]
    if len(splits) >= 2:
        return splits
        
    # Double newline paragraphs if long enough
    splits = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) >= 15]
    if len(splits) >= 2:
        return splits
        
    return [text]

def process_all_models():
    print("=" * 95)
    print("EXTRACTING & SAVING VIEWPOINT-SEPARATED DATASETS (PRECISION NORMALIZED)")
    print("=" * 95)
    
    gen_files = sorted(glob.glob("data/llm_outputs/**/processed_generations.csv", recursive=True))
    all_master_records = []
    
    for fpath in gen_files:
        model_key = os.path.basename(os.path.dirname(fpath))
        if model_key in ["ibm_debater", "kialo"]:
            continue
            
        print(f"\nProcessing model: {model_key} from {fpath}...")
        df = pd.read_csv(fpath)
        
        text_col = "generated_text" if "generated_text" in df.columns else ("argument" if "argument" in df.columns else "combined_text")
        
        model_records = []
        for idx, row in df.iterrows():
            cid = row.get("claim_id")
            sid = row.get("sample_id", idx)
            orig_text = str(row[text_col])
            
            # 1. Try structured JSON parsing first
            parsed_json = clean_json_text(orig_text)
            if parsed_json:
                for vp_idx, item in enumerate(parsed_json):
                    vp_text = item['viewpoint_text']
                    w_count = len(vp_text.split())
                    rec = {
                        "model_key": model_key,
                        "claim_id": cid,
                        "sample_id": sid,
                        "viewpoint_id": vp_idx + 1,
                        "viewpoint_text": vp_text,
                        "word_count": w_count,
                        "stance": item.get('stance', '')
                    }
                    model_records.append(rec)
                    all_master_records.append(rec)
            else:
                # 2. Multi-viewpoint list splitting
                vps = split_multiviewpoint(orig_text)
                for vp_idx, vp in enumerate(vps):
                    # Clean up leading numbers or markdown headers
                    vp_clean = re.sub(r'^\*{1,2}[^*]+\*{1,2}:?\s*', '', vp).strip()
                    vp_clean = re.sub(r'^\s*[-*•\d\.)]+\s*', '', vp_clean).strip()
                    w_count = len(vp_clean.split())
                    
                    rec = {
                        "model_key": model_key,
                        "claim_id": cid,
                        "sample_id": sid,
                        "viewpoint_id": vp_idx + 1,
                        "viewpoint_text": vp_clean,
                        "word_count": w_count,
                        "stance": ""
                    }
                    model_records.append(rec)
                    all_master_records.append(rec)
                    
        model_df = pd.DataFrame(model_records)
        model_out_dir = os.path.join(OUTPUT_DIR, model_key)
        os.makedirs(model_out_dir, exist_ok=True)
        model_csv_path = os.path.join(model_out_dir, "viewpoints.csv")
        model_df.to_csv(model_csv_path, index=False)
        
        w_counts = model_df["word_count"]
        in_35_60 = ((w_counts >= 35) & (w_counts <= 60)).mean() * 100
        in_30_65 = ((w_counts >= 30) & (w_counts <= 65)).mean() * 100
        print(f"  + Saved {len(model_df)} viewpoints to {model_csv_path}")
        print(f"  + Lengths: Mean={w_counts.mean():.1f}w, Median={w_counts.median():.1f}w | 35-60w: {in_35_60:.1f}%, 30-65w: {in_30_65:.1f}%")
        
    master_df = pd.DataFrame(all_master_records)
    master_csv_path = os.path.join(OUTPUT_DIR, "master_viewpoints_all_models.csv")
    master_df.to_csv(master_csv_path, index=False)
    
    print("\n" + "=" * 95)
    print(f"MASTER FILE CREATED: {master_csv_path}")
    print(f"Total extracted viewpoints across all models: {len(master_df)}")
    print("=" * 95)

if __name__ == "__main__":
    process_all_models()
