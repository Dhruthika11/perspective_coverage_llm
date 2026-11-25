#!/usr/bin/env python3
"""
Prints and exports head(5) of each model's viewpoint split dataset.
"""

import os
import sys
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "data", "viewpoint_separated")
OUTPUT_MD = os.path.join(REPO_ROOT, "tests", "head5_all_models_viewpoints.md")
OUTPUT_CSV = os.path.join(REPO_ROOT, "tests", "head5_all_models_viewpoints.csv")

def main():
    models = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith(".")
    ])
    
    all_head_dfs = []
    lines = []
    lines.append(f"# Head(5) Viewpoints for All {len(models)} Models\n")

    print(f"Total models found: {len(models)}")
    for idx, m in enumerate(models, 1):
        vp_path = os.path.join(BASE_DIR, m, "viewpoints.csv")
        if os.path.exists(vp_path) and os.path.getsize(vp_path) > 0:
            df = pd.read_csv(vp_path, nrows=5)
            all_head_dfs.append(df)
            
            header = f"### [{idx}/{len(models)}] `{m}` (Total rows in head: {len(df)})"
            print(f"\n{header}")
            lines.append(f"\n{header}\n")
            
            # Format markdown table
            cols_show = ["claim_id", "sample_id", "viewpoint_id", "normalized_stance", "extraction_method", "word_count", "viewpoint_text"]
            cols_show = [c for c in cols_show if c in df.columns]
            
            table_header = "| CID | SID | VID | Stance | Extraction Method | Words | Viewpoint Text |"
            table_sep    = "|:---:|:---:|:---:|:---:|:---|:---:|:---|"
            print(table_header)
            print(table_sep)
            lines.append(table_header)
            lines.append(table_sep)
            
            for _, r in df[cols_show].iterrows():
                cid = r.get("claim_id", "")
                sid = r.get("sample_id", "")
                vid = r.get("viewpoint_id", "")
                st = r.get("normalized_stance", "")
                meth = r.get("extraction_method", "")
                wc = r.get("word_count", "")
                txt = str(r.get("viewpoint_text", "")).replace("\n", " ").replace("|", "\\|")
                txt_trunc = (txt[:80] + "...") if len(txt) > 83 else txt
                
                row_str = f"| {cid} | {sid} | {vid} | `{st}` | `{meth}` | {wc} | {txt_trunc} |"
                print(row_str)
                lines.append(f"| {cid} | {sid} | {vid} | `{st}` | `{meth}` | {wc} | {txt} |")
        else:
            print(f"[{idx}/{len(models)}] {m}: No viewpoints found (or empty).")
            lines.append(f"\n### [{idx}/{len(models)}] `{m}`\n*No viewpoints found (or empty).*\n")

    # Save to CSV and Markdown
    if all_head_dfs:
        df_all = pd.concat(all_head_dfs, ignore_index=True)
        df_all.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved combined CSV to: {OUTPUT_CSV} ({len(df_all)} rows)")
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved complete markdown to: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
