#!/usr/bin/env python3
"""
Model Quality Dashboard Generator for Perspective Coverage.

Compiles a unified model-level quality dashboard across all 34 models:
  - Raw generations (N_raw)
  - Retained generations (N_retained)
  - Quarantined generations (N_quarantined)
  - Quarantine rate
  - Abstention count & Abstention rate (explicit model refusals)
  - Defect quarantine count & Defect rate (leakage, task hallucinations, degenerate loops)
  - Task Usability Rate (1 - quarantine_rate)
  - Extracted viewpoints (N_viewpoints)
  - Mean viewpoints per retained generation
  - Median and Mean viewpoint words
  - Explicit stance viewpoints count & Explicit stance coverage %
  - Unknown stance viewpoints count & Unknown stance rate %
  - Conditional Support % and Conditional Undermine % (among explicit stance viewpoints)

Outputs:
  - data/viewpoint_separated/model_quality_dashboard.csv
  - tests/model_quality_dashboard.csv
"""

import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "viewpoint_separated")
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

MASTER_VP_PATH = os.path.join(OUTPUT_DIR, "master_viewpoints_all_models.csv")
MASTER_QUAR_PATH = os.path.join(OUTPUT_DIR, "master_quarantined_all_models.csv")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "extraction_manifest.csv")


def main():
    print("=" * 80)
    print("BUILDING UNIFIED MODEL QUALITY DASHBOARD")
    print("=" * 80)

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    df_manifest = pd.read_csv(MANIFEST_PATH)

    df_quar = pd.DataFrame()
    if os.path.exists(MASTER_QUAR_PATH) and os.path.getsize(MASTER_QUAR_PATH) > 0:
        df_quar = pd.read_csv(MASTER_QUAR_PATH)

    # We read per-model viewpoints to be memory efficient and fast
    dashboard_rows = []

    for _, m_row in df_manifest.iterrows():
        m = m_row["model_key"]
        raw_n = int(m_row["raw_samples"])
        quar_n = int(m_row["quarantined_samples"])
        retained_n = raw_n - quar_n
        q_rate = quar_n / raw_n if raw_n > 0 else 0.0
        usability_rate = retained_n / raw_n if raw_n > 0 else 0.0

        # Load model quarantined
        m_q = df_quar[df_quar["model_key"] == m] if len(df_quar) > 0 and "model_key" in df_quar.columns else pd.DataFrame()
        q_reasons = m_q["quarantine_reason"].value_counts().to_dict() if len(m_q) > 0 else {}
        
        abstentions = q_reasons.get("ABSTENTION", q_reasons.get("QuarantineReason.ABSTENTION", 0))
        prompt_leakage = q_reasons.get("PROMPT_LEAKAGE", q_reasons.get("QuarantineReason.PROMPT_LEAKAGE", 0))
        hallucinated_task = q_reasons.get("HALLUCINATED_TASK", q_reasons.get("QuarantineReason.HALLUCINATED_TASK", 0))
        truncation = q_reasons.get("UNRECOVERABLE_TRUNCATION", q_reasons.get("QuarantineReason.UNRECOVERABLE_TRUNCATION", 0))
        empty = q_reasons.get("EMPTY_OR_WHITESPACE", q_reasons.get("QuarantineReason.EMPTY_OR_WHITESPACE", 0))
        exec_err = q_reasons.get("EXECUTION_ERROR", q_reasons.get("QuarantineReason.EXECUTION_ERROR", 0))

        defect_count = prompt_leakage + hallucinated_task + truncation + empty + exec_err
        abstention_rate = abstentions / raw_n if raw_n > 0 else 0.0
        defect_rate = defect_count / raw_n if raw_n > 0 else 0.0

        # Load model viewpoints
        vp_path = os.path.join(OUTPUT_DIR, m, "viewpoints.csv")
        df_m_vp = pd.DataFrame()
        if os.path.exists(vp_path) and os.path.getsize(vp_path) > 0:
            df_m_vp = pd.read_csv(vp_path)

        total_vps = len(df_m_vp)
        mean_vp_per_retained = round(total_vps / retained_n, 2) if retained_n > 0 else 0.0
        median_words = round(float(df_m_vp["word_count"].median()), 1) if total_vps > 0 else 0.0
        mean_words = round(float(df_m_vp["word_count"].mean()), 1) if total_vps > 0 else 0.0

        # Stance counts
        st_counts = df_m_vp["normalized_stance"].value_counts().to_dict() if total_vps > 0 else {}
        n_sup = st_counts.get("support", 0)
        n_und = st_counts.get("undermine", 0)
        n_unk = st_counts.get("unknown", 0) + st_counts.get("unknown_or_ambiguous", 0)
        n_explicit = n_sup + n_und

        explicit_coverage_pct = round(n_explicit / total_vps * 100, 2) if total_vps > 0 else 0.0
        unknown_rate_pct = round(n_unk / total_vps * 100, 2) if total_vps > 0 else 0.0

        cond_sup_pct = round(n_sup / n_explicit * 100, 2) if n_explicit > 0 else np.nan
        cond_und_pct = round(n_und / n_explicit * 100, 2) if n_explicit > 0 else np.nan

        # Format archetype
        methods = df_m_vp["extraction_method"].value_counts(normalize=True).to_dict() if total_vps > 0 else {}
        pct_json = round(methods.get("balanced_json", methods.get("balanced_json_object", 0.0)) * 100, 1)
        pct_bullet = round(methods.get("bulleted_list_item", 0.0) * 100, 1)
        pct_numbered = round(methods.get("numbered_list_item", 0.0) * 100, 1)
        pct_headings = round((methods.get("markdown_heading_section", 0.0) + methods.get("markdown_bold_heading", 0.0)) * 100, 1)
        pct_prose = round((methods.get("prose_paragraph", 0.0) + methods.get("single_prose_block", 0.0)) * 100, 1)

        if pct_json > 75:
            archetype = "Structured JSON"
        elif pct_bullet + pct_numbered > 50:
            archetype = "List / Bullet Structure"
        elif pct_headings > 50:
            archetype = "Section Headings"
        elif pct_prose > 50:
            archetype = "Prose Paragraphs"
        else:
            archetype = "Hybrid / Multi-format"

        # Model Family
        if m.startswith("gemma"):
            family = "Gemma 2"
        elif m.startswith("llama"):
            family = "LLaMA 3"
        elif m.startswith("olmoe"):
            family = "OLMoE"
        elif m.startswith("olmo"):
            family = "OLMo 2"
        elif m.startswith("qwen"):
            family = "Qwen 2.5"
        elif m.startswith("smollm"):
            family = "SmolLM2"
        else:
            family = "Other"

        dashboard_rows.append({
            "model_key": m,
            "family": family,
            "archetype": archetype,
            "raw_generations": raw_n,
            "retained_generations": retained_n,
            "quarantined_generations": quar_n,
            "quarantine_rate_pct": round(q_rate * 100, 2),
            "task_usability_rate_pct": round(usability_rate * 100, 2),
            "abstention_count": abstentions,
            "abstention_rate_pct": round(abstention_rate * 100, 2),
            "defect_count": defect_count,
            "defect_rate_pct": round(defect_rate * 100, 2),
            "viewpoints_extracted": total_vps,
            "mean_viewpoints_per_retained_gen": mean_vp_per_retained,
            "median_viewpoint_words": median_words,
            "mean_viewpoint_words": mean_words,
            "explicit_stance_viewpoints": n_explicit,
            "explicit_stance_coverage_pct": explicit_coverage_pct,
            "unknown_stance_viewpoints": n_unk,
            "unknown_stance_rate_pct": unknown_rate_pct,
            "support_viewpoints_count": n_sup,
            "undermine_viewpoints_count": n_und,
            "conditional_support_pct": cond_sup_pct,
            "conditional_undermine_pct": cond_und_pct,
            "pct_json": pct_json,
            "pct_bullet": pct_bullet,
            "pct_numbered": pct_numbered,
            "pct_headings": pct_headings,
            "pct_prose": pct_prose
        })

    df_dashboard = pd.DataFrame(dashboard_rows).sort_values(
        ["family", "model_key"]
    ).reset_index(drop=True)

    out_csv1 = os.path.join(OUTPUT_DIR, "model_quality_dashboard.csv")
    out_csv2 = os.path.join(TESTS_DIR, "model_quality_dashboard.csv")
    df_dashboard.to_csv(out_csv1, index=False)
    df_dashboard.to_csv(out_csv2, index=False)

    print(f"Saved Unified Model Quality Dashboard to:\n  - {out_csv1}\n  - {out_csv2}")
    print(f"Total Models Cataloged: {len(df_dashboard)}")
    print(f"Total Raw Generations: {df_dashboard['raw_generations'].sum():,}")
    print(f"Total Retained Generations: {df_dashboard['retained_generations'].sum():,}")
    print(f"Total Quarantined: {df_dashboard['quarantined_generations'].sum():,}")
    print(f"Total Extracted Viewpoints: {df_dashboard['viewpoints_extracted'].sum():,}")
    print(f"Benchmark-Wide Explicit Stances: {df_dashboard['explicit_stance_viewpoints'].sum():,} ({df_dashboard['explicit_stance_viewpoints'].sum()/df_dashboard['viewpoints_extracted'].sum()*100:.1f}%)")
    print(f"Benchmark-Wide Unknown Stances: {df_dashboard['unknown_stance_viewpoints'].sum():,} ({df_dashboard['unknown_stance_viewpoints'].sum()/df_dashboard['viewpoints_extracted'].sum()*100:.1f}%)")


if __name__ == "__main__":
    main()

