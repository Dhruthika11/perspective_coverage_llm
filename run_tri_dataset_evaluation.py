"""
Master Tri-Dataset Evaluator (Perspectrum + IBM Debater + Kialo)
Runs directly on HPC.

Key Features:
1. Dynamically detects and calculates embeddings for:
   - All 4 Embedding Models (BGE-Large, BGE-Base, MPNet, Nomic)
   - Gold Perspectives across all 3 datasets
   - Model Generations across all 3 datasets
   - Caches all embeddings to disk under `data/embeddings/tri_dataset/` as .npy files.
     If embeddings exist, loads them instantly; if missing, calculates and saves them.
2. Evaluates all LLMs present across data/llm_outputs/
   (Qwen 0.5B-32B, LLaMA-3.2 1B-3B, Gemma-2 2B/9B, Gemma-3 12B, Mistral 7B).
3. Stage 2 DeBERTa Argument-NLI Cross-Encoder Verification on candidates (gamma=0.80).
4. Produces two complete CSV tables:
   - `results/combined_tri_dataset_cross_model_comparison.csv` (Primary Two-Stage Master Table)
   - `results/combined_tri_dataset_cross_embedding_matrix.csv` (Complete 4-Embedding Matrix)
"""
import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 80)
print("TRI-DATASET EVALUATION: WITH AUTOMATIC EMBEDDING CALCULATION & CACHING")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compute Device: {device}")

# --- 1. UNIFY CLAIMS & PERSPECTIVES ---
out_combined_dir = "data/processed/combined_tri_dataset"
os.makedirs(out_combined_dir, exist_ok=True)

# Helper to find text column safely
def get_text_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return df.select_dtypes(include=['object']).columns[0]

# Load Perspectrum
p_claims = pd.read_csv("data/processed/claims.csv")
p_persp = pd.read_csv("data/processed/perspectives.csv")
p_claims["dataset"] = "perspectrum"
p_persp["dataset"] = "perspectrum"

# Load IBM Debater
ibm_claims = pd.read_csv("data/processed/ibm_debater/claims.csv")
ibm_persp = pd.read_csv("data/processed/ibm_debater/perspectives.csv")
ibm_claims["dataset"] = "ibm_debater"
ibm_persp["dataset"] = "ibm_debater"

# Load Kialo
kialo_claims = pd.read_csv("data/processed/kialo/claims.csv")
kialo_persp = pd.read_csv("data/processed/kialo/perspectives.csv")
kialo_claims["dataset"] = "kialo"
kialo_persp["dataset"] = "kialo"

# Normalize text column names
p_p_col = get_text_col(p_persp, ["text", "perspective_text", "perspective_title"])
ibm_p_col = get_text_col(ibm_persp, ["text", "perspective_text", "argument"])
kialo_p_col = get_text_col(kialo_persp, ["text", "perspective_text", "argument"])

p_persp["text"] = p_persp[p_p_col].astype(str)
ibm_persp["text"] = ibm_persp[ibm_p_col].astype(str)
kialo_persp["text"] = kialo_persp[kialo_p_col].astype(str)

p_c_col = get_text_col(p_claims, ["claim_text", "claim", "topic"])
ibm_c_col = get_text_col(ibm_claims, ["claim_text", "claim", "topic"])
kialo_c_col = get_text_col(kialo_claims, ["claim_text", "claim", "topic"])

p_claims["claim_text"] = p_claims[p_c_col].astype(str)
ibm_claims["claim_text"] = ibm_claims[ibm_c_col].astype(str)
kialo_claims["claim_text"] = kialo_claims[kialo_c_col].astype(str)

# Global ID offsets
p_max_cid = int(p_claims["claim_id"].max())
p_max_pid = int(p_persp["perspective_id"].max())

ibm_claims["orig_claim_id"] = ibm_claims["claim_id"]
ibm_claims["claim_id"] = ibm_claims["orig_claim_id"] + p_max_cid
ibm_persp["orig_claim_id"] = ibm_persp["claim_id"]
ibm_persp["claim_id"] = ibm_persp["orig_claim_id"] + p_max_cid
ibm_persp["orig_perspective_id"] = ibm_persp["perspective_id"]
ibm_persp["perspective_id"] = ibm_persp["orig_perspective_id"] + p_max_pid

ibm_max_cid = int(ibm_claims["claim_id"].max())
ibm_max_pid = int(ibm_persp["perspective_id"].max())

kialo_claims["orig_claim_id"] = kialo_claims["claim_id"]
kialo_claims["claim_id"] = kialo_claims["orig_claim_id"] + ibm_max_cid
kialo_persp["orig_claim_id"] = kialo_persp["claim_id"]
kialo_persp["claim_id"] = kialo_persp["orig_claim_id"] + ibm_max_cid
kialo_persp["orig_perspective_id"] = kialo_persp["perspective_id"]
kialo_persp["perspective_id"] = kialo_persp["orig_perspective_id"] + ibm_max_pid

master_claims = pd.concat([
    p_claims[["claim_id", "claim_text", "dataset"]],
    ibm_claims[["claim_id", "claim_text", "dataset"]],
    kialo_claims[["claim_id", "claim_text", "dataset"]]
], ignore_index=True)

# Normalize stance column
for df in [p_persp, ibm_persp, kialo_persp]:
    if "stance" not in df.columns:
        df["stance"] = "support"

master_persp = pd.concat([
    p_persp[["claim_id", "perspective_id", "text", "stance", "dataset"]],
    ibm_persp[["claim_id", "perspective_id", "text", "stance", "dataset"]],
    kialo_persp[["claim_id", "perspective_id", "text", "stance", "dataset"]]
], ignore_index=True)

master_persp["stance"] = master_persp["stance"].astype(str).str.lower().str.strip()
master_persp["stance"] = master_persp["stance"].replace({
    "pro": "support", "con": "undermine", "supporting": "support", "opposing": "undermine"
})
master_persp["text_clean"] = master_persp["text"].fillna("").astype(str).str.strip()

master_claims.to_csv(os.path.join(out_combined_dir, "claims.csv"), index=False)
master_persp.to_csv(os.path.join(out_combined_dir, "perspectives.csv"), index=False)

print(f"\nUnified Tri-Dataset Claims: {len(master_claims):,}")
print(f"  - Perspectrum: {len(p_claims):,}")
print(f"  - IBM Debater: {len(ibm_claims):,}")
print(f"  - Kialo:       {len(kialo_claims):,}")

print(f"\nUnified Tri-Dataset Gold Perspectives: {len(master_persp):,}")
print(f"  - Perspectrum: {len(p_persp):,}")
print(f"  - IBM Debater: {len(ibm_persp):,}")
print(f"  - Kialo:       {len(kialo_persp):,}")

ibm_cid_map = dict(zip(ibm_claims["orig_claim_id"], ibm_claims["claim_id"]))
kialo_cid_map = dict(zip(kialo_claims["orig_claim_id"], kialo_claims["claim_id"]))

# --- 2. DISCOVER ALL AVAILABLE LLM MODELS ---
all_possible_models = [
    "qwen2_5_0_5b_instruct", "qwen2_5_1_5b_instruct", "qwen2_5_3b_instruct",
    "qwen2_5_7b_instruct", "qwen2_5_14b_instruct", "qwen2_5_32b_instruct",
    "llama_3_2_1b_instruct", "llama_3_2_3b_instruct",
    "gemma_2_2b_it", "gemma_2_9b_it", "gemma_3_12b_it", "mistral_7b_instruct_v0_3"
]

available_models = []
for m in all_possible_models:
    p_path = f"data/llm_outputs/{m}/processed_generations.csv"
    ibm_path = f"data/llm_outputs/ibm_debater/{m}/processed_generations.csv"
    kialo_path = f"data/llm_outputs/kialo/{m}/processed_generations.csv"
    if os.path.exists(p_path) or os.path.exists(ibm_path) or os.path.exists(kialo_path):
        available_models.append(m)

print(f"\nDiscovered {len(available_models)} LLM Models to Evaluate:")
for m in available_models:
    p_s = "P" if os.path.exists(f"data/llm_outputs/{m}/processed_generations.csv") else "-"
    ibm_s = "IBM" if os.path.exists(f"data/llm_outputs/ibm_debater/{m}/processed_generations.csv") else "-"
    k_s = "Kialo" if os.path.exists(f"data/llm_outputs/kialo/{m}/processed_generations.csv") else "-"
    print(f"  - {m:26} [{p_s:1} | {ibm_s:3} | {k_s:5}]")

# --- 3. CONSOLIDATE MODEL GENERATIONS ---
print("\nConsolidating model generations...")
model_generations = {}
for m in available_models:
    dfs = []
    p_gen = f"data/llm_outputs/{m}/processed_generations.csv"
    ibm_gen = f"data/llm_outputs/ibm_debater/{m}/processed_generations.csv"
    kialo_gen = f"data/llm_outputs/kialo/{m}/processed_generations.csv"

    if os.path.exists(p_gen):
        df_p = pd.read_csv(p_gen)
        df_p["dataset"] = "perspectrum"
        dfs.append(df_p)

    if os.path.exists(ibm_gen):
        df_ibm = pd.read_csv(ibm_gen)
        df_ibm["dataset"] = "ibm_debater"
        df_ibm["claim_id"] = df_ibm["claim_id"].map(ibm_cid_map)
        dfs.append(df_ibm)

    if os.path.exists(kialo_gen):
        df_kialo = pd.read_csv(kialo_gen)
        df_kialo["dataset"] = "kialo"
        df_kialo["claim_id"] = df_kialo["claim_id"].map(kialo_cid_map)
        dfs.append(df_kialo)

    if dfs:
        comb = pd.concat(dfs, ignore_index=True)
        # Select best text column
        text_col = None
        for cand in ["viewpoint_text", "combined_text", "argument_text", "generated_text", "text"]:
            if cand in comb.columns:
                text_col = cand
                break
        if text_col is None:
            text_col = comb.select_dtypes(include=['object']).columns[0]
        comb["text_clean"] = comb[text_col].fillna("").astype(str).str.strip()
        model_generations[m] = comb
        print(f"  + {m:26}: {len(comb):,} total viewpoints consolidated")

# Directory for caching all computed embeddings
EMB_CACHE_DIR = "data/embeddings/tri_dataset"
os.makedirs(EMB_CACHE_DIR, exist_ok=True)

# --- 4. 4 EMBEDDING MODEL CONFIGURATIONS ---
EMBEDDING_CONFIGS = [
    {
        "name": "BGE-Large",
        "slug": "bge_large",
        "model_id": "BAAI/bge-large-en-v1.5",
        "tau": 0.75,
        "is_primary": True,
    },
    {
        "name": "BGE-Base",
        "slug": "bge_base",
        "model_id": "BAAI/bge-base-en-v1.5",
        "tau": 0.74,
        "is_primary": False,
    },
    {
        "name": "All-MPNet-Base",
        "slug": "mpnet_base",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "tau": 0.68,
        "is_primary": False,
    },
    {
        "name": "Nomic-Embed-Text",
        "slug": "nomic_v1_5",
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "tau": 0.72,
        "is_primary": False,
    },
]

# Load Stage 2 NLI Model
nli_model_name = "models/argument_nli_deberta"
if not os.path.exists(nli_model_name):
    nli_model_name = "cross-encoder/nli-deberta-v3-small"
print(f"\nLoading Stage 2 NLI cross-encoder: {nli_model_name}...")
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
nli_model.eval()

id2label = {i: l.lower() for i, l in nli_model.config.id2label.items()}
entail_idx = next((i for i, l in id2label.items() if "entail" in l), 1)
GAMMA_NLI = 0.80

# Index gold perspectives by claim
master_persp["gold_row_idx"] = np.arange(len(master_persp))
persp_claim_groups = master_persp.groupby("claim_id")

primary_results = []
cross_embedding_matrix = {m: {} for m in available_models}

# --- 5. EVALUATION LOOP ACROSS EMBEDDINGS ---
for emb_cfg in EMBEDDING_CONFIGS:
    emb_name = emb_cfg["name"]
    slug = emb_cfg["slug"]
    model_id = emb_cfg["model_id"]
    tau = emb_cfg["tau"]
    is_primary = emb_cfg["is_primary"]

    print("\n" + "=" * 80)
    print(f"STEP: EMBEDDING MODEL -> {emb_name} ({model_id}) | tau = {tau}")
    print("=" * 80)

    # 5.1 Check / Calculate Gold Perspective Embeddings
    gold_emb_path = os.path.join(EMB_CACHE_DIR, f"gold_perspectives_{slug}.npy")
    embedder = None

    if os.path.exists(gold_emb_path):
        print(f"  [Found Cached] Gold perspective embeddings: {gold_emb_path}")
        gold_embs_all = np.load(gold_emb_path)
    else:
        print(f"  [Not Found] Calculating gold perspective embeddings for {len(master_persp)} texts...")
        try:
            if "nomic" in model_id.lower():
                embedder = SentenceTransformer(model_id, trust_remote_code=True, device=device)
            else:
                embedder = SentenceTransformer(model_id, device=device)
        except Exception as e:
            print(f"  Could not load {model_id}: {e}. Skipping {emb_name}.")
            continue

        gold_texts = master_persp["text_clean"].tolist()
        gold_embs_all = embedder.encode(
            gold_texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        np.save(gold_emb_path, gold_embs_all)
        print(f"  Saved gold embeddings to: {gold_emb_path} (shape: {gold_embs_all.shape})")

    # 5.2 Check / Calculate LLM Generation Embeddings and Evaluate
    for m in available_models:
        comb_gen = model_generations.get(m)
        if comb_gen is None or len(comb_gen) == 0:
            continue

        gen_emb_path = os.path.join(EMB_CACHE_DIR, f"{m}_{slug}.npy")
        if os.path.exists(gen_emb_path):
            gen_embs_all = np.load(gen_emb_path)
        else:
            print(f"  [Calculating Embeddings] {m} ({len(comb_gen)} texts) with {emb_name}...")
            if embedder is None:
                if "nomic" in model_id.lower():
                    embedder = SentenceTransformer(model_id, trust_remote_code=True, device=device)
                else:
                    embedder = SentenceTransformer(model_id, device=device)

            gen_texts = comb_gen["text_clean"].tolist()
            gen_embs_all = embedder.encode(
                gen_texts,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            np.save(gen_emb_path, gen_embs_all)
            print(f"  Saved {m} embeddings to: {gen_emb_path}")

        comb_gen["gen_row_idx"] = np.arange(len(comb_gen))
        gen_claim_groups = comb_gen.groupby("claim_id")

        covered_s1_pids = set()
        covered_s2_pids = set()

        claim_ids = master_claims["claim_id"].unique()

        for cid in tqdm(claim_ids, desc=f"{emb_name} -> {m}", leave=False):
            if cid not in persp_claim_groups.groups or cid not in gen_claim_groups.groups:
                continue

            c_persps = persp_claim_groups.get_group(cid)
            c_gens = gen_claim_groups.get_group(cid)

            p_indices = c_persps["gold_row_idx"].values
            g_indices = c_gens["gen_row_idx"].values

            gold_vecs = gold_embs_all[p_indices]  # shape: (n_gold, dim)
            gen_vecs = gen_embs_all[g_indices]    # shape: (n_gen, dim)

            sim_matrix = np.dot(gen_vecs, gold_vecs.T)  # shape: (n_gen, n_gold)
            max_sims = np.max(sim_matrix, axis=0)       # shape: (n_gold,)
            best_gen_idxs = np.argmax(sim_matrix, axis=0)

            nli_pairs = []
            candidate_pids = []

            pids = c_persps["perspective_id"].values
            gold_texts = c_persps["text_clean"].values
            gen_texts = c_gens["text_clean"].values

            for g_idx, sim in enumerate(max_sims):
                pid = pids[g_idx]
                if sim >= tau:
                    covered_s1_pids.add(pid)
                    if is_primary:
                        nli_pairs.append((gen_texts[best_gen_idxs[g_idx]], gold_texts[g_idx]))
                        candidate_pids.append(pid)

            if is_primary and len(nli_pairs) > 0:
                with torch.no_grad():
                    for b_start in range(0, len(nli_pairs), 32):
                        b_pairs = nli_pairs[b_start : b_start + 32]
                        enc = nli_tokenizer(b_pairs, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                        logits = nli_model(**enc).logits
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()

                        for row_i, prob_vec in enumerate(probs):
                            if np.argmax(prob_vec) == entail_idx and prob_vec[entail_idx] >= GAMMA_NLI:
                                covered_s2_pids.add(candidate_pids[b_start + row_i])

        tot_p = len(master_persp)
        s1_cov = round(len(covered_s1_pids) / tot_p, 4)
        cross_embedding_matrix[m][f"{emb_name} (tau={tau})"] = s1_cov

        if is_primary:
            s2_cov = round(len(covered_s2_pids) / tot_p, 4)
            master_persp["is_covered_twostage"] = master_persp["perspective_id"].isin(covered_s2_pids)
            claim_cov_counts = master_persp.groupby("claim_id")["is_covered_twostage"].sum()
            zero_cov_claims = int((claim_cov_counts == 0).sum())

            entry = {
                "model": m,
                "total_claims": len(master_claims),
                "total_perspectives": tot_p,
                "combined_two_stage_coverage": s2_cov,
                "combined_single_stage_coverage": s1_cov,
                "combined_zero_coverage_claims": zero_cov_claims,
            }

            for dset in ["perspectrum", "ibm_debater", "kialo"]:
                sub_p = master_persp[master_persp["dataset"] == dset]
                if len(sub_p) > 0:
                    entry[f"{dset}_coverage"] = round(sub_p["is_covered_twostage"].mean(), 4)
                    entry[f"{dset}_support_cov"] = round(sub_p[sub_p["stance"] == "support"]["is_covered_twostage"].mean(), 4)
                    entry[f"{dset}_undermine_cov"] = round(sub_p[sub_p["stance"] == "undermine"]["is_covered_twostage"].mean(), 4)

            primary_results.append(entry)
            print(f"  => {m}: Two-Stage = {s2_cov:.4f} | Single-Stage = {s1_cov:.4f} (Zero Claims: {zero_cov_claims})")
        else:
            print(f"  => {m}: {emb_name} Single-Stage Coverage = {s1_cov:.4f}")

    if embedder is not None:
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- 6. SAVE RESULTS ---
os.makedirs("results", exist_ok=True)
df_primary = pd.DataFrame(primary_results)
df_primary.to_csv("results/combined_tri_dataset_cross_model_comparison.csv", index=False)

df_matrix = pd.DataFrame.from_dict(cross_embedding_matrix, orient="index")
df_matrix.index.name = "model"
df_matrix.reset_index(inplace=True)
df_matrix.to_csv("results/combined_tri_dataset_cross_embedding_matrix.csv", index=False)

print("\n" + "=" * 80)
print("PRIMARY TWO-STAGE MASTER COMPARISON TABLE:")
print("=" * 80)
print(df_primary.to_string())

print("\n" + "=" * 80)
print("CROSS-EMBEDDING MATRIX (ALL 4 EMBEDDINGS):")
print("=" * 80)
print(df_matrix.to_string())
print("\nALL EVALUATIONS COMPLETE!")
