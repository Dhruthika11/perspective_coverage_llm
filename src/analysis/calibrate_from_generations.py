"""
Calibrates the similarity threshold (tau) empirically by analyzing the internal
similarity of the LLM's own generations.

Because the new generation script uses Aspect Stratification and History Injection
(Mixed Strategy 1+2), every sample generated for a claim is explicitly forced to
be a DISTINCT argument (e.g., one Economic, one Legal, one Ethical). 

Therefore, by calculating the pairwise cosine similarity (dot product) between
the k generated samples for each claim, we can observe exactly how similar the 
embedding model considers two *distinct* LLM viewpoints to be.

The recommended threshold is the upper bound (e.g., 95th percentile) of this
distribution to prevent the pipeline from accidentally clustering genuinely 
distinct generated arguments together.
"""
import os
import sys
import numpy as np

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config
from src.analysis.similarity_matrix import load_embeddings, cosine_sim_matrix
from src.utils.logging_utils import get_logger

logger = get_logger("calibrate_from_generations")

def main():
    config = load_config()
    logger.info(f"Loading embeddings... Model: {config['embedding']['model']}")
    
    # Load embeddings (only LLM embeddings are needed here)
    _, _, llm_emb, llm_idx = load_embeddings(config)
    
    claim_ids = sorted(set(llm_idx["claim_id"]))
    intra_claim_sims = []
    
    for claim_id in claim_ids:
        l_mask = (llm_idx["claim_id"] == claim_id).values
        l_emb_claim = llm_emb[l_mask]
        
        n_samples = l_emb_claim.shape[0]
        if n_samples > 1:
            # Pairwise cosine similarities of the generated samples
            sim = cosine_sim_matrix(l_emb_claim, l_emb_claim)
            
            # Extract upper triangle (excluding the self-similarity diagonal of 1.0)
            triu_idx = np.triu_indices(n_samples, k=1)
            intra_claim_sims.extend(sim[triu_idx])
            
    if not intra_claim_sims:
        logger.error("No LLM generation pairs found to analyze! Have you generated samples and embedded them yet?")
        return

    intra_claim_sims = np.array(intra_claim_sims)
    
    print("\n" + "="*80)
    print(f" LLM GENERATION THRESHOLD CALIBRATION ({config['embedding']['model']})")
    print("="*80)
    print(f"Analyzed {len(intra_claim_sims)} pairs of LLM-generated viewpoints.")
    print("Because the Mixed Strategy forces the model to generate orthogonal arguments")
    print("(e.g., Economic vs Legal), these pairs represent sentences that are semantically")
    print("distinct despite being about the same claim.")
    print("\nDistribution of Dot Products (Cosine Similarity) between distinct generations:")
    print("-" * 75)
    
    percentiles = [50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(intra_claim_sims, p)
        print(f" {p:>4}th percentile : {val:.4f}")
        
    print(f"\n Mean similarity: {np.mean(intra_claim_sims):.4f}")
    print(f" Max similarity:  {np.max(intra_claim_sims):.4f}")
    
    recommended_tau = np.percentile(intra_claim_sims, 95)
    print("\n" + "*" * 80)
    print(f" RECOMMENDED THRESHOLD (tau): {recommended_tau:.3f}")
    print("*" * 80)
    print(f"At tau = {recommended_tau:.3f}, the pipeline is 95% confident that it will NOT")
    print("falsely merge two distinct LLM-generated arguments into a single cluster.")
    print(f"You can set this in your configs/config.yaml under analysis.similarity_threshold.")

if __name__ == "__main__":
    main()
