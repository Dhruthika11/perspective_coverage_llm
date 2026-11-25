"""
Sentence-Level Chunking & Max-Pooling Retriever for Stage 1 Dense Retrieval.

Solves the length dilution penalty in dense embedding retrieval:
  Instead of embedding a 200-word paragraph as a single diluted vector e_p,
  we segment the premise into individual sentences {s_1, s_2, ..., s_k}, embed each,
  and compute the max-pooled similarity against the concise perspective hypothesis e_h:
      sim_max(p, h) = max_{s in sentences(p)} cos(e_s, e_h)

This ensures a 200-word argument is evaluated on its best matching sentence
rather than being penalized for its length.
"""

import os
import re
import sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 1. Robust Sentence Segmentation
COMMON_ABBREVIATIONS = ["e.g.", "i.e.", "u.s.", "dr.", "mr.", "ms.", "vs.", "etc.", "inc.", "ltd."]

def split_into_sentences(text: str, min_words: int = 4) -> List[str]:
    """
    Splits text into clean sentences or clauses.
    Filters out trivial fragments and protects common abbreviations.
    """
    if not isinstance(text, str) or not text.strip():
        return [str(text)]

    text = text.strip()
    
    # Protect common abbreviations so periods don't trigger false sentence splits
    clean_text = text
    for a in COMMON_ABBREVIATIONS:
        clean_text = re.sub(re.escape(a), a.replace(".", "@DOT@"), clean_text, flags=re.IGNORECASE)

    # Split on sentence terminals followed by whitespace or line breaks
    raw_sents = re.split(r"(?<=[.!?])\s+|\n+", clean_text)
    sents = []
    
    for s in raw_sents:
        # Restore periods
        s_clean = s.replace("@DOT@", ".").strip()
        # Strip markdown headers or bold prefix tags (e.g. "**Legal Perspective:**")
        s_clean = re.sub(r"^\*{1,2}[^*]+\*{1,2}:?\s*", "", s_clean)
        # Strip bullet points, list numbers (e.g. "1.", "- ", "* ")
        s_clean = re.sub(r"^\s*[-*•\d\.)]+\s*", "", s_clean).strip()
        
        if len(s_clean.split()) >= min_words:
            sents.append(s_clean)

    # If segmentation produced nothing, fall back to the full text
    return sents if sents else [text]


class SentenceMaxPoolingRetriever:
    """
    Stage 1 Dense Retriever with Sentence-Level Chunking & Max-Pooling.
    
    Computes both:
      1. Monolithic Similarity: cos(e_premise, e_hypothesis)
      2. Sentence Max-Pooled Similarity: max_{s in sentences(premise)} cos(e_s, e_hypothesis)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[SentenceMaxPoolingRetriever] Loading {model_name} on {self.device}...")
        
        # Load HuggingFace model + tokenizer directly for maximum compatibility
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.hidden_dim = getattr(self.model.config, "hidden_size", 1024)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Computes L2-normalized CLS embeddings in batches."""
        if not texts:
            return np.empty((0, self.hidden_dim), dtype=np.float32)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embs = outputs.last_hidden_state[:, 0]
                if self.normalize:
                    cls_embs = F.normalize(cls_embs, p=2, dim=1)
                all_embs.append(cls_embs.cpu().numpy())

        return np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, self.hidden_dim), dtype=np.float32)

    def score_pair(self, premise: str, hypothesis: str, min_words: int = 4) -> Dict[str, Any]:
        """Computes monolithic vs sentence max-pooled similarity for a single pair."""
        sents = split_into_sentences(premise, min_words=min_words)
        
        # Embed all components together
        all_texts = [premise] + sents + [hypothesis]
        embs = self.embed_texts(all_texts)
        
        e_mono = embs[0]
        e_sents = embs[1 : 1 + len(sents)]
        e_hyp = embs[-1]
        
        mono_sim = float(np.dot(e_mono, e_hyp))
        sent_sims = [float(np.dot(es, e_hyp)) for es in e_sents]
        
        best_idx = int(np.argmax(sent_sims))
        max_sim = float(sent_sims[best_idx])
        winning_sent = sents[best_idx]
        
        return {
            "monolithic_similarity": round(mono_sim, 4),
            "max_similarity": round(max_sim, 4),
            "similarity_delta": round(max_sim - mono_sim, 4),
            "winning_sentence": winning_sent,
            "winning_sentence_index": best_idx,
            "num_sentences": len(sents),
            "sentence_similarities": [round(s, 4) for s in sent_sims]
        }

    def batch_score_pairs(
        self,
        premises: List[str],
        hypotheses: List[str],
        batch_size: int = 64,
        min_words: int = 4
    ) -> pd.DataFrame:
        """
        Efficiently scores a batch of pairs:
        Deduplicates premises, sentences, and hypotheses to minimize forward passes.
        """
        assert len(premises) == len(hypotheses), "Length of premises and hypotheses must match."
        n_pairs = len(premises)
        
        # 1. Segment all premises into sentences
        pair_sentences = [split_into_sentences(p, min_words=min_words) for p in premises]
        
        # 2. Collect unique strings to embed
        unique_texts = list(set(premises + [s for sents in pair_sentences for s in sents] + hypotheses))
        print(f"[SentenceMaxPoolingRetriever] Embedding {len(unique_texts)} unique text strings...")
        
        embs = self.embed_texts(unique_texts, batch_size=batch_size)
        emb_lookup = {text: embs[i] for i, text in enumerate(unique_texts)}
        
        # 3. Compute monolithic and max-pooled scores
        results = []
        for i in range(n_pairs):
            p = premises[i]
            h = hypotheses[i]
            sents = pair_sentences[i]
            
            e_p = emb_lookup[p]
            e_h = emb_lookup[h]
            mono_sim = float(np.dot(e_p, e_h))
            
            sent_sims = [float(np.dot(emb_lookup[s], e_h)) for s in sents]
            best_idx = int(np.argmax(sent_sims))
            max_sim = float(sent_sims[best_idx])
            winning_sent = sents[best_idx]
            
            results.append({
                "monolithic_similarity": round(mono_sim, 4),
                "max_similarity": round(max_sim, 4),
                "similarity_delta": round(max_sim - mono_sim, 4),
                "num_sentences": len(sents),
                "winning_sentence": winning_sent,
                "winning_sentence_index": best_idx
            })
            
        return pd.DataFrame(results)
