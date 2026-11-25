import sys, os
sys.path.append(os.path.abspath('.'))
import time
import numpy as np
from src.embeddings.embed_model import BGEEmbedder

embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", normalize=True)
dummy_texts = ["This is a test sentence about perspectives and arguments." for _ in range(64)]

t0 = time.time()
embs = embedder.embed(dummy_texts, batch_size=32)
elapsed = time.time() - t0
print(f"Embedding 64 texts took {elapsed:.2f}s ({elapsed/64:.4f}s per text, {64/elapsed:.1f} texts/s)")
print(f"Estimated time for 1,656 texts: {elapsed/64 * 1656 / 60:.2f} minutes")
