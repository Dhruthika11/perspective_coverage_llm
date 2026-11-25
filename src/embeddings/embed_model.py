"""
Loads embedding models (BGE-Large, BGE-Base, MPNet, Nomic) with robust multi-backend support:
  1. Primary: sentence_transformers (with transformers compatibility monkey-patches).
  2. Fallback: native Hugging Face AutoModel + AutoTokenizer with CLS pooling and L2 normalization.
"""
import torch
import numpy as np
import transformers
import transformers.utils

# 1. Compatibility patch for transformers.utils.is_kernels_available
if not hasattr(transformers.utils, "is_kernels_available"):
    transformers.utils.is_kernels_available = lambda: False

try:
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, "is_kernels_available"):
        transformers.utils.import_utils.is_kernels_available = lambda: False
except Exception:
    pass

# 2. Compatibility patch for nomic-bert-2048 remote code with transformers >= 4.45
if not hasattr(transformers.PreTrainedModel, "get_extended_attention_mask"):
    def _get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")
        return (1.0 - extended) * -10000.0
    transformers.PreTrainedModel.get_extended_attention_mask = _get_extended_attention_mask


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = "sentence_transformer"

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, trust_remote_code=True).to(self.device)
        except Exception:
            # Native Hugging Face fallback (never fails on sentence_transformers import errors)
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.mode = "hf_native"

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            dim = getattr(self.model.config, "hidden_size", 1024) if hasattr(self.model, "config") else 1024
            return np.empty((0, dim), dtype=np.float32)

        if self.mode == "sentence_transformer":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings
        else:
            all_embs = []
            for i in range(0, len(texts), batch_size):
                b_texts = texts[i : i + batch_size]
                enc = self.tokenizer(
                    b_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    out = self.model(**enc)
                    # BGE official CLS representation
                    cls_emb = out.last_hidden_state[:, 0]
                    if self.normalize:
                        cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                    all_embs.append(cls_emb.cpu().numpy())
            dim = getattr(self.model.config, "hidden_size", 1024) if hasattr(self.model, "config") else 1024
            return np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, dim), dtype=np.float32)
