import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

print("=" * 80)
print("TESTING FLAIR2 GPU & MODEL LOADING PIPELINE")
print("=" * 80)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Test BGE
print("Loading BGE-Large (BAAI/bge-large-en-v1.5)...")
bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
bge_mod = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device).eval()
print("BGE-Large loaded successfully!")

# Test DeBERTa
deberta_path = "models/argument_nli_deberta_v3_v3"
print(f"Loading DeBERTa from {deberta_path}...")
deb_tok = AutoTokenizer.from_pretrained(deberta_path)
deb_mod = AutoModelForSequenceClassification.from_pretrained(deberta_path).to(device).eval()
print("DeBERTa-v3 loaded successfully!")

# Test dummy inference
text1 = ["Compulsory vaccination protects public health and saves lives."]
text2 = ["The government has a duty to vaccinate citizens to keep society safe."]

bge_in = bge_tok(text1, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    bge_out = bge_mod(**bge_in)[0][:, 0]
    bge_norm = torch.nn.functional.normalize(bge_out, p=2, dim=1)
print(f"BGE dummy embedding shape: {bge_norm.shape}")

deb_in = deb_tok(text1, text2, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    deb_out = torch.softmax(deb_mod(**deb_in).logits, dim=1).cpu().numpy()
print(f"DeBERTa dummy probabilities: contra={deb_out[0,0]:.4f}, entail={deb_out[0,1]:.4f}, neutral={deb_out[0,2]:.4f}")
print("ALL TESTS PASSED SUCCESSFULLY!")

