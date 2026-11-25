import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()
print(f"Loaded {model_id} in {time.time() - t0:.2f}s")

prompt = """Extract the Toulmin argument components from this viewpoint.
Viewpoint: "Implementing the death penalty for terrorist acts that result in death can lead to significant cost savings in the criminal justice system. By removing convicted terrorists from the population permanently, it eliminates the ongoing costs associated with incarceration, such as housing, healthcare, and security."

Output JSON with keys:
"claim": The central proposition or stance.
"reasoning": The evidence, causal mechanism, or grounds provided.
"warrant": The underlying general principle or assumption connecting reasoning to claim.
JSON:"""

messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")

t1 = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generation took {time.time() - t1:.2f}s")
print("Generated output:")
print(gen_text)
