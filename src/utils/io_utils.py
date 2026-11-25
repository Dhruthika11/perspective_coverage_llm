"""Small shared helpers: config loading, JSON/CSV I/O, path resolution."""
from copy import deepcopy
import json
import os
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_ALIASES = {
    "gemma-2b": "google/gemma-2-2b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma-9b": "google/gemma-2-9b-it",
    "gemma2-9b": "google/gemma-2-9b-it",
    "gemma-27b": "google/gemma-2-27b-it",
    "gemma2-27b": "google/gemma-2-27b-it",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "olmoe-1b-7b": "allenai/OLMoE-1B-7B-0924-Instruct",
    # OLMo 2 1B
    "olmo2-1b-base": "allenai/OLMo-2-0425-1B",
    "olmo2-1b-sft": "allenai/OLMo-2-0425-1B-SFT",
    "olmo2-1b-dpo": "allenai/OLMo-2-0425-1B-DPO",
    "olmo2-1b-instruct": "allenai/OLMo-2-0425-1B-Instruct",
    # OLMo 2 7B
    "olmo2-7b-base": "allenai/OLMo-2-1124-7B",
    "olmo2-7b-sft": "allenai/OLMo-2-1124-7B-SFT",
    "olmo2-7b-dpo": "allenai/OLMo-2-1124-7B-DPO",
    "olmo2-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",
    # OLMo 2 13B
    "olmo2-13b-base": "allenai/OLMo-2-1124-13B",
    "olmo2-13b-sft": "allenai/OLMo-2-1124-13B-SFT",
    "olmo2-13b-dpo": "allenai/OLMo-2-1124-13B-DPO",
    "olmo2-13b-instruct": "allenai/OLMo-2-1124-13B-Instruct",
    # OLMo 2 32B
    "olmo2-32b-base": "allenai/OLMo-2-0325-32B",
    "olmo2-32b-sft": "allenai/OLMo-2-0325-32B-SFT",
    "olmo2-32b-dpo": "allenai/OLMo-2-0325-32B-DPO",
    "olmo2-32b-instruct": "allenai/OLMo-2-0325-32B-Instruct",
}


def resolve_model_alias(name: str) -> str:
    if not name:
        return name
    return MODEL_ALIASES.get(name.lower().strip(), name)


def get_model_tag(model_name: str) -> str:
    resolved = resolve_model_alias(model_name)
    return resolved.split("/")[-1].lower().replace(".", "_").replace("-", "_")


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    if overrides is not None:
        # Deep-copy: the caller's dict must not gain the derived
        # model-namespaced path keys as a side effect of loading.
        config = deepcopy(overrides)
    else:
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    model_name = config.get("llm", {}).get("model")
    if model_name:
        model_name = resolve_model_alias(model_name)
        config["llm"]["model"] = model_name
    else:
        raise ValueError(
            "Config has no llm.model set - cannot derive the model-namespaced "
            "output paths (results/<model_tag>/..., data/llm_outputs/<model_tag>/...)."
        )
    model_tag = get_model_tag(model_name)
    
    ds_name = config.get("dataset", {}).get("name", "perspectrum")
    ds_prefix = f"{ds_name}/" if ds_name != "perspectrum" else ""

    emb_model = config.get("embedding", {}).get("model", "nomic-ai/nomic-embed-text-v1.5")
    emb_tag = get_model_tag(emb_model)
    emb_suffix = "" if "bge_large" in emb_tag else f"_{emb_tag}"
    
    paths = config.get("paths", {})
    paths["llm_outputs_raw"] = f"data/llm_outputs/raw_generations/{ds_prefix}"
    paths["llm_outputs_processed"] = f"data/llm_outputs/{ds_prefix}{model_tag}/processed_generations.csv"
    paths["embeddings_dir"] = f"results/{ds_prefix}{model_tag}/embeddings{emb_suffix}/"
    paths["metrics_dir"] = f"results/{ds_prefix}{model_tag}/metrics{emb_suffix}/"
    paths["figures_dir"] = f"results/{ds_prefix}{model_tag}/figures{emb_suffix}/"
    paths["summary_file"] = f"results/{ds_prefix}{model_tag}/final_report_summary{emb_suffix}.json"
    config["paths"] = paths
    
    return config


def resolve(path: str) -> str:
    """Turn a repo-relative path from config.yaml into an absolute path."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
