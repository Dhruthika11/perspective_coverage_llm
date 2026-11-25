"""Unit tests for io_utils.load_config (no GPU/model needed).

Run: python -m pytest tests/ -q   (from the project root)

These lock in two load_config contracts: callers' dicts are never mutated
in place, and a config without llm.model fails loudly instead of silently
deriving paths under a "default_model" tag.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.io_utils import load_config


def _base_cfg():
    return {
        "llm": {"model": "org/some-model", "provider": "huggingface"},
        "paths": {"metrics_dir": "results/metrics/"},
    }


def test_load_config_does_not_mutate_overrides():
    cfg = _base_cfg()
    snapshot = {"llm": dict(cfg["llm"]), "paths": dict(cfg["paths"])}
    loaded = load_config(overrides=cfg)
    # caller's nested dicts untouched...
    assert cfg == snapshot
    # ...while the returned config carries the derived paths.
    assert loaded["paths"]["metrics_dir"] != snapshot["paths"]["metrics_dir"]
    assert "some_model" in loaded["paths"]["metrics_dir"]


def test_load_config_missing_model_raises():
    with pytest.raises(ValueError, match="llm.model"):
        load_config(overrides={"llm": {"provider": "groq"}, "paths": {}})
    with pytest.raises(ValueError, match="llm.model"):
        load_config(overrides={"paths": {}})


def test_load_config_dataset_namespacing():
    # Perspectrum: ds_prefix should be empty
    cfg_persp = _base_cfg()
    cfg_persp["dataset"] = {"name": "perspectrum"}
    loaded_persp = load_config(overrides=cfg_persp)
    assert loaded_persp["paths"]["llm_outputs_raw"] == "data/llm_outputs/raw_generations/"

    # ibm_debater: ds_prefix should be ibm_debater/
    cfg_ibm = _base_cfg()
    cfg_ibm["dataset"] = {"name": "ibm_debater"}
    loaded_ibm = load_config(overrides=cfg_ibm)
    assert loaded_ibm["paths"]["llm_outputs_raw"] == "data/llm_outputs/raw_generations/ibm_debater/"
    assert "ibm_debater" in loaded_ibm["paths"]["llm_outputs_processed"]

    # kialo: ds_prefix should be kialo/
    cfg_kialo = _base_cfg()
    cfg_kialo["dataset"] = {"name": "kialo"}
    loaded_kialo = load_config(overrides=cfg_kialo)
    assert loaded_kialo["paths"]["llm_outputs_raw"] == "data/llm_outputs/raw_generations/kialo/"
    assert "kialo" in loaded_kialo["paths"]["llm_outputs_processed"]

