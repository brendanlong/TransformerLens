"""
Generalized model accuracy tests.

Verifies that TransformerLens models produce outputs matching their
HuggingFace counterparts across all supported architectures.

Tests:
1. All weights loaded without unexpected issues (no all-zero non-bias params)
2. Forward pass logits match HuggingFace (compared after softmax)
3. Weight processing (fold_ln, etc.) preserves model behavior
"""

import gc
import os

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer
from transformer_lens.utils import clear_huggingface_cache

# Models already downloaded by other tests in the default suite (no HF_TOKEN needed).
# One per architecture family, picking the smallest available.
DEFAULT_MODELS = [
    ("gpt2", "GPT2LMHeadModel"),
    ("EleutherAI/pythia-70m", "GPTNeoXForCausalLM"),
    ("facebook/opt-125m", "OPTForCausalLM"),
    ("bigscience/bloom-560m", "BloomForCausalLM"),
    ("Qwen/Qwen2-0.5B", "Qwen2ForCausalLM"),
]

# Additional models for full architecture coverage (require HF_TOKEN).
# Note: google/gemma-2b (GemmaForCausalLM) is excluded because loading both TL and HF
# copies simultaneously requires too much memory for typical CI runners.
EXTENDED_MODELS = DEFAULT_MODELS + [
    ("EleutherAI/gpt-neo-125M", "GPTNeoForCausalLM"),
    ("microsoft/phi-1", "PhiForCausalLM"),
]

MODELS_TO_TEST = EXTENDED_MODELS if os.environ.get("HF_TOKEN", "") else DEFAULT_MODELS

# Extract just model names for parametrize
MODEL_NAMES = [m[0] for m in MODELS_TO_TEST]

# Prompt used for forward pass comparison
TEST_PROMPT = "The quick brown fox jumps over the lazy dog."


def _needs_remote_code(model_name: str) -> bool:
    """Some models require trust_remote_code=True."""
    return any(
        kw in model_name.lower() for kw in ["qwen", "phi-3", "phi-4", "santacoder"]
    )


def _load_models(model_name: str):
    """Load both TL and HF versions of a model. Returns (tl_model, hf_model)."""
    trust = _needs_remote_code(model_name)

    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name, device="cpu", trust_remote_code=trust
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cpu", trust_remote_code=trust
    )
    return tl_model, hf_model


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_weights_loaded(model_name):
    """Verify all parameters were loaded (no accidentally all-zero weight matrices)."""
    trust = _needs_remote_code(model_name)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name, device="cpu", trust_remote_code=trust
    )

    for name, param in tl_model.named_parameters():
        if param.shape.numel() == 0:
            pytest.fail(f"Empty parameter: {name}")

        # Bias terms and small params can legitimately be all zeros
        is_bias = "b_" in name or name.endswith(".b")
        if not is_bias and param.shape.numel() > 1:
            # Weight matrices should not be all zeros (would indicate failed loading)
            if torch.all(param == 0):
                pytest.fail(f"All-zero weight matrix (likely not loaded): {name}")

    del tl_model
    gc.collect()
    if "GITHUB_ACTIONS" in os.environ:
        clear_huggingface_cache()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_logits_match_huggingface(model_name):
    """End-to-end logits should match HF model (compared after softmax)."""
    tl_model, hf_model = _load_models(model_name)

    tokens = tl_model.to_tokens(TEST_PROMPT, prepend_bos=True)

    with torch.no_grad():
        tl_logits = tl_model(tokens, prepend_bos=False).float()
        hf_logits = hf_model(tokens).logits.float()

    tl_probs = torch.softmax(tl_logits, dim=-1)
    hf_probs = torch.softmax(hf_logits, dim=-1)

    # Small differences arise from TL's custom hooked modules (e.g. LayerNorm reimplementations).
    # GPTNeoX models (pythia) can reach ~1.3e-4 due to parallel attention decomposition.
    assert torch.allclose(tl_probs, hf_probs, atol=2e-4), (
        f"Logit mismatch for {model_name}. "
        f"Max diff: {(tl_probs - hf_probs).abs().max().item():.2e}"
    )

    del tl_model, hf_model
    gc.collect()
    if "GITHUB_ACTIONS" in os.environ:
        clear_huggingface_cache()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_processing_preserves_output(model_name):
    """Verify that fold_ln and other weight processing preserves model behavior."""
    trust = _needs_remote_code(model_name)

    raw_model = HookedTransformer.from_pretrained_no_processing(
        model_name, device="cpu", trust_remote_code=trust
    )
    processed_model = HookedTransformer.from_pretrained(
        model_name, device="cpu", trust_remote_code=trust
    )

    tokens = raw_model.to_tokens(TEST_PROMPT, prepend_bos=True)

    with torch.no_grad():
        raw_logits = raw_model(tokens, prepend_bos=False).float()
        processed_logits = processed_model(tokens, prepend_bos=False).float()

    raw_probs = torch.softmax(raw_logits, dim=-1)
    processed_probs = torch.softmax(processed_logits, dim=-1)

    # Looser tolerance: fold_ln introduces floating point differences
    assert torch.allclose(raw_probs, processed_probs, atol=1e-3), (
        f"Processing changed outputs too much for {model_name}. "
        f"Max diff: {(raw_probs - processed_probs).abs().max().item():.2e}"
    )

    del raw_model, processed_model
    gc.collect()
    if "GITHUB_ACTIONS" in os.environ:
        clear_huggingface_cache()
