"""
Generalized model accuracy tests.

Verifies that TransformerLens models produce outputs matching their
HuggingFace counterparts across all supported architectures.

Tests:
1. All weights loaded without unexpected issues (no all-zero non-bias params)
2. Forward pass logits match HuggingFace (compared after softmax)
3. Weight processing (fold_ln, etc.) preserves model behavior

Architecture families not covered here due to model size (smallest variant >2B):
- GPT-J (gptj.py) - smallest is EleutherAI/gpt-j-6B (6B)
- Mistral (mistral.py) - smallest is mistralai/Mistral-7B-v0.1 (7B)
- Mixtral (mixtral.py) - smallest is mistralai/Mixtral-8x7B-v0.1 (47B)
- Phi-3 (phi3.py) - smallest is microsoft/Phi-3-mini-4k-instruct (3.8B)

Gated models (require accepting license terms on HuggingFace) are included but
automatically skipped if access is denied.
"""

import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
import torch
from huggingface_hub.errors import GatedRepoError
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
# Gated models (e.g. Llama) are included but automatically skipped if access is denied.
EXTENDED_MODELS = DEFAULT_MODELS + [
    ("EleutherAI/gpt-neo-125M", "GPTNeoForCausalLM"),
    ("microsoft/phi-1", "PhiForCausalLM"),
    ("google/gemma-2b", "GemmaForCausalLM"),
    ("google/gemma-3-270m", "Gemma3ForCausalLM"),
    ("Qwen/Qwen3-0.6B", "Qwen3ForCausalLM"),
    ("meta-llama/Llama-3.2-1B", "LlamaForCausalLM"),
]

MODELS_TO_TEST = EXTENDED_MODELS if os.environ.get("HF_TOKEN", "") else DEFAULT_MODELS

# Extract just model names for parametrize
MODEL_NAMES = [m[0] for m in MODELS_TO_TEST]

# Prompt used for forward pass comparison
TEST_PROMPT = "The quick brown fox jumps over the lazy dog."


def _needs_remote_code(model_name: str) -> bool:
    """Some models require trust_remote_code=True."""
    return any(kw in model_name.lower() for kw in ["qwen", "santacoder"])


@dataclass
class ModelTestResults:
    """Pre-computed results from loading models, so models can be freed before tests run."""

    model_name: str
    # For test_weights_loaded: list of (name, numel, is_bias, all_zero) per parameter
    param_info: List[Tuple[str, int, bool, bool]] = field(default_factory=list)
    # For test_logits_match_huggingface: softmax probabilities
    tl_probs: Optional[torch.Tensor] = None
    hf_probs: Optional[torch.Tensor] = None
    # For test_processing_preserves_output: softmax probabilities
    raw_probs: Optional[torch.Tensor] = None
    processed_probs: Optional[torch.Tensor] = None


def _compute_results(model_name: str) -> ModelTestResults:
    """Load models, compute all needed tensors, then free the models.

    This ensures at most 2 full model copies exist at any time, and models
    are freed as soon as their outputs are captured.

    Raises pytest.skip if the model is gated and access is denied.
    """
    trust = _needs_remote_code(model_name)
    results = ModelTestResults(model_name=model_name)

    # Phase 1: Load raw TL model, capture weight info and logits
    try:
        raw_model = HookedTransformer.from_pretrained_no_processing(
            model_name, device="cpu", trust_remote_code=trust
        )
    except (GatedRepoError, OSError) as e:
        if "gated repo" in str(e).lower() or "Cannot access gated repo" in str(e):
            pytest.skip(f"Gated model {model_name} not accessible: {e}")
        raise
    tokens = raw_model.to_tokens(TEST_PROMPT, prepend_bos=True)

    for name, param in raw_model.named_parameters():
        is_bias = "b_" in name or name.endswith(".b")
        results.param_info.append(
            (name, param.shape.numel(), is_bias, bool(torch.all(param == 0)))
        )

    with torch.no_grad():
        raw_logits = raw_model(tokens, prepend_bos=False).float()
    results.tl_probs = torch.softmax(raw_logits, dim=-1)
    results.raw_probs = results.tl_probs  # same model, same logits

    del raw_model
    gc.collect()

    # Phase 2: Load HF model, compare logits, then free it
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cpu", trust_remote_code=trust
    )
    with torch.no_grad():
        hf_logits = hf_model(tokens).logits.float()
    results.hf_probs = torch.softmax(hf_logits, dim=-1)

    del hf_model
    gc.collect()

    # Phase 3: Load processed TL model, compare logits, then free it
    processed_model = HookedTransformer.from_pretrained(
        model_name, device="cpu", trust_remote_code=trust
    )
    with torch.no_grad():
        processed_logits = processed_model(tokens, prepend_bos=False).float()
    results.processed_probs = torch.softmax(processed_logits, dim=-1)

    del processed_model
    gc.collect()

    if "GITHUB_ACTIONS" in os.environ:
        clear_huggingface_cache()

    return results


class TestModelAccuracy:
    """Tests that TL models faithfully reproduce HuggingFace model outputs."""

    @pytest.fixture(scope="class", params=MODEL_NAMES)
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def results(self, model_name):
        return _compute_results(model_name)

    def test_weights_loaded(self, results):
        """Verify all parameters were loaded (no accidentally all-zero weight matrices)."""
        for name, numel, is_bias, all_zero in results.param_info:
            if numel == 0:
                pytest.fail(f"Empty parameter: {name}")
            if not is_bias and numel > 1 and all_zero:
                pytest.fail(f"All-zero weight matrix (likely not loaded): {name}")

    def test_logits_match_huggingface(self, results):
        """End-to-end logits should match HF model (compared after softmax)."""
        # Small differences arise from TL's custom hooked modules (e.g. LayerNorm reimplementations).
        # Gemma models can reach ~3.5e-4 due to embedding scaling and RMSNorm differences.
        assert torch.allclose(results.tl_probs, results.hf_probs, atol=5e-4), (
            f"Logit mismatch for {results.model_name}. "
            f"Max diff: {(results.tl_probs - results.hf_probs).abs().max().item():.2e}"
        )

    def test_processing_preserves_output(self, results):
        """Verify that fold_ln and other weight processing preserves model behavior."""
        # Looser tolerance: fold_ln introduces floating point differences
        assert torch.allclose(results.raw_probs, results.processed_probs, atol=1e-3), (
            f"Processing changed outputs too much for {results.model_name}. "
            f"Max diff: {(results.raw_probs - results.processed_probs).abs().max().item():.2e}"
        )
