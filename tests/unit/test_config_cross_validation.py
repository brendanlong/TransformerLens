"""
Config cross-validation tests: verify TransformerLens configs match HuggingFace.

These tests download only the HuggingFace config JSON (a few KB per model, no
weights) and cross-reference critical fields against TransformerLens's
convert_hf_model_config() output. They catch silent config drift -- the most
common source of bugs in model support (historically 6+ bugs).

NOT run in CI by default (requires network access to HuggingFace Hub for every
model). Use the config_cross_validation marker:

    poetry run pytest tests/unit/test_config_cross_validation.py -m config_cross_validation

Filter by model name:
    poetry run pytest tests/unit/test_config_cross_validation.py -m config_cross_validation -k "gemma"
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pytest
from transformers import AutoConfig

from transformer_lens.loading_from_pretrained import (
    NEED_REMOTE_CODE_MODELS,
    OFFICIAL_MODEL_NAMES,
    convert_hf_model_config,
)

logger = logging.getLogger(__name__)

# Models that use custom config loading (not AutoConfig compatible)
_CUSTOM_CONFIG_PREFIXES = ("NeelNanda/", "ArthurConmy/", "Baidicoot/")

# Models not hosted on HuggingFace with standard configs
_NON_HF_MODELS = {"llama-7b-hf", "llama-13b-hf", "llama-30b-hf", "llama-65b-hf"}

# Models requiring special AutoConfig classes (not AutoConfig.from_pretrained compatible)
_SPECIAL_ARCH_PREFIXES = ("facebook/hubert-", "facebook/wav2vec2-")


# Known intentional overrides where TL differs from HF config.
# Each entry documents WHY the override exists so reviewers can verify it's still valid.
# Format: {(model_prefix, field): (tl_value, hf_value, reason)}
_KNOWN_OVERRIDES: Dict[tuple, tuple] = {
    # Gemma v1: HF config says "gelu" but the actual HF implementation uses gelu with
    # tanh approximation. TL correctly uses "gelu_new" to match the real behavior.
    # See: https://github.com/TransformerLensOrg/TransformerLens/pull/596
    ("google/gemma-2b", "act_fn"): ("gelu_new", "gelu", "HF Gemma impl uses tanh approx despite config"),
    ("google/gemma-7b", "act_fn"): ("gelu_new", "gelu", "HF Gemma impl uses tanh approx despite config"),
}


def _is_known_override(model_name: str, field: str, tl_value: Any, hf_value: Any) -> Optional[str]:
    """Check if a mismatch is a known intentional override. Returns reason if so."""
    for (prefix, f), (expected_tl, expected_hf, reason) in _KNOWN_OVERRIDES.items():
        if model_name.startswith(prefix) and f == field:
            if tl_value == expected_tl and hf_value == expected_hf:
                return reason
    return None


def _is_cross_validatable(model_name: str) -> bool:
    """Check if we can cross-validate this model's config against HuggingFace."""
    if model_name in _NON_HF_MODELS:
        return False
    if any(model_name.startswith(p) for p in _CUSTOM_CONFIG_PREFIXES):
        return False
    if any(model_name.startswith(p) for p in _SPECIAL_ARCH_PREFIXES):
        return False
    return True


def _get_cross_validatable_models() -> List[str]:
    return [name for name in OFFICIAL_MODEL_NAMES if _is_cross_validatable(name)]


def _load_hf_config(model_name: str) -> AutoConfig:
    """Load HuggingFace config, handling gated repos and remote code."""
    huggingface_token = os.environ.get("HF_TOKEN", "")
    kwargs: Dict[str, Any] = {}
    if huggingface_token:
        kwargs["token"] = huggingface_token
    if model_name.startswith(NEED_REMOTE_CODE_MODELS):
        kwargs["trust_remote_code"] = True

    return AutoConfig.from_pretrained(model_name, **kwargs)


def _get_hf_field(hf_config: AutoConfig, *field_names: str) -> Optional[Any]:
    """Try multiple field names, return the first one found."""
    for name in field_names:
        val = getattr(hf_config, name, None)
        if val is not None:
            return val
    return None


# HF config field name for "hidden size" varies by architecture
_HF_HIDDEN_SIZE_FIELDS = ("hidden_size", "n_embd", "d_model")
_HF_NUM_HEADS_FIELDS = ("num_attention_heads", "n_head", "num_heads")
_HF_NUM_LAYERS_FIELDS = ("num_hidden_layers", "n_layer", "num_layers")
_HF_EPS_FIELDS = ("rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon")
_HF_ACT_FN_FIELDS = ("hidden_act", "activation_function", "feed_forward_proj")
_HF_N_CTX_FIELDS = ("max_position_embeddings", "n_ctx", "n_positions", "max_length")
_HF_VOCAB_SIZE_FIELDS = ("vocab_size",)
_HF_INTERMEDIATE_FIELDS = ("intermediate_size", "ffn_dim", "d_ff")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "config_cross_validation: config cross-validation tests (not run by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip config_cross_validation tests unless explicitly selected."""
    if "config_cross_validation" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(
        reason="config_cross_validation tests only run with -m config_cross_validation"
    )
    for item in items:
        if "config_cross_validation" in item.keywords:
            item.add_marker(skip)


@pytest.mark.config_cross_validation
class TestConfigCrossValidation:
    """Cross-validate TransformerLens configs against HuggingFace for all supported models."""

    @pytest.fixture(scope="class", params=_get_cross_validatable_models())
    def model_configs(self, request):
        """Load both TL and HF configs for a model."""
        model_name = request.param
        try:
            hf_config = _load_hf_config(model_name)
        except Exception as e:
            if "gated" in str(e).lower() or "access" in str(e).lower():
                pytest.skip(f"Gated model {model_name}: {e}")
            raise

        kwargs = {}
        if model_name.startswith(NEED_REMOTE_CODE_MODELS):
            kwargs["trust_remote_code"] = True
        tl_cfg = convert_hf_model_config(model_name, **kwargs)

        return model_name, tl_cfg, hf_config

    def test_d_model(self, model_configs):
        """d_model must match HuggingFace hidden_size."""
        model_name, tl_cfg, hf_config = model_configs
        hf_d_model = _get_hf_field(hf_config, *_HF_HIDDEN_SIZE_FIELDS)
        if hf_d_model is None:
            pytest.skip(f"No hidden_size field found in HF config for {model_name}")
        assert tl_cfg["d_model"] == hf_d_model, (
            f"{model_name}: d_model mismatch: TL={tl_cfg['d_model']}, HF={hf_d_model}"
        )

    def test_n_heads(self, model_configs):
        """n_heads must match HuggingFace num_attention_heads."""
        model_name, tl_cfg, hf_config = model_configs
        hf_n_heads = _get_hf_field(hf_config, *_HF_NUM_HEADS_FIELDS)
        if hf_n_heads is None:
            pytest.skip(f"No num_attention_heads field found in HF config for {model_name}")
        assert tl_cfg["n_heads"] == hf_n_heads, (
            f"{model_name}: n_heads mismatch: TL={tl_cfg['n_heads']}, HF={hf_n_heads}"
        )

    def test_n_layers(self, model_configs):
        """n_layers must match HuggingFace num_hidden_layers."""
        model_name, tl_cfg, hf_config = model_configs
        hf_n_layers = _get_hf_field(hf_config, *_HF_NUM_LAYERS_FIELDS)
        if hf_n_layers is None:
            pytest.skip(f"No num_hidden_layers field found in HF config for {model_name}")
        assert tl_cfg["n_layers"] == hf_n_layers, (
            f"{model_name}: n_layers mismatch: TL={tl_cfg['n_layers']}, HF={hf_n_layers}"
        )

    def test_d_vocab(self, model_configs):
        """d_vocab must match HuggingFace vocab_size."""
        model_name, tl_cfg, hf_config = model_configs
        hf_d_vocab = _get_hf_field(hf_config, *_HF_VOCAB_SIZE_FIELDS)
        if hf_d_vocab is None:
            pytest.skip(f"No vocab_size field found in HF config for {model_name}")
        # Some multimodal models (Gemma 3 4B+) have slightly different vocab sizes
        # between the text model and the full model config
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            hf_d_vocab = getattr(text_config, "vocab_size", hf_d_vocab)
        assert tl_cfg["d_vocab"] == hf_d_vocab, (
            f"{model_name}: d_vocab mismatch: TL={tl_cfg['d_vocab']}, HF={hf_d_vocab}"
        )

    def test_eps(self, model_configs):
        """Layer norm epsilon must match HuggingFace."""
        model_name, tl_cfg, hf_config = model_configs
        # Check text_config for multimodal models
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_eps = _get_hf_field(text_config, *_HF_EPS_FIELDS)
        if hf_eps is None:
            pytest.skip(f"No eps field found in HF config for {model_name}")
        assert abs(tl_cfg["eps"] - hf_eps) < 1e-10, (
            f"{model_name}: eps mismatch: TL={tl_cfg['eps']}, HF={hf_eps}"
        )

    def test_act_fn(self, model_configs):
        """Activation function must match HuggingFace."""
        model_name, tl_cfg, hf_config = model_configs
        # Check text_config for multimodal models
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_act_fn = _get_hf_field(text_config, *_HF_ACT_FN_FIELDS)
        if hf_act_fn is None:
            pytest.skip(f"No act_fn field found in HF config for {model_name}")

        tl_act = tl_cfg["act_fn"]

        # Some architectures intentionally override the HF act_fn
        # (e.g., BERT always uses "gelu" regardless of HF config,
        #  Bloom uses "gelu_fast" which isn't in HF config)
        arch = tl_cfg.get("original_architecture", "")
        if arch in ("BertForMaskedLM", "BloomForCausalLM"):
            pytest.skip(f"Architecture {arch} uses a hardcoded act_fn")

        # gelu_pytorch_tanh and gelu_new are functionally equivalent
        equivalent_groups = [
            {"gelu_new", "gelu_pytorch_tanh"},
        ]
        for group in equivalent_groups:
            if tl_act in group and hf_act_fn in group:
                return  # Match via equivalence

        # Check known intentional overrides
        reason = _is_known_override(model_name, "act_fn", tl_act, hf_act_fn)
        if reason is not None:
            return  # Known intentional override

        assert tl_act == hf_act_fn, (
            f"{model_name}: act_fn mismatch: TL={tl_act!r}, HF={hf_act_fn!r}"
        )

    def test_rotary_base(self, model_configs):
        """rotary_base (rope_theta) must match HuggingFace when applicable."""
        model_name, tl_cfg, hf_config = model_configs
        if tl_cfg.get("positional_embedding_type") != "rotary":
            pytest.skip("Not a rotary model")

        # Check text_config for multimodal models
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_rope_theta = getattr(text_config, "rope_theta", None)
        if hf_rope_theta is None:
            pytest.skip(f"No rope_theta in HF config for {model_name}")

        tl_rotary_base = tl_cfg.get("rotary_base")
        if tl_rotary_base is None:
            pytest.skip(f"No rotary_base in TL config for {model_name}")

        assert float(tl_rotary_base) == float(hf_rope_theta), (
            f"{model_name}: rotary_base mismatch: TL={tl_rotary_base}, HF={hf_rope_theta}"
        )

    def test_n_key_value_heads(self, model_configs):
        """n_key_value_heads must be consistent with HuggingFace for GQA models."""
        model_name, tl_cfg, hf_config = model_configs
        # Check text_config for multimodal models
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_kv_heads = getattr(text_config, "num_key_value_heads", None)
        if hf_kv_heads is None:
            pytest.skip(f"No num_key_value_heads in HF config for {model_name}")

        hf_n_heads = _get_hf_field(text_config, *_HF_NUM_HEADS_FIELDS)
        tl_kv_heads = tl_cfg.get("n_key_value_heads")

        # TL convention: n_key_value_heads is None when equal to n_heads (MHA)
        if hf_kv_heads == hf_n_heads:
            # MHA: TL should have None or same as n_heads
            if tl_kv_heads is not None:
                assert tl_kv_heads == hf_kv_heads, (
                    f"{model_name}: n_key_value_heads should be None or {hf_kv_heads} for MHA, "
                    f"got {tl_kv_heads}"
                )
        else:
            # GQA: TL must match HF
            assert tl_kv_heads == hf_kv_heads, (
                f"{model_name}: n_key_value_heads mismatch: TL={tl_kv_heads}, HF={hf_kv_heads}"
            )

    def test_d_mlp(self, model_configs):
        """d_mlp must match HuggingFace intermediate_size when available."""
        model_name, tl_cfg, hf_config = model_configs
        # Check text_config for multimodal models
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_d_mlp = _get_hf_field(text_config, *_HF_INTERMEDIATE_FIELDS)
        if hf_d_mlp is None:
            pytest.skip(f"No intermediate_size field found in HF config for {model_name}")

        tl_d_mlp = tl_cfg.get("d_mlp")
        if tl_d_mlp is None:
            pytest.skip(f"No d_mlp in TL config for {model_name}")

        # Some architectures compute d_mlp differently:
        # - GPT-2/Neo/J use hidden_size * 4 instead of intermediate_size
        # - Qwen v1 uses intermediate_size // 2
        # - Bloom uses hidden_size * 4
        arch = tl_cfg.get("original_architecture", "")
        if arch in (
            "GPT2LMHeadModel",
            "GPTNeoForCausalLM",
            "GPTJForCausalLM",
            "GPT2LMHeadCustomModel",
            "BloomForCausalLM",
        ):
            # These compute d_mlp as hidden_size * 4, not from intermediate_size
            hf_hidden = _get_hf_field(text_config, *_HF_HIDDEN_SIZE_FIELDS)
            if hf_hidden is not None:
                expected = hf_hidden * 4
                assert tl_d_mlp == expected, (
                    f"{model_name}: d_mlp mismatch: TL={tl_d_mlp}, expected={expected} (hidden*4)"
                )
            return
        if arch == "QWenLMHeadModel":
            # Qwen v1 halves intermediate_size for gated MLP
            assert tl_d_mlp == hf_d_mlp // 2, (
                f"{model_name}: d_mlp mismatch: TL={tl_d_mlp}, expected={hf_d_mlp // 2} (intermediate//2)"
            )
            return

        assert tl_d_mlp == hf_d_mlp, (
            f"{model_name}: d_mlp mismatch: TL={tl_d_mlp}, HF={hf_d_mlp}"
        )

    def test_no_none_in_required_fields(self, model_configs):
        """Required config fields must not be None."""
        model_name, tl_cfg, hf_config = model_configs
        required_fields = ["d_model", "n_heads", "n_layers", "d_vocab", "act_fn"]
        for field in required_fields:
            assert tl_cfg.get(field) is not None, (
                f"{model_name}: required field {field!r} is None"
            )

    def test_window_size_type(self, model_configs):
        """window_size must be int (not None) when sliding window attention is used."""
        model_name, tl_cfg, hf_config = model_configs
        if not tl_cfg.get("use_local_attn"):
            pytest.skip("Model doesn't use local attention")
        window_size = tl_cfg.get("window_size")
        assert window_size is not None and isinstance(window_size, int), (
            f"{model_name}: use_local_attn=True but window_size={window_size!r} "
            f"(expected int, not None)"
        )

    def test_n_ctx_not_larger_than_hf(self, model_configs):
        """n_ctx should not exceed the HuggingFace max_position_embeddings.

        TL intentionally caps n_ctx below the HF value for memory safety on many
        models, which is fine. But n_ctx should never be LARGER than the HF value
        (which would create oversized positional embeddings).
        """
        model_name, tl_cfg, hf_config = model_configs
        n_ctx = tl_cfg.get("n_ctx")
        if n_ctx is None:
            pytest.skip("No n_ctx in config")
        # ALiBi models (Bloom) don't use max_position_embeddings meaningfully
        if tl_cfg.get("positional_embedding_type") == "alibi":
            pytest.skip("ALiBi models don't use positional embeddings")
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_n_ctx = _get_hf_field(text_config, *_HF_N_CTX_FIELDS)
        if hf_n_ctx is None:
            pytest.skip(f"No max_position_embeddings in HF config for {model_name}")
        assert n_ctx <= hf_n_ctx, (
            f"{model_name}: n_ctx={n_ctx} exceeds HF max_position_embeddings={hf_n_ctx}. "
            f"This could cause oversized positional embeddings."
        )

    def test_d_head_consistent(self, model_configs):
        """d_head should be consistent with d_model and n_heads."""
        model_name, tl_cfg, hf_config = model_configs
        d_model = tl_cfg["d_model"]
        n_heads = tl_cfg["n_heads"]
        d_head = tl_cfg.get("d_head")
        if d_head is None:
            pytest.skip("No d_head in config")

        # Check text_config for multimodal models that might have explicit head_dim
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_head_dim = getattr(text_config, "head_dim", None)

        if hf_head_dim is not None:
            # If HF specifies head_dim explicitly, TL should match
            assert d_head == hf_head_dim, (
                f"{model_name}: d_head mismatch: TL={d_head}, HF head_dim={hf_head_dim}"
            )
        else:
            # Otherwise d_head should be d_model // n_heads
            expected = d_model // n_heads
            assert d_head == expected, (
                f"{model_name}: d_head={d_head} != d_model//n_heads={expected}"
            )

    def test_attn_types_length_matches_n_layers(self, model_configs):
        """If attn_types is specified, its length must equal n_layers."""
        model_name, tl_cfg, hf_config = model_configs
        attn_types = tl_cfg.get("attn_types")
        if attn_types is None:
            pytest.skip("No attn_types in config")
        n_layers = tl_cfg["n_layers"]
        assert len(attn_types) == n_layers, (
            f"{model_name}: attn_types length ({len(attn_types)}) != n_layers ({n_layers})"
        )
