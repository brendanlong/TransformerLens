"""
Unit tests for the Gemma 4 architecture adapter.

Tests cover:
1. Adapter instantiation and configuration
2. Per-layer weight conversion correctness (sliding vs full attention)
3. K=V handling (v_proj omission on global layers)
4. Component mapping structure
5. Path translation from TransformerLens to HuggingFace
6. Factory registration and model_type mapping
"""

import pytest

from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.supported_architectures.gemma4 import (
    Gemma4ArchitectureAdapter,
)

# ============================================================================
# Fixtures
# ============================================================================


def _make_layer_types(n_layers, pattern=6):
    """Generate 5:1 sliding/full attention pattern."""
    return [
        "sliding_attention" if (i + 1) % pattern != 0 else "full_attention" for i in range(n_layers)
    ]


def _make_cfg(
    n_layers=30,
    n_heads=8,
    d_head=256,
    n_kv_heads=4,
    d_model=2304,
    d_mlp=9216,
    attention_k_eq_v=False,
    global_head_dim=512,
    num_global_kv_heads=None,
    **kwargs,
):
    """Create a TransformerBridgeConfig with Gemma 4-specific attributes."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=8192,
        d_vocab=262144,
        d_mlp=d_mlp,
        n_key_value_heads=n_kv_heads,
        architecture="Gemma4ForCausalLM",
    )
    # Set Gemma 4-specific attributes that are propagated during boot
    cfg.layer_types = _make_layer_types(n_layers)
    cfg.attention_k_eq_v = attention_k_eq_v
    cfg.global_head_dim = global_head_dim
    cfg.num_global_key_value_heads = num_global_kv_heads
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def dense_cfg():
    """Config for a Gemma 4 dense model (no K=V)."""
    return _make_cfg()


@pytest.fixture
def k_eq_v_cfg():
    """Config for a Gemma 4 model with K=V enabled (like the 31B)."""
    return _make_cfg(
        n_layers=48,
        n_heads=16,
        d_head=256,
        n_kv_heads=16,
        d_model=3840,
        d_mlp=15360,
        attention_k_eq_v=True,
        global_head_dim=512,
        num_global_kv_heads=4,
    )


@pytest.fixture
def dense_adapter(dense_cfg):
    return Gemma4ArchitectureAdapter(dense_cfg)


@pytest.fixture
def k_eq_v_adapter(k_eq_v_cfg):
    return Gemma4ArchitectureAdapter(k_eq_v_cfg)


# ============================================================================
# Test: Factory registration
# ============================================================================


class TestGemma4Registration:
    """Test that Gemma4 is properly registered in the architecture adapter factory."""

    def test_gemma4_in_supported_architectures(self):
        assert "Gemma4ForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_creates_gemma4_adapter(self, dense_cfg):
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(dense_cfg)
        assert isinstance(adapter, Gemma4ArchitectureAdapter)


# ============================================================================
# Test: Config properties
# ============================================================================


class TestGemma4ConfigProperties:
    """Test that the adapter sets correct config properties."""

    def test_normalization_type(self, dense_adapter):
        assert dense_adapter.cfg.normalization_type == "RMS"

    def test_uses_rms_norm(self, dense_adapter):
        assert dense_adapter.cfg.uses_rms_norm is True

    def test_no_rmsnorm_offset(self, dense_adapter):
        """Gemma 4 stores RMSNorm weights directly (no +1 offset like Gemma 3)."""
        assert dense_adapter.cfg.rmsnorm_uses_offset is False

    def test_positional_embedding_type(self, dense_adapter):
        assert dense_adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp(self, dense_adapter):
        assert dense_adapter.cfg.gated_mlp is True

    def test_eager_attention(self, dense_adapter):
        assert dense_adapter.cfg.attn_implementation == "eager"


# ============================================================================
# Test: Component mapping
# ============================================================================


class TestGemma4ComponentMapping:
    """Test the component mapping structure."""

    def test_top_level_components(self, dense_adapter):
        keys = list(dense_adapter.component_mapping.keys())
        assert "embed" in keys
        assert "rotary_emb" in keys
        assert "blocks" in keys
        assert "ln_final" in keys
        assert "unembed" in keys

    def test_block_submodules(self, dense_adapter):
        blocks = dense_adapter.component_mapping["blocks"]
        submodule_keys = list(blocks.submodules.keys())
        assert "ln1" in submodule_keys
        assert "ln1_post" in submodule_keys
        assert "ln2" in submodule_keys
        assert "ln2_post" in submodule_keys
        assert "attn" in submodule_keys
        assert "mlp" in submodule_keys

    def test_attention_submodules(self, dense_adapter):
        attn = dense_adapter.component_mapping["blocks"].submodules["attn"]
        submodule_keys = list(attn.submodules.keys())
        assert "q" in submodule_keys
        assert "k" in submodule_keys
        assert "v" in submodule_keys
        assert "o" in submodule_keys
        assert "q_norm" in submodule_keys
        assert "k_norm" in submodule_keys
        assert "v_norm" in submodule_keys

    def test_mlp_submodules(self, dense_adapter):
        mlp = dense_adapter.component_mapping["blocks"].submodules["mlp"]
        submodule_keys = list(mlp.submodules.keys())
        assert "gate" in submodule_keys
        assert "in" in submodule_keys
        assert "out" in submodule_keys


# ============================================================================
# Test: Path translation
# ============================================================================


class TestGemma4PathTranslation:
    """Test path translation from TransformerLens to HuggingFace naming."""

    def test_embed_path(self, dense_adapter):
        assert dense_adapter.translate_transformer_lens_path("embed") == "model.embed_tokens"

    def test_unembed_path(self, dense_adapter):
        assert dense_adapter.translate_transformer_lens_path("unembed") == "lm_head"

    def test_ln_final_path(self, dense_adapter):
        assert dense_adapter.translate_transformer_lens_path("ln_final") == "model.norm"

    def test_block_attn_q_path(self, dense_adapter):
        assert (
            dense_adapter.translate_transformer_lens_path("blocks.0.attn.q")
            == "model.layers.0.self_attn.q_proj"
        )

    def test_block_attn_v_norm_path(self, dense_adapter):
        assert (
            dense_adapter.translate_transformer_lens_path("blocks.0.attn.v_norm")
            == "model.layers.0.self_attn.v_norm"
        )

    def test_block_mlp_gate_path(self, dense_adapter):
        assert (
            dense_adapter.translate_transformer_lens_path("blocks.0.mlp.gate")
            == "model.layers.0.mlp.gate_proj"
        )

    def test_block_ln1_path(self, dense_adapter):
        assert (
            dense_adapter.translate_transformer_lens_path("blocks.0.ln1")
            == "model.layers.0.input_layernorm"
        )

    def test_block_ln1_post_path(self, dense_adapter):
        assert (
            dense_adapter.translate_transformer_lens_path("blocks.0.ln1_post")
            == "model.layers.0.post_attention_layernorm"
        )


# ============================================================================
# Test: Weight conversions - sliding layers
# ============================================================================


class TestGemma4SlidingLayerConversions:
    """Test weight conversions for sliding attention layers."""

    def test_q_weight_conversion_exists(self, dense_adapter):
        assert "blocks.0.attn.q.weight" in dense_adapter.weight_processing_conversions

    def test_k_weight_conversion_exists(self, dense_adapter):
        assert "blocks.0.attn.k.weight" in dense_adapter.weight_processing_conversions

    def test_v_weight_conversion_exists(self, dense_adapter):
        assert "blocks.0.attn.v.weight" in dense_adapter.weight_processing_conversions

    def test_o_weight_conversion_exists(self, dense_adapter):
        assert "blocks.0.attn.o.weight" in dense_adapter.weight_processing_conversions

    def test_mlp_gate_conversion_exists(self, dense_adapter):
        assert "blocks.0.mlp.gate.weight" in dense_adapter.weight_processing_conversions

    def test_mlp_in_conversion_exists(self, dense_adapter):
        assert "blocks.0.mlp.in.weight" in dense_adapter.weight_processing_conversions

    def test_mlp_out_conversion_exists(self, dense_adapter):
        assert "blocks.0.mlp.out.weight" in dense_adapter.weight_processing_conversions


# ============================================================================
# Test: Weight conversions - global layers
# ============================================================================


class TestGemma4GlobalLayerConversions:
    """Test weight conversions for full attention (global) layers."""

    def test_global_layer_q_weight_exists(self, dense_adapter):
        """Layer 5 is the first global layer (5:1 pattern)."""
        assert "blocks.5.attn.q.weight" in dense_adapter.weight_processing_conversions

    def test_global_layer_k_weight_exists(self, dense_adapter):
        assert "blocks.5.attn.k.weight" in dense_adapter.weight_processing_conversions

    def test_global_layer_v_weight_exists_no_k_eq_v(self, dense_adapter):
        """Without K=V, global layers still have V weight conversions."""
        assert "blocks.5.attn.v.weight" in dense_adapter.weight_processing_conversions

    def test_global_layer_o_weight_exists(self, dense_adapter):
        assert "blocks.5.attn.o.weight" in dense_adapter.weight_processing_conversions


# ============================================================================
# Test: K=V handling
# ============================================================================


class TestGemma4KEqualsV:
    """Test K=V behavior when attention_k_eq_v is True."""

    def test_sliding_layer_has_v_weight(self, k_eq_v_adapter):
        """Sliding layers always have V weights, even with K=V enabled."""
        assert "blocks.0.attn.v.weight" in k_eq_v_adapter.weight_processing_conversions

    def test_global_layer_no_v_weight(self, k_eq_v_adapter):
        """Global layers omit V weight when K=V is True."""
        assert "blocks.5.attn.v.weight" not in k_eq_v_adapter.weight_processing_conversions

    def test_global_layer_has_k_weight(self, k_eq_v_adapter):
        """Global layers still have K weight (which doubles as V)."""
        assert "blocks.5.attn.k.weight" in k_eq_v_adapter.weight_processing_conversions

    def test_all_global_layers_skip_v(self, k_eq_v_adapter):
        """All global layers should skip V weight conversion."""
        layer_types = _make_layer_types(48)
        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                assert (
                    f"blocks.{i}.attn.v.weight" not in k_eq_v_adapter.weight_processing_conversions
                ), f"Global layer {i} should not have V weight conversion"

    def test_all_sliding_layers_have_v(self, k_eq_v_adapter):
        """All sliding layers should have V weight conversion."""
        layer_types = _make_layer_types(48)
        for i, lt in enumerate(layer_types):
            if lt == "sliding_attention":
                assert (
                    f"blocks.{i}.attn.v.weight" in k_eq_v_adapter.weight_processing_conversions
                ), f"Sliding layer {i} should have V weight conversion"


# ============================================================================
# Test: Unembed conversion
# ============================================================================


class TestGemma4UnembedConversion:
    """Test unembed weight conversion."""

    def test_unembed_weight_conversion_exists(self, dense_adapter):
        assert "unembed.weight" in dense_adapter.weight_processing_conversions


# ============================================================================
# Test: No RMSNorm offset conversions
# ============================================================================


class TestGemma4NoRMSNormOffset:
    """Verify that Gemma 4 does NOT add +1 to RMSNorm weights (unlike Gemma 3)."""

    def test_no_ln1_weight_conversion(self, dense_adapter):
        """ln1 weight should NOT have a conversion (no +1 offset needed)."""
        for key in dense_adapter.weight_processing_conversions:
            assert "ln1.weight" not in key, f"Unexpected ln1 weight conversion: {key}"

    def test_no_ln_final_weight_conversion(self, dense_adapter):
        assert "ln_final.weight" not in dense_adapter.weight_processing_conversions


# ============================================================================
# Test: Layer type pattern
# ============================================================================


class TestGemma4LayerTypePattern:
    """Test the 5:1 sliding/full attention pattern helper."""

    def test_30_layer_pattern(self):
        types = _make_layer_types(30)
        assert len(types) == 30
        global_indices = [i for i, t in enumerate(types) if t == "full_attention"]
        assert global_indices == [5, 11, 17, 23, 29]

    def test_48_layer_pattern(self):
        types = _make_layer_types(48)
        assert len(types) == 48
        global_count = types.count("full_attention")
        assert global_count == 8
