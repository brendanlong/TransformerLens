"""Gemma4 architecture adapter.

Supports the Gemma 4 text model family (Gemma4ForCausalLM), including:
- Dense models (e.g. 31B)
- MoE models (e.g. 26B-A4B with 128 experts, top-8 routing)
- PLE models (e.g. E2B/E4B with Per-Layer Embeddings)

Key differences from the Gemma 3 adapter:
- Per-layer attention parameters: sliding layers use head_dim/num_key_value_heads,
  global layers use global_head_dim/num_global_key_value_heads.
- K=V on global layers: when attention_k_eq_v is True, v_proj is None for global
  (full_attention) layers and value_states reuse key_states. The bridge omits the
  V linear for those layers and lets HF handle the K=V logic natively.
- v_norm: new RMSNorm on value states (with_scale=False, so no learnable weight).
- Activation changed from silu (Gemma 3) to gelu_pytorch_tanh.
- RMSNorm weights are stored directly (no +1 offset unlike Gemma 3).
- layer_scalar buffer multiplies each layer's output (handled natively by HF).
- MoE: when enable_moe_block is True, each layer has a router + experts alongside
  the standard MLP. The MoE output is combined with the MLP output. HF handles
  the routing logic natively; the bridge exposes router/experts as submodules.
- PLE: when hidden_size_per_layer_input > 0, each layer has a per-layer embedding
  gate and projection. Model-level per-layer embedding and projection components
  are also exposed for hook access.
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    RearrangeTensorConversion,
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


class Gemma4ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma 4 text models (Gemma4ForCausalLM).

    Handles both dense and MoE variants:
    - 5:1 sliding/full attention pattern with per-layer head dimensions
    - K=V parameter sharing on global layers
    - v_norm on value states
    - Optional MoE (router + experts alongside standard MLP)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma4 architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        # Gemma 4 RMSNorm stores weight directly (initialized to ones),
        # unlike Gemma 3 which stores (weight - 1) and applies (1 + weight).
        self.cfg.rmsnorm_uses_offset = False
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"

        # Read per-layer attention config from the HF config that was
        # propagated onto the bridge config during boot.
        layer_types = getattr(self.cfg, "layer_types", None) or []
        attention_k_eq_v = getattr(self.cfg, "attention_k_eq_v", False)
        global_head_dim = getattr(self.cfg, "global_head_dim", None)
        head_dim = getattr(self.cfg, "d_head", 256)
        n_heads = self.cfg.n_heads
        n_kv_heads = getattr(self.cfg, "n_key_value_heads", n_heads)
        n_global_kv_heads = getattr(self.cfg, "num_global_key_value_heads", n_kv_heads)
        enable_moe = getattr(self.cfg, "enable_moe_block", False)
        ple_dim = getattr(self.cfg, "hidden_size_per_layer_input", 0) or 0

        # Build per-layer weight conversions. Gemma 4 has different head_dim and
        # n_kv_heads for sliding vs full attention layers, so we generate
        # layer-specific conversion rules.
        self.weight_processing_conversions = {}

        for i, layer_type in enumerate(layer_types):
            is_global = layer_type == "full_attention"
            use_k_eq_v = attention_k_eq_v and is_global
            layer_hd = global_head_dim if (is_global and global_head_dim) else head_dim
            layer_n_kv = n_global_kv_heads if use_k_eq_v else n_kv_heads

            # Q weight: (n_heads * layer_hd, d_model) -> (n_heads, d_model, layer_hd)
            self.weight_processing_conversions[
                f"blocks.{i}.attn.q.weight"
            ] = ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_heads),
            )
            # K weight
            self.weight_processing_conversions[
                f"blocks.{i}.attn.k.weight"
            ] = ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=layer_n_kv),
            )
            # V weight (only if v_proj exists for this layer)
            if not use_k_eq_v:
                self.weight_processing_conversions[
                    f"blocks.{i}.attn.v.weight"
                ] = ParamProcessingConversion(
                    tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=layer_n_kv),
                )
            # O weight: (d_model, n_heads * layer_hd) -> (n_heads, layer_hd, d_model)
            self.weight_processing_conversions[
                f"blocks.{i}.attn.o.weight"
            ] = ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=n_heads),
            )

        # MLP weight conversions (uniform across layers for dense model)
        n_layers = self.cfg.n_layers
        for i in range(n_layers):
            self.weight_processing_conversions[
                f"blocks.{i}.mlp.gate.weight"
            ] = ParamProcessingConversion(tensor_conversion=TransposeTensorConversion())
            self.weight_processing_conversions[
                f"blocks.{i}.mlp.in.weight"
            ] = ParamProcessingConversion(tensor_conversion=TransposeTensorConversion())
            self.weight_processing_conversions[
                f"blocks.{i}.mlp.out.weight"
            ] = ParamProcessingConversion(tensor_conversion=TransposeTensorConversion())

        # PLE weight conversions (per-layer gate and projection are Linear layers)
        if ple_dim > 0:
            for i in range(n_layers):
                self.weight_processing_conversions[
                    f"blocks.{i}.ple_gate.weight"
                ] = ParamProcessingConversion(tensor_conversion=TransposeTensorConversion())
                self.weight_processing_conversions[
                    f"blocks.{i}.ple_proj.weight"
                ] = ParamProcessingConversion(tensor_conversion=TransposeTensorConversion())
            # Model-level PLE projection
            self.weight_processing_conversions["ple_model_proj.weight"] = ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion()
            )

        # Unembed weight conversion
        self.weight_processing_conversions["unembed.weight"] = ParamProcessingConversion(
            tensor_conversion=TransposeTensorConversion(),
        )

        # Build attention submodules. For K=V layers, we still include v as a
        # LinearBridge pointing to v_proj — when v_proj is None on the HF side,
        # the bridge simply won't find a component to wrap, which is fine since
        # HF's forward pass handles the K=V reuse natively.
        attn_submodules = {
            "q": LinearBridge(name="q_proj"),
            "k": LinearBridge(name="k_proj"),
            "v": LinearBridge(name="v_proj"),
            "o": LinearBridge(name="o_proj"),
            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
            "v_norm": RMSNormalizationBridge(name="v_norm", config=self.cfg),
        }

        # Build block submodules
        block_submodules = {
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln1_post": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="pre_feedforward_layernorm", config=self.cfg),
            "ln2_post": RMSNormalizationBridge(name="post_feedforward_layernorm", config=self.cfg),
            "attn": PositionEmbeddingsAttentionBridge(
                name="self_attn",
                config=self.cfg,
                submodules=attn_submodules,
            ),
            "mlp": GatedMLPBridge(
                name="mlp",
                config=self.cfg,
                submodules={
                    "gate": LinearBridge(name="gate_proj"),
                    "in": LinearBridge(name="up_proj"),
                    "out": LinearBridge(name="down_proj"),
                },
            ),
        }

        # PLE support: when hidden_size_per_layer_input > 0, each decoder layer
        # has a per-layer embedding gate, projection, and norm. HF handles the
        # PLE computation natively; the bridge exposes components for hook access.
        ple_dim = getattr(self.cfg, "hidden_size_per_layer_input", 0) or 0
        if ple_dim > 0:
            block_submodules["ple_gate"] = LinearBridge(name="per_layer_input_gate")
            block_submodules["ple_proj"] = LinearBridge(name="per_layer_projection")
            block_submodules["ple_norm"] = RMSNormalizationBridge(
                name="post_per_layer_input_norm", config=self.cfg
            )

        # MoE support: when enable_moe_block is True, each decoder layer has a
        # router + experts alongside the standard MLP. The MoE output is combined
        # with the MLP output inside HF's forward pass.
        #
        # In Gemma 4, the router and experts are siblings in the decoder layer
        # (unlike Mixtral where they're nested inside a single MoE module).
        # We map them as separate block submodules for hook access:
        # - moe_router: the router's projection (for inspecting expert selection)
        # - moe: the experts module (wrapped by MoEBridge for hook_router_scores)
        if enable_moe:
            block_submodules["moe_router"] = LinearBridge(name="router.proj")
            block_submodules["moe"] = MoEBridge(
                name="experts",
                config=self.cfg,
            )
            # Extra norms for MoE pathway
            block_submodules["ln2_post_mlp"] = RMSNormalizationBridge(
                name="post_feedforward_layernorm_1", config=self.cfg
            )
            block_submodules["ln2_pre_moe"] = RMSNormalizationBridge(
                name="pre_feedforward_layernorm_2", config=self.cfg
            )
            block_submodules["ln2_post_moe"] = RMSNormalizationBridge(
                name="post_feedforward_layernorm_2", config=self.cfg
            )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules=block_submodules,
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

        # PLE model-level components: per-layer embedding table, projection,
        # and projection norm. These are children of Gemma4TextModel (model.*).
        if ple_dim > 0:
            self.component_mapping["ple_embed"] = EmbeddingBridge(
                name="model.embed_tokens_per_layer"
            )
            self.component_mapping["ple_model_proj"] = LinearBridge(
                name="model.per_layer_model_projection"
            )
            self.component_mapping["ple_model_proj_norm"] = RMSNormalizationBridge(
                name="model.per_layer_projection_norm", config=self.cfg
            )

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Setup hook compatibility for Gemma4 models.

        Like Gemma 3, Gemma 4 uses Gemma4TextScaledWordEmbedding which scales
        embeddings by sqrt(hidden_size) inside the embedding layer's forward().
        No additional hook conversion is needed.

        Args:
            bridge: The TransformerBridge instance
        """
        pass

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Gemma 4 component testing.

        Gemma 4 uses multi-type RoPE (different parameters for sliding vs full
        attention). We set the rotary_emb reference on all attention bridge
        instances for component testing.

        Args:
            hf_model: The HuggingFace Gemma 4 model instance
            bridge_model: The TransformerBridge model (if available)
        """
        rotary_emb = hf_model.model.rotary_emb

        # Force eager attention for numerical parity with bridge
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

                    # Enable native autograd for norms to match HF exactly
                    if hasattr(block.attn, "original_component"):
                        hf_attn = block.attn.original_component
                        for norm_name in ("q_norm", "k_norm", "v_norm"):
                            if hasattr(hf_attn, norm_name):
                                norm = getattr(hf_attn, norm_name)
                                if hasattr(norm, "use_native_layernorm_autograd"):
                                    norm.use_native_layernorm_autograd = True

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
