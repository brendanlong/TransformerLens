# Gemma 4 Support Implementation Plan (dev-3.x branch)

## Overview

This plan targets the `dev-3.x` branch which uses the TransformerBridge architecture.
HuggingFace Transformers already supports Gemma 4 natively (v5.5+), so the bridge
approach can delegate most complexity to HF's implementation.

### Gemma 4 Variants and Their Key Features

| Feature | 31B (Dense) | 26B-A4B (MoE) | E2B / E4B |
|---|---|---|---|
| Per-layer attention type | 5:1 sliding/full | 5:1 sliding/full | 4:1 (E2B) / 5:1 (E4B) |
| GQA (local) | 16 KV heads | 8 KV heads | 1-2 KV heads |
| GQA (global) | 4 KV heads | 2 KV heads | same as local |
| Global head_dim | 512 (vs 256 local) | 512 (vs 256 local) | 512 (vs 256 local) |
| K=V (global layers) | Yes | Yes | No |
| p-RoPE (global, factor=0.25) | Yes | Yes | Yes |
| MoE (128 experts, top-8) | No | Yes | No |
| Per-Layer Embeddings (PLE) | No | No | Yes (256-dim) |
| KV sharing across layers | No | No | Yes (18-20 layers) |
| Double-wide MLP | No | No | E2B only |
| Audio encoder | No | No | Yes |
| Activation | gelu_pytorch_tanh | gelu_pytorch_tanh | gelu_pytorch_tanh |

---

## Phase 1: Gemma 4 31B Dense (Easiest)

**Why first**: Closest to existing Gemma 3 adapter. No MoE, no PLE. The new features
(per-layer GQA, K=V, p-RoPE, global head_dim) are all handled natively by HF — the
bridge just needs to expose the right hooks.

### Step 1.1: Create `Gemma4ArchitectureAdapter`

Create `transformer_lens/model_bridge/supported_architectures/gemma4.py`, modeled on
the Gemma 3 adapter.

**Key differences from Gemma 3 adapter:**

1. **Per-layer head dimensions vary**: Global attention layers use `global_head_dim=512`
   while sliding layers use `head_dim=256`. The weight processing conversions need to
   handle this per-layer. In HF, `Gemma4TextAttention` already picks the right dim
   based on `self.is_sliding`. The bridge needs to:
   - Determine per-layer whether it's sliding or full from `config.layer_types`
   - Use the correct head_dim in weight rearrangement conversions for Q/K/V/O

2. **Per-layer KV head count varies**: Global layers use `num_global_key_value_heads`
   (4 for 31B) while sliding layers use `num_key_value_heads` (16 for 31B). The K/V
   weight rearrangement needs per-layer `n` values.

3. **K=V on global layers**: When `attention_k_eq_v=True`, global attention layers have
   `v_proj=None` and reuse `key_states` as `value_states`. The bridge needs to:
   - Not create a `v` LinearBridge for global layers (or create one that's a no-op)
   - Hook `hook_v` to expose the key states (which ARE the value states)
   - Weight conversions should skip V weights for these layers

4. **v_norm**: Gemma 4 adds `v_norm` (RMSNorm on values, with_scale=False). This is
   new vs Gemma 3 which only had q_norm and k_norm.

5. **p-RoPE**: Global attention uses `partial_rotary_factor=0.25` via rope_parameters.
   HF handles this natively in `Gemma4TextRotaryEmbedding`. The bridge just needs
   to pass through the rotary embeddings correctly per layer type.

6. **Activation**: Changed from `silu` (Gemma 3) to `gelu_pytorch_tanh`.

7. **layer_scalar**: Each decoder layer has a `layer_scalar` buffer that multiplies
   the output. Need to expose or preserve this.

**Implementation tasks:**

- [ ] Copy `gemma3.py` adapter as starting point
- [ ] Update component mapping for per-layer attention differences
- [ ] Handle K=V in the attention bridge (may need a new bridge variant or config flag)
- [ ] Add v_norm to attention submodules
- [ ] Update weight processing conversions with per-layer head_dim/n_kv_heads
- [ ] Register `Gemma4ForCausalLM` in the architecture adapter factory
- [ ] Write unit tests for config loading
- [ ] Write component bridge tests

### Step 1.2: Handle Per-Layer Attention Differences in the Bridge

The biggest architectural challenge: the bridge currently assumes uniform attention
parameters across all layers. Options:

**Option A (Recommended)**: Let HF handle the per-layer differences natively. The
`PositionEmbeddingsAttentionBridge` wraps the HF attention module's forward call.
Since each HF `Gemma4TextAttention` already knows its own head_dim, KV heads, and
whether it uses K=V, the bridge just needs correct weight conversions per layer.

For weight conversions, use layer-index-aware conversions:
```python
# Pseudo-code
for i in range(num_layers):
    is_global = layer_types[i] == "full_attention"
    hd = global_head_dim if is_global else head_dim
    n_kv = num_global_kv_heads if is_global and k_eq_v else num_kv_heads
    # Register per-layer conversions
```

**Option B**: Create two attention bridge templates (one for sliding, one for global)
and select per layer. More explicit but more code.

### Step 1.3: Test with 31B

- Test config loading from HuggingFace
- Test forward pass matches HF output (use small random inputs)
- Test hook access (hook_q, hook_k, hook_v, hook_pattern, hook_resid_pre, etc.)
- Test activation patching works

---

## Phase 2: Gemma 4 26B-A4B MoE (Medium)

**Why second**: Builds on Phase 1, adds MoE which already has bridge support via
`MoEBridge` (used by Mixtral). The Gemma 4 MoE is slightly different from Mixtral.

### Step 2.1: Understand Gemma 4 MoE vs Mixtral

| Feature | Mixtral | Gemma 4 26B-A4B |
|---|---|---|
| Num experts | 8 | 128 |
| Top-k | 2 | 8 |
| Router | simple linear gate | Gemma4TextRouter with per_expert_scale and scalar_root_size |
| Expert impl | batched 3D tensors | `MixtralExperts` (same!) |
| Shared expert | No | **No** (despite the blog post, the actual config has no shared expert) |

Good news: Looking at the actual HF code, `Gemma4TextExperts` inherits from
`MixtralExperts` directly. The router is different but the bridge wraps the entire
MoE module.

### Step 2.2: Create MoE-aware Gemma 4 Adapter

Extend the Phase 1 adapter to handle MoE layers:

- [ ] In the component mapping, use `MoEBridge` for the MLP when `enable_moe_block=True`
- [ ] Map the `router` as a submodule of the MoE bridge (like Mixtral does)
- [ ] The `experts` are batched (3D tensors), same as Mixtral — existing bridge handles this
- [ ] Handle the fact that MoE layers have extra norms: `pre_feedforward_layernorm_2`
      and `post_feedforward_layernorm_2` for the second path
- [ ] Weight conversions: MoE expert weights are 3D tensors, need appropriate handling

### Step 2.3: Test with 26B-A4B

- Test config loading
- Test forward pass matches HF (use small inputs or mock weights)
- Test router hook access (hook_router_scores / hook_expert_weights / hook_expert_indices)
- Test that MoE routing can be inspected via hooks

---

## Phase 3: Gemma 4 E2B/E4B - Per-Layer Embeddings (Hardest)

**Why last**: Per-Layer Embeddings (PLE) is a fundamentally new architectural concept
not present in any existing TransformerLens model.

### Step 3.1: Understand PLE Architecture

From the HF implementation:

1. **At model level** (`Gemma4TextModel`):
   - `embed_tokens_per_layer`: Embedding table of shape
     `(vocab_size, num_layers * hidden_size_per_layer_input)` — one 256-dim embedding
     per token per layer
   - `per_layer_model_projection`: Linear projection from `hidden_size` to
     `num_layers * hidden_size_per_layer_input`
   - `per_layer_projection_norm`: RMSNorm

2. **Per lookup** (`get_per_layer_inputs`):
   - Look up input_ids in `embed_tokens_per_layer`
   - Reshape to `(batch, seq, num_layers, 256)`

3. **Per projection** (`project_per_layer_inputs`):
   - Project main embeddings via `per_layer_model_projection`
   - Add to per-layer embeddings, scale by `2^-0.5`

4. **At each decoder layer** (`Gemma4TextDecoderLayer`):
   - After attention+MLP, apply PLE:
     - `per_layer_input_gate`: Linear(hidden_size → 256)
     - Apply activation function
     - Element-wise multiply with that layer's per-layer embedding
     - `per_layer_projection`: Linear(256 → hidden_size)
     - `post_per_layer_input_norm`: RMSNorm
     - Add to residual

### Step 3.2: Create PLE Bridge Components

This is the novel work. Options:

**Option A (Recommended)**: Create a `PerLayerEmbeddingBridge` generalized component
that wraps the PLE lookup and projection at the model level, and modify the
`BlockBridge` to pass per-layer inputs through.

New components needed:
- [ ] `PerLayerEmbeddingBridge` — wraps `embed_tokens_per_layer` + projection logic
- [ ] `PerLayerGateBridge` — wraps the per-layer gate+projection inside each block

New hooks to expose:
- `hook_per_layer_embed` — the raw per-layer embedding lookup
- `hook_per_layer_projected` — after projection and combination
- `blocks.{i}.hook_per_layer_input` — the per-layer input at each block
- `blocks.{i}.hook_per_layer_gate` — gate activation
- `blocks.{i}.hook_per_layer_output` — gated + projected output before adding to residual

**Option B**: Let HF handle PLE entirely in its forward pass and only hook the
residual stream. Simpler but loses interpretability of the PLE mechanism.

### Step 3.3: Handle KV Sharing

E2B and E4B use `num_kv_shared_layers` (20 and 18 respectively). The last N layers
reuse KV projections from earlier layers. In HF this is handled via
`past_key_values.shared_layers`.

For the bridge:
- [ ] The KV sharing is handled inside HF's forward pass, so for inference it "just works"
- [ ] For interpretability, we need to be aware that `hook_k` / `hook_v` on shared
      layers will show the reused values from the source layer, not freshly computed ones
- [ ] Document this behavior clearly

### Step 3.4: Handle Double-Wide MLP (E2B only)

E2B uses `use_double_wide_mlp=True` for KV-shared layers, which doubles the
`intermediate_size` for those layers. The GatedMLPBridge needs to handle non-uniform
MLP dimensions. Since HF handles this per-layer in `Gemma4TextMLP.__init__`, the
bridge just needs correct weight conversions.

### Step 3.5: Test E2B/E4B

- Test config loading
- Test forward pass matches HF
- Test PLE hooks are accessible and contain expected shapes
- Test that PLE embeddings can be inspected/patched

---

## Phase 4: Multimodal Support (Optional/Future)

All Gemma 4 models support image input. E2B/E4B also support audio.

### Step 4.1: Vision Encoder

Similar to `Gemma3MultimodalArchitectureAdapter`, create a multimodal variant:
- [ ] `Gemma4MultimodalArchitectureAdapter` wrapping `Gemma4ForConditionalGeneration`
- [ ] Vision encoder: `Gemma4VisionEncoder` uses a different architecture from Gemma 3's
      SigLIP — it has its own `Gemma4VisionAttention` with 2D RoPE and pooling
- [ ] Need a `Gemma4VisionEncoderBridge` (or extend existing vision bridge)
- [ ] Vision projection layer bridge

### Step 4.2: Audio Encoder (E2B/E4B only)

- [ ] `Gemma4AudioModel` uses a Conformer architecture — completely new
- [ ] Would need a `ConformerBridge` or `AudioEncoderBridge`
- [ ] Lower priority — TransformerLens is primarily text-focused

---

## Estimated Effort

| Phase | Effort | New Components Needed | Risk |
|---|---|---|---|
| Phase 1 (31B) | ~2-3 days | Per-layer weight conversion logic | Low — closest to Gemma 3 |
| Phase 2 (MoE) | ~1-2 days | Minor MoE bridge extensions | Low — MoE bridge exists |
| Phase 3 (PLE) | ~3-5 days | PerLayerEmbeddingBridge, PerLayerGateBridge | Medium — novel concept |
| Phase 4 (Multimodal) | ~3-5 days | VisionEncoderBridge variant, AudioBridge | Medium-High |

**Total for text-only support (Phases 1-3): ~6-10 days of focused work**

---

## Prerequisites

1. `transformers >= 5.5.0` (Gemma 4 support)
2. Working dev-3.x branch checkout
3. GPU access for testing larger models (26B-A4B and 31B need significant VRAM)
   - E2B (~5B params): ~10GB in float16
   - E4B (~8B params): ~16GB in float16
   - 26B-A4B: ~52GB in float16 (but only 4B active)
   - 31B: ~62GB in float16

## Testing Strategy

For each phase:
1. **Config test**: Load HF config, verify TransformerLens config mapping
2. **Component test**: Per-bridge-component forward pass comparison with HF
3. **End-to-end test**: Full model forward pass, compare logits
4. **Hook test**: Verify all hooks fire and contain expected tensor shapes
5. **Interpretability test**: Run basic activation patching or logit lens
