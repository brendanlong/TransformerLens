"""Shared model fixture system for TransformerLens tests.

Downloads each model once per test session, groups all tests needing a model together,
and selectively evicts that model from the HF cache when its group finishes.

IMPORTANT: This file must NOT import transformer_lens at the top level.
The jaxtyping pytest plugin needs to install import hooks before transformer_lens
is imported, so we defer all TL imports to inside functions/fixtures.
"""

import gc
import hashlib
import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Models used by the migrated test files.
# Without HF_TOKEN we only test the freely-downloadable subset.
_DEFAULT_MODELS = [
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/pythia-14m",
    # Small NeelNanda / custom models (kept cached, never evicted)
    "attn-only-demo",
    "solu-4l-old",
    "solu-6l",
    "attn-only-3l",
    "gelu-2l",
    "othello-gpt",
    "tiny-stories-33M",
    "solu-1l",
    "redwood_attn_2l",
    # Encoder / encoder-decoder
    "bert-base-cased",
    "t5-small",
    # Public HF models
    "microsoft/phi-1",
    "google/gemma-2b",
    "EleutherAI/pythia-70m",
]

_EXTENDED_MODELS = _DEFAULT_MODELS + [
    "EleutherAI/gpt-neo-125M",
    "stanford-gpt2-small-a",
    "bigscience/bloom-560m",
    "bigcode/santacoder",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "google/gemma-7b",
    "Qwen/Qwen2-0.5B",
]

ALL_TEST_MODELS = _EXTENDED_MODELS if os.environ.get("HF_TOKEN", "") else _DEFAULT_MODELS


# ---------------------------------------------------------------------------
# Tiny models not worth evicting from cache
# ---------------------------------------------------------------------------

KEEP_CACHED = {
    "solu-1l",
    "solu-2l",
    "solu-4l-old",
    "solu-6l",
    "attn-only-1l",
    "attn-only-2l",
    "attn-only-3l",
    "attn-only-4l",
    "attn-only-demo",
    "gelu-2l",
    "gelu-4l",
    "tiny-stories-1M",
    "tiny-stories-33M",
    "othello-gpt",
    "redwood_attn_2l",
}


# ---------------------------------------------------------------------------
# Alias resolution: map any TL alias to its canonical HF repo name
#
# Deferred: _CANONICAL_MAP is built lazily on first access to avoid
# importing transformer_lens at module load time.
# ---------------------------------------------------------------------------

_CANONICAL_MAP = None
_seen_official = None


def _ensure_alias_map():
    """Build the alias map on first use (defers transformer_lens import)."""
    global _CANONICAL_MAP, _seen_official
    if _CANONICAL_MAP is not None:
        return

    from transformer_lens.loading_from_pretrained import get_official_model_name

    _CANONICAL_MAP = {}
    _seen_official = {}
    for name in ALL_TEST_MODELS:
        try:
            official = get_official_model_name(name)
        except ValueError:
            official = name
        if official not in _seen_official:
            _seen_official[official] = name
        _CANONICAL_MAP[name] = _seen_official[official]


def _resolve_to_official(model_name: str) -> str:
    """Resolve a TL alias to its official HF repo name."""
    from transformer_lens.loading_from_pretrained import get_official_model_name

    try:
        return get_official_model_name(model_name)
    except ValueError:
        return model_name


def canonical_model_name(name: str) -> str:
    """Return the ALL_TEST_MODELS entry that this alias resolves to."""
    _ensure_alias_map()
    if name in _CANONICAL_MAP:
        return _CANONICAL_MAP[name]
    official = _resolve_to_official(name)
    return _seen_official.get(official, name)


# ---------------------------------------------------------------------------
# Marker: @pytest.mark.needs_model("gpt2-small", "opt-125m", ...)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_model(*model_names): run test only when current_model_name matches one of the given models",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        markers = list(item.iter_markers("needs_model"))
        if not markers:
            continue
        needed_canonical = set()
        for marker in markers:
            for name in marker.args:
                needed_canonical.add(canonical_model_name(name))
        if hasattr(item, "callspec") and "current_model_name" in item.callspec.params:
            current = item.callspec.params["current_model_name"]
            if current not in needed_canonical:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"needs one of {needed_canonical}, current is {current}"
                    )
                )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", params=ALL_TEST_MODELS)
def current_model_name(request):
    """Iterates over all test models. Pytest groups all dependent tests per model value."""
    yield request.param

    # Teardown: evict from HF cache unless it's a tiny kept model
    if request.param not in KEEP_CACHED and os.environ.get("EVICT_MODEL_CACHE", "1") == "1":
        _evict_model(request.param)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _param_checksum(model) -> str:
    """Quick checksum of model parameters to detect mutation."""
    h = hashlib.md5(usedforsecurity=False)
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes()[:64])
    return h.hexdigest()


@pytest.fixture(scope="session")
def loaded_model(current_model_name):
    """Shared read-only HookedTransformer. Teardown asserts no mutation occurred."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(current_model_name, device="cpu")
    checksum = _param_checksum(model)
    yield model
    assert _param_checksum(model) == checksum, (
        f"loaded_model for {current_model_name} was mutated! "
        "Tests that modify models should load their own via current_model_name."
    )
    del model
    gc.collect()


@pytest.fixture(scope="session")
def loaded_model_no_processing(current_model_name):
    """Shared read-only model loaded without weight processing."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained_no_processing(current_model_name, device="cpu")
    checksum = _param_checksum(model)
    yield model
    assert (
        _param_checksum(model) == checksum
    ), f"loaded_model_no_processing for {current_model_name} was mutated!"
    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Selective HF cache eviction
# ---------------------------------------------------------------------------


def _evict_model(model_name: str):
    """Delete a single model from the HF cache (CI only)."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    try:
        from huggingface_hub import scan_cache_dir

        hf_repo = _resolve_to_official(model_name)
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == hf_repo:
                strategy = cache_info.delete_revisions(*(rev.commit_hash for rev in repo.revisions))
                strategy.execute()
                break
    except Exception:
        pass  # best-effort eviction
