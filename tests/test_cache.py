from __future__ import annotations

from memorytask.cache import make_cache_key


def _base_kwargs(**overrides):
    kwargs = dict(
        model_slug="openai/gpt-6-luna",
        provider_order=["OpenAI"],
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        response_format=None,
        temperature=1.0,
        max_tokens=64,
        seed=7,
        sample_index=0,
        cache_namespace="t01_smoke",
    )
    kwargs.update(overrides)
    return kwargs


def test_cache_key_deterministic_and_sample_index_matters():
    k1 = make_cache_key(**_base_kwargs())
    k2 = make_cache_key(**_base_kwargs())
    assert k1 == k2
    assert len(k1) == 64
    k3 = make_cache_key(**_base_kwargs(sample_index=1))
    assert k3 != k1


def test_different_cache_namespace_gives_different_key():
    k1 = make_cache_key(**_base_kwargs(cache_namespace="t01_smoke"))
    k2 = make_cache_key(**_base_kwargs(cache_namespace="t05_killtest"))
    assert k1 != k2


def test_cache_key_includes_max_tokens():
    """Degraded (shrunk) max_tokens responses must never be reused at original limits."""
    k_full = make_cache_key(**_base_kwargs(max_tokens=2048))
    k_shrunk = make_cache_key(**_base_kwargs(max_tokens=256))
    assert k_full != k_shrunk
    k_none = make_cache_key(**_base_kwargs(max_tokens=None))
    assert k_none != k_full
