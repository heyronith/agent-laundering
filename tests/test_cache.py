from __future__ import annotations

from memorytask.cache import make_cache_key


def test_cache_key_deterministic_and_sample_index_matters():
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
    )
    k1 = make_cache_key(**kwargs)
    k2 = make_cache_key(**kwargs)
    assert k1 == k2
    assert len(k1) == 64

    kwargs2 = dict(kwargs)
    kwargs2["sample_index"] = 1
    k3 = make_cache_key(**kwargs2)
    assert k3 != k1
