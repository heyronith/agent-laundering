from __future__ import annotations

import json
import warnings
from pathlib import Path

import httpx
import pytest
import respx

from memorytask.cache import make_cache_key
from memorytask.calllog import append_call_log, read_jsonl
from memorytask.client import chat


def _write_price_snapshot(prices_dir: Path, slug: str = "deepseek/deepseek-v4.1-flash-1017"):
    prices_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "by_slug": {
            slug: {
                "slug": slug,
                "input_per_million": 0.3,
                "output_per_million": 1.2,
            }
        },
        "models": {
            "deepseek": {
                "slug": slug,
                "input_per_million": 0.3,
                "output_per_million": 1.2,
            }
        },
    }
    (prices_dir / "prices_20261109.json").write_text(json.dumps(payload), encoding="utf-8")


def _seed_successful_log(logs_dir: Path, *, cache_key: str, provider: str = "DeepSeek"):
    record = {
        "call_id": "seed-1",
        "run_id": "seed_run",
        "task_id": "seed",
        "role": "subject",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "attempt": 1,
        "model_alias": "deepseek",
        "model_slug_requested": "deepseek/deepseek-v4.1-flash-1017",
        "model_served": "deepseek/deepseek-v4.1-flash-1017",
        "provider_requested": provider,
        "provider_served": provider,
        "request": {"messages": [{"role": "user", "content": "hi"}], "tools": None, "params": {}},
        "response_raw": {
            "id": "gen_seed",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "provider": provider,
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0.0001},
        },
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        "cost_usd": 0.0001,
        "cost_source": "response",
        "latency_ms": 12,
        "http_status": 200,
        "error": None,
        "cache_key": cache_key,
        "cache_hit": False,
        "sample_index": 0,
        "seed": None,
        "provider_mismatch": False,
    }
    append_call_log(logs_dir, "seed_run", record)


@respx.mock
def test_cache_hit_makes_zero_http_calls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    messages = [{"role": "user", "content": "hi"}]
    cache_key = make_cache_key(
        model_slug="deepseek/deepseek-v4.1-flash-1017",
        provider_order=["DeepSeek"],
        messages=messages,
        tools=None,
        response_format=None,
        temperature=1.0,
        max_tokens=256,
        seed=None,
        sample_index=0,
    )
    _seed_successful_log(logs_dir, cache_key=cache_key)

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"should": "not be called"})
    )

    result = chat(
        "deepseek",
        messages,
        run_id="cache_run",
        task_id="t",
        role="subject",
        budget_cap_usd=1.0,
        logs_dir=logs_dir,
        prices_dir=prices_dir,
        sleep_fn=lambda _s: None,
    )
    assert route.call_count == 0
    assert result["cache_hit"] is True
    assert result["cost_usd"] == 0.0
    rows = read_jsonl(logs_dir / "cache_run.jsonl")
    assert rows[-1]["cache_hit"] is True
    assert rows[-1]["cost_usd"] == 0.0
    assert "sk_or_" not in json.dumps(rows[-1])


@respx.mock
def test_retry_on_429_then_give_up(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "rate limit"})
    )

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        chat(
            "deepseek",
            [{"role": "user", "content": "hi"}],
            run_id="retry_run",
            task_id="t",
            role="subject",
            budget_cap_usd=1.0,
            logs_dir=logs_dir,
            prices_dir=prices_dir,
            sleep_fn=lambda _s: None,
        )
    assert route.call_count == 5
    rows = read_jsonl(logs_dir / "retry_run.jsonl")
    assert len(rows) == 5
    assert all(r["http_status"] == 429 for r in rows)


@respx.mock
def test_retry_on_500(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(
                200,
                json={
                    "id": "gen_ok",
                    "model": "deepseek/deepseek-v4.1-flash-1017",
                    "provider": "DeepSeek",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.00001},
                },
            ),
        ]
    )
    result = chat(
        "deepseek",
        [{"role": "user", "content": "hi"}],
        run_id="retry500",
        task_id="t",
        role="subject",
        budget_cap_usd=1.0,
        logs_dir=logs_dir,
        prices_dir=prices_dir,
        sleep_fn=lambda _s: None,
    )
    assert route.call_count == 3
    assert result["cache_hit"] is False
    assert result["cost_usd"] == 0.00001


@respx.mock
def test_provider_mismatch_flagged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen_mismatch",
                "model": "deepseek/deepseek-v4.1-flash-1017",
                "provider": "Fireworks",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.00001},
            },
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = chat(
            "deepseek",
            [{"role": "user", "content": "hi"}],
            run_id="mismatch_run",
            task_id="t",
            role="subject",
            budget_cap_usd=1.0,
            logs_dir=logs_dir,
            prices_dir=prices_dir,
            sleep_fn=lambda _s: None,
        )
    assert result["provider_mismatch"] is True
    assert any("provider_mismatch" in str(w.message) for w in caught)
    row = read_jsonl(logs_dir / "mismatch_run.jsonl")[-1]
    assert row["provider_mismatch"] is True
    assert row["provider_requested"] == "DeepSeek"
    assert row["provider_served"] == "Fireworks"
