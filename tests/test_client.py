from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from memorytask.cache import make_cache_key, reset_process_cache
from memorytask.calllog import append_call_log, read_jsonl
from memorytask.client import ProviderMismatchError, chat
from memorytask.config import get_model


PINNED = get_model("deepseek")["provider_order"][0]
SLUG = get_model("deepseek")["slug"]


def _write_price_snapshot(prices_dir: Path, slug: str = SLUG):
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


def _seed_successful_log(logs_dir: Path, *, cache_key: str, provider: str = PINNED):
    record = {
        "call_id": "seed-1",
        "run_id": "seed_run",
        "task_id": "seed",
        "role": "subject",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "attempt": 1,
        "model_alias": "deepseek",
        "model_slug_requested": SLUG,
        "model_served": SLUG,
        "provider_requested": provider,
        "provider_served": provider,
        "request": {"messages": [{"role": "user", "content": "hi"}], "tools": None, "params": {}},
        "response_raw": {
            "id": "gen_seed",
            "model": SLUG,
            "provider": provider,
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0.0001},
        },
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "reasoning_tokens": 0},
        "cost_usd": 0.0001,
        "cost_source": "response",
        "latency_ms": 12,
        "http_status": 200,
        "error": None,
        "cache_key": cache_key,
        "cache_hit": False,
        "cache_namespace": "t01_smoke",
        "sample_index": 0,
        "seed": None,
        "provider_mismatch": False,
    }
    append_call_log(logs_dir, "seed_run", record)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_process_cache()
    yield
    reset_process_cache()


@respx.mock
def test_cache_hit_makes_zero_http_calls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    messages = [{"role": "user", "content": "hi"}]
    model = get_model("deepseek")
    cache_key = make_cache_key(
        model_slug=SLUG,
        provider_order=list(model["provider_order"]),
        messages=messages,
        tools=None,
        response_format=None,
        temperature=model["temperature"],
        max_tokens=model["max_tokens"],
        seed=None,
        sample_index=0,
        cache_namespace="t01_smoke",
    )
    _seed_successful_log(logs_dir, cache_key=cache_key)

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"should": "not be called"})
    )

    result = chat(
        "deepseek",
        messages,
        cache_namespace="t01_smoke",
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
    assert rows[-1]["cache_namespace"] == "t01_smoke"
    assert ("sk-or" + "-") not in json.dumps(rows[-1])


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
            cache_namespace="t01_smoke",
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
                    "model": SLUG,
                    "provider": PINNED,
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cost": 0.00001,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            ),
        ]
    )
    result = chat(
        "deepseek",
        [{"role": "user", "content": "hi"}],
        cache_namespace="t01_smoke",
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
def test_provider_mismatch_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen_mismatch",
                "model": SLUG,
                "provider": "Fireworks",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.00001},
            },
        )
    )

    with pytest.raises(ProviderMismatchError, match="provider_mismatch"):
        chat(
            "deepseek",
            [{"role": "user", "content": "hi"}],
            cache_namespace="t01_smoke",
            run_id="mismatch_run",
            task_id="t",
            role="subject",
            budget_cap_usd=1.0,
            logs_dir=logs_dir,
            prices_dir=prices_dir,
            sleep_fn=lambda _s: None,
            strict_provider=True,
        )
    row = read_jsonl(logs_dir / "mismatch_run.jsonl")[-1]
    assert row["provider_mismatch"] is True
    assert row["provider_requested"] == PINNED
    assert row["provider_served"] == "Fireworks"


@respx.mock
def test_reasoning_field_present_in_request_payload(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen_reason",
                "model": SLUG,
                "provider": PINNED,
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cost": 0.00001,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )
    )
    chat(
        "deepseek",
        [{"role": "user", "content": "hi"}],
        cache_namespace="t01_smoke",
        run_id="reason_run",
        task_id="t",
        role="subject",
        budget_cap_usd=1.0,
        logs_dir=logs_dir,
        prices_dir=prices_dir,
        sleep_fn=lambda _s: None,
    )
    sent = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert "reasoning" in sent
    assert sent["reasoning"] == get_model("deepseek")["reasoning"]
    row = read_jsonl(logs_dir / "reason_run.jsonl")[-1]
    assert row["request"]["params"]["reasoning"] == get_model("deepseek")["reasoning"]
    assert "reasoning_tokens" in row["usage"]


@respx.mock
def test_cache_index_does_not_reread_logs_on_second_call(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-not-real")
    logs_dir = tmp_path / "logs"
    prices_dir = tmp_path / "prices"
    _write_price_snapshot(prices_dir)

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen_ok",
                "model": SLUG,
                "provider": PINNED,
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.00001},
            },
        )
    )

    common = dict(
        cache_namespace="t01_smoke",
        role="subject",
        budget_cap_usd=1.0,
        logs_dir=logs_dir,
        prices_dir=prices_dir,
        sleep_fn=lambda _s: None,
        sample_index=0,
    )

    with patch("memorytask.cache.iter_run_logs", wraps=__import__("memorytask.cache", fromlist=["iter_run_logs"]).iter_run_logs) as spy:
        chat(
            "deepseek",
            [{"role": "user", "content": "hi"}],
            run_id="idx_run",
            task_id="t1",
            **common,
        )
        first_reads = spy.call_count
        assert first_reads >= 1
        # Second call with different sample_index → cache miss → paid call, but index already built.
        chat(
            "deepseek",
            [{"role": "user", "content": "hi"}],
            run_id="idx_run",
            task_id="t2",
            **{**common, "sample_index": 1},
        )
        assert spy.call_count == first_reads
    assert route.call_count == 2
