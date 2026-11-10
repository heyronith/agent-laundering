"""OpenRouter chat client — the ONLY module that performs API HTTP calls."""

from __future__ import annotations

import json
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from memorytask.budget import (
    BudgetExceeded,
    append_ledger,
    check_budget,
    estimate_worst_case_cost_usd,
)
from memorytask.cache import PROCESS_CACHE_INDEX, make_cache_key
from memorytask.calllog import append_call_log
from memorytask.config import (
    get_api_key,
    get_model,
    load_run_defaults,
    repo_root,
)


class ProviderMismatchError(Exception):
    """Raised when provider_served differs from provider_requested."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _latest_price_snapshot(prices_dir: Path) -> dict[str, Any] | None:
    if not prices_dir.exists():
        return None
    files = sorted(prices_dir.glob("prices_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _prices_for_slug(snapshot: dict[str, Any] | None, slug: str) -> tuple[float, float]:
    """Return (input_$/1M, output_$/1M) from snapshot; fall back to expensive defaults only for estimation."""
    if not snapshot:
        return 10.0, 40.0
    models = snapshot.get("models") or snapshot.get("data") or []
    if isinstance(models, dict):
        for entry in models.values():
            if entry.get("slug") == slug:
                return float(entry["input_per_million"]), float(entry["output_per_million"])
        if slug in models:
            entry = models[slug]
            return float(entry["input_per_million"]), float(entry["output_per_million"])
    elif isinstance(models, list):
        for entry in models:
            if entry.get("id") == slug or entry.get("slug") == slug:
                pricing = entry.get("pricing") or {}
                inp = float(pricing.get("prompt") or 0.0) * 1_000_000
                out = float(pricing.get("completion") or 0.0) * 1_000_000
                return inp, out
    by_slug = snapshot.get("by_slug") or {}
    if slug in by_slug:
        entry = by_slug[slug]
        return float(entry["input_per_million"]), float(entry["output_per_million"])
    return 10.0, 40.0


def _estimate_prompt_tokens(messages: list[dict[str, Any]], tools: Any) -> int:
    blob = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    return max(32, len(blob) // 2)


def _extract_usage(response_json: dict[str, Any]) -> dict[str, Any]:
    usage = response_json.get("usage") or {}
    out = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": None,
    }
    details = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        out["reasoning_tokens"] = details.get("reasoning_tokens")
    if "reasoning_tokens" in usage:
        out["reasoning_tokens"] = usage.get("reasoning_tokens")
    for key in ("native_tokens_reasoning", "output_tokens_reasoning"):
        if key in usage:
            out["reasoning_tokens"] = usage.get(key)
    return out


def _cost_from_response(response_json: dict[str, Any]) -> float | None:
    usage = response_json.get("usage") or {}
    if "cost" in usage and usage["cost"] is not None:
        return float(usage["cost"])
    if "total_cost" in usage and usage["total_cost"] is not None:
        return float(usage["total_cost"])
    return None


def _cost_from_price_snapshot(
    snapshot: dict[str, Any] | None,
    slug: str,
    usage: dict[str, Any],
) -> float:
    inp_m, out_m = _prices_for_slug(snapshot, slug)
    prompt = float(usage.get("prompt_tokens") or 0)
    completion = float(usage.get("completion_tokens") or 0)
    return (prompt / 1_000_000.0) * inp_m + (completion / 1_000_000.0) * out_m


def _provider_from_response(response_json: dict[str, Any], headers: httpx.Headers) -> str | None:
    if "provider" in response_json and response_json["provider"]:
        return str(response_json["provider"])
    for header in ("x-openrouter-provider", "X-OpenRouter-Provider"):
        if header in headers:
            return headers.get(header)
    return None


def _request_params(
    *,
    temp: float | None,
    max_toks: int | None,
    seed: int | None,
    response_format: Any,
    provider: dict[str, Any],
    reasoning: Any,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "temperature": temp,
        "max_tokens": max_toks,
        "seed": seed,
        "response_format": response_format,
        "provider": provider,
        "reasoning": reasoning,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
    }


def chat(
    model_alias: str,
    messages: list[dict[str, Any]],
    *,
    cache_namespace: str,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    stop: list[str] | None = None,
    seed: int | None = None,
    run_id: str,
    task_id: str,
    role: str,
    sample_index: int = 0,
    budget_cap_usd: float,
    strict_provider: bool = True,
    logs_dir: Path | None = None,
    prices_dir: Path | None = None,
    http_client: httpx.Client | None = None,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    """
    Perform a chat completion via OpenRouter with cache, budget, retry, and logging.

    Returns a dict with keys: response_raw, usage, cost_usd, cost_source, cache_hit,
    call_id, provider_served, model_served, log_record.
    """
    if not cache_namespace:
        raise ValueError("cache_namespace is required")

    defaults = load_run_defaults()
    root = repo_root()
    logs_dir = logs_dir or (root / defaults.get("logs_dir", "logs"))
    prices_dir = prices_dir or (root / defaults.get("prices_dir", "data/prices"))
    base_url = defaults.get("openrouter_base_url", "https://openrouter.ai/api/v1").rstrip("/")
    max_retries = int(defaults.get("max_retries", 5))
    timeout_s = float(defaults.get("request_timeout_s", 120))
    backoff_base = float(defaults.get("backoff_base_s", 1.0))
    backoff_max = float(defaults.get("backoff_max_s", 30.0))
    global_cap = float(defaults.get("global_cap_usd", 95.0))

    model_cfg = get_model(model_alias)
    slug = model_cfg["slug"]
    provider_order = list(model_cfg["provider_order"])
    if len(provider_order) != 1:
        raise ValueError(f"Model {model_alias}: provider_order must pin exactly one provider")
    allow_fallbacks = bool(model_cfg.get("allow_fallbacks", False))
    if allow_fallbacks:
        raise ValueError(f"Model {model_alias} must have allow_fallbacks: false")

    temp = model_cfg["temperature"] if temperature is None else temperature
    max_toks = model_cfg["max_tokens"] if max_tokens is None else max_tokens
    reasoning_cfg = model_cfg.get("reasoning")

    cache_key = make_cache_key(
        model_slug=slug,
        provider_order=provider_order,
        messages=messages,
        tools=tools,
        response_format=response_format,
        temperature=temp,
        max_tokens=max_toks,
        seed=seed,
        sample_index=sample_index,
        cache_namespace=cache_namespace,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )

    request_payload: dict[str, Any] = {
        "model": slug,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_toks,
        "provider": {"order": provider_order, "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if reasoning_cfg is not None:
        request_payload["reasoning"] = reasoning_cfg
    if tools is not None:
        request_payload["tools"] = tools
    if response_format is not None:
        request_payload["response_format"] = response_format
    if seed is not None:
        request_payload["seed"] = seed
    if top_p is not None:
        request_payload["top_p"] = top_p
    if frequency_penalty is not None:
        request_payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        request_payload["presence_penalty"] = presence_penalty
    if stop is not None:
        request_payload["stop"] = stop

    # Cache check — process-level index (scanned once per logs_dir).
    index = PROCESS_CACHE_INDEX.get(logs_dir)
    if cache_key in index:
        cached = index[cache_key]
        call_id = str(uuid.uuid4())
        record = {
            "call_id": call_id,
            "run_id": run_id,
            "task_id": task_id,
            "role": role,
            "timestamp_utc": _utc_now(),
            "attempt": 0,
            "model_alias": model_alias,
            "model_slug_requested": slug,
            "model_served": cached.get("model_served"),
            "provider_requested": provider_order[0],
            "provider_served": cached.get("provider_served"),
            "request": {
                "messages": messages,
                "tools": tools,
                "params": _request_params(
                    temp=temp,
                    max_toks=max_toks,
                    seed=seed,
                    response_format=response_format,
                    provider=request_payload["provider"],
                    reasoning=reasoning_cfg,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                ),
            },
            "response_raw": cached.get("response_raw"),
            "usage": cached.get("usage") or {},
            "cost_usd": 0.0,
            "cost_source": "cache",
            "latency_ms": 0,
            "http_status": 200,
            "error": None,
            "cache_key": cache_key,
            "cache_hit": True,
            "cache_namespace": cache_namespace,
            "sample_index": sample_index,
            "seed": seed,
            "provider_mismatch": False,
        }
        append_call_log(logs_dir, run_id, record)
        return {
            "response_raw": record["response_raw"],
            "usage": record["usage"],
            "cost_usd": 0.0,
            "cost_source": "cache",
            "cache_hit": True,
            "call_id": call_id,
            "provider_served": record["provider_served"],
            "model_served": record["model_served"],
            "log_record": record,
        }

    snapshot = _latest_price_snapshot(prices_dir)
    inp_m, out_m = _prices_for_slug(snapshot, slug)
    estimate = estimate_worst_case_cost_usd(
        prompt_tokens_estimate=_estimate_prompt_tokens(messages, tools),
        max_tokens=int(max_toks),
        input_price_per_million=inp_m,
        output_price_per_million=out_m,
    )
    check_budget(
        logs_dir,
        run_id=run_id,
        run_cap_usd=budget_cap_usd,
        global_cap_usd=global_cap,
        estimate_usd=estimate,
    )

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout_s)
    api_key = get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/heyronith/memorytask",
        "X-Title": "memorytask",
    }

    last_error: str | None = None
    last_status: int | None = None
    response_json: dict[str, Any] | None = None
    response_headers: httpx.Headers | None = None
    latency_ms = 0
    attempt_used = 0

    try:
        for attempt in range(1, max_retries + 1):
            attempt_used = attempt
            call_id = str(uuid.uuid4())
            t0 = time.perf_counter()
            try:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=request_payload,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                last_status = resp.status_code
                try:
                    response_json = resp.json()
                except Exception:
                    response_json = {"raw_text": resp.text}

                retryable = resp.status_code == 429 or resp.status_code >= 500
                if retryable:
                    last_error = f"HTTP {resp.status_code}"
                    err_record = _failed_record(
                        call_id=call_id,
                        run_id=run_id,
                        task_id=task_id,
                        role=role,
                        attempt=attempt,
                        model_alias=model_alias,
                        slug=slug,
                        provider_order=provider_order,
                        messages=messages,
                        tools=tools,
                        temp=temp,
                        max_toks=max_toks,
                        seed=seed,
                        response_format=response_format,
                        request_payload=request_payload,
                        reasoning=reasoning_cfg,
                        response_json=response_json,
                        latency_ms=latency_ms,
                        http_status=resp.status_code,
                        error=last_error,
                        cache_key=cache_key,
                        cache_namespace=cache_namespace,
                        sample_index=sample_index,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                    )
                    append_call_log(logs_dir, run_id, err_record)
                    if attempt >= max_retries:
                        break
                    sleep_fn(min(backoff_max, backoff_base * (2 ** (attempt - 1))))
                    continue

                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {json.dumps(response_json)[:500]}"
                    err_record = _failed_record(
                        call_id=call_id,
                        run_id=run_id,
                        task_id=task_id,
                        role=role,
                        attempt=attempt,
                        model_alias=model_alias,
                        slug=slug,
                        provider_order=provider_order,
                        messages=messages,
                        tools=tools,
                        temp=temp,
                        max_toks=max_toks,
                        seed=seed,
                        response_format=response_format,
                        request_payload=request_payload,
                        reasoning=reasoning_cfg,
                        response_json=response_json,
                        latency_ms=latency_ms,
                        http_status=resp.status_code,
                        error=last_error,
                        cache_key=cache_key,
                        cache_namespace=cache_namespace,
                        sample_index=sample_index,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                    )
                    append_call_log(logs_dir, run_id, err_record)
                    raise RuntimeError(last_error)

                response_headers = resp.headers
                break

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                last_error = f"{type(exc).__name__}: {exc}"
                last_status = None
                err_record = _failed_record(
                    call_id=call_id,
                    run_id=run_id,
                    task_id=task_id,
                    role=role,
                    attempt=attempt,
                    model_alias=model_alias,
                    slug=slug,
                    provider_order=provider_order,
                    messages=messages,
                    tools=tools,
                    temp=temp,
                    max_toks=max_toks,
                    seed=seed,
                    response_format=response_format,
                    request_payload=request_payload,
                    reasoning=reasoning_cfg,
                    response_json=None,
                    latency_ms=latency_ms,
                    http_status=None,
                    error=last_error,
                    cache_key=cache_key,
                    cache_namespace=cache_namespace,
                    sample_index=sample_index,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                append_call_log(logs_dir, run_id, err_record)
                if attempt >= max_retries:
                    break
                sleep_fn(min(backoff_max, backoff_base * (2 ** (attempt - 1))))
                continue

        if response_json is None or last_status is None or last_status >= 400:
            raise RuntimeError(
                f"chat failed after {attempt_used} attempts: {last_error}"
            )

        usage = _extract_usage(response_json)
        cost = _cost_from_response(response_json)
        cost_source = "response"

        if cost is None:
            gen_id = response_json.get("id")
            if gen_id:
                try:
                    gen_resp = client.get(
                        f"{base_url}/generation",
                        params={"id": gen_id},
                        headers=headers,
                    )
                    if gen_resp.status_code == 200:
                        gen_json = gen_resp.json()
                        data = gen_json.get("data") or gen_json
                        if data.get("total_cost") is not None:
                            cost = float(data["total_cost"])
                            cost_source = "generation_api"
                        elif data.get("cost") is not None:
                            cost = float(data["cost"])
                            cost_source = "generation_api"
                except Exception:
                    pass

        if cost is None:
            cost = _cost_from_price_snapshot(snapshot, slug, usage)
            cost_source = "price_snapshot"

        provider_requested = provider_order[0]
        provider_served = _provider_from_response(response_json, response_headers or httpx.Headers())
        model_served = response_json.get("model")
        provider_mismatch = bool(
            provider_served is not None
            and provider_requested is not None
            and provider_served != provider_requested
        )
        if provider_mismatch:
            warnings.warn(
                f"provider_mismatch: requested={provider_requested!r} served={provider_served!r}",
                stacklevel=2,
            )

        call_id = str(uuid.uuid4())
        record = {
            "call_id": call_id,
            "run_id": run_id,
            "task_id": task_id,
            "role": role,
            "timestamp_utc": _utc_now(),
            "attempt": attempt_used,
            "model_alias": model_alias,
            "model_slug_requested": slug,
            "model_served": model_served,
            "provider_requested": provider_requested,
            "provider_served": provider_served,
            "request": {
                "messages": messages,
                "tools": tools,
                "params": _request_params(
                    temp=temp,
                    max_toks=max_toks,
                    seed=seed,
                    response_format=response_format,
                    provider=request_payload["provider"],
                    reasoning=reasoning_cfg,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                ),
            },
            "response_raw": response_json,
            "usage": usage,
            "cost_usd": float(cost),
            "cost_source": cost_source,
            "latency_ms": latency_ms,
            "http_status": last_status,
            "error": None,
            "cache_key": cache_key,
            "cache_hit": False,
            "cache_namespace": cache_namespace,
            "sample_index": sample_index,
            "seed": seed,
            "provider_mismatch": provider_mismatch,
        }
        append_call_log(logs_dir, run_id, record)
        append_ledger(
            logs_dir,
            call_id=call_id,
            run_id=run_id,
            task_id=task_id,
            cost_usd=float(cost),
            timestamp=record["timestamp_utc"],
        )
        PROCESS_CACHE_INDEX.update_after_success(logs_dir, record)

        if provider_mismatch and strict_provider:
            raise ProviderMismatchError(
                f"provider_mismatch: requested={provider_requested!r} served={provider_served!r}"
            )

        return {
            "response_raw": response_json,
            "usage": usage,
            "cost_usd": float(cost),
            "cost_source": cost_source,
            "cache_hit": False,
            "call_id": call_id,
            "provider_served": provider_served,
            "model_served": model_served,
            "log_record": record,
            "provider_mismatch": provider_mismatch,
        }
    finally:
        if owns_client:
            client.close()


def _failed_record(**kwargs: Any) -> dict[str, Any]:
    return {
        "call_id": kwargs["call_id"],
        "run_id": kwargs["run_id"],
        "task_id": kwargs["task_id"],
        "role": kwargs["role"],
        "timestamp_utc": _utc_now(),
        "attempt": kwargs["attempt"],
        "model_alias": kwargs["model_alias"],
        "model_slug_requested": kwargs["slug"],
        "model_served": None,
        "provider_requested": kwargs["provider_order"][0] if kwargs["provider_order"] else None,
        "provider_served": None,
        "request": {
            "messages": kwargs["messages"],
            "tools": kwargs["tools"],
            "params": _request_params(
                temp=kwargs["temp"],
                max_toks=kwargs["max_toks"],
                seed=kwargs["seed"],
                response_format=kwargs["response_format"],
                provider=kwargs["request_payload"]["provider"],
                reasoning=kwargs.get("reasoning"),
                top_p=kwargs.get("top_p"),
                frequency_penalty=kwargs.get("frequency_penalty"),
                presence_penalty=kwargs.get("presence_penalty"),
                stop=kwargs.get("stop"),
            ),
        },
        "response_raw": kwargs["response_json"],
        "usage": {},
        "cost_usd": 0.0,
        "cost_source": None,
        "latency_ms": kwargs["latency_ms"],
        "http_status": kwargs["http_status"],
        "error": kwargs["error"],
        "cache_key": kwargs["cache_key"],
        "cache_hit": False,
        "cache_namespace": kwargs["cache_namespace"],
        "sample_index": kwargs["sample_index"],
        "seed": kwargs["seed"],
        "provider_mismatch": False,
    }


__all__ = ["chat", "BudgetExceeded", "ProviderMismatchError"]
