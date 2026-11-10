#!/usr/bin/env python3
"""T04 MiniMax smoke (a)+(b) only — namespace t04_smoke_minimax."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from memorytask.calllog import read_jsonl  # noqa: E402
from memorytask.client import chat  # noqa: E402
from memorytask.config import repo_root  # noqa: E402

RUN_ID = "t04_smoke_minimax"
CACHE_NAMESPACE = "t04_smoke_minimax"
BUDGET_CAP = 0.50
ALIAS = "minimax"

WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _assistant_text(response_raw: dict) -> str:
    choices = response_raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            (p.get("text") or "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def main() -> int:
    load_dotenv(ROOT / ".env")
    results = {}

    plain = chat(
        ALIAS,
        [{"role": "user", "content": "Reply with exactly the word PONG."}],
        run_id=RUN_ID,
        task_id=f"{ALIAS}_plain",
        role="subject",
        budget_cap_usd=BUDGET_CAP,
        cache_namespace=CACHE_NAMESPACE,
        strict_provider=True,
        temperature=0.0,
        max_tokens=64,
        sample_index=0,
    )
    results["plain"] = {
        "text": _assistant_text(plain["response_raw"]),
        "provider_served": plain.get("provider_served"),
        "provider_requested": plain.get("log_record", {}).get("provider_requested")
        if plain.get("log_record")
        else None,
        "call_id": plain.get("call_id"),
        "cost_usd": plain.get("cost_usd"),
        "reasoning_requested": {"effort": "low"},
    }

    tool = chat(
        ALIAS,
        [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}],
        tools=WEATHER_TOOL,
        run_id=RUN_ID,
        task_id=f"{ALIAS}_tool",
        role="subject",
        budget_cap_usd=BUDGET_CAP,
        cache_namespace=CACHE_NAMESPACE,
        strict_provider=True,
        temperature=0.0,
        max_tokens=256,
        sample_index=1,
    )
    raw = tool["response_raw"]
    msg = ((raw.get("choices") or [{}])[0].get("message") or {})
    results["tool"] = {
        "tool_calls": msg.get("tool_calls"),
        "text": _assistant_text(raw),
        "provider_served": tool.get("provider_served"),
        "call_id": tool.get("call_id"),
        "cost_usd": tool.get("cost_usd"),
    }

    # Summarize from logs
    logs = read_jsonl(repo_root() / "logs" / f"{RUN_ID}.jsonl")
    for r in logs:
        if r.get("task_id") == f"{ALIAS}_plain":
            results["plain"]["provider_requested"] = r.get("provider_requested")
            results["plain"]["provider_served"] = r.get("provider_served")
            results["plain"]["reasoning_in_request"] = ((r.get("request") or {}).get("params") or {}).get(
                "reasoning"
            )
        if r.get("task_id") == f"{ALIAS}_tool":
            results["tool"]["provider_requested"] = r.get("provider_requested")
            results["tool"]["provider_served"] = r.get("provider_served")

    out = repo_root() / "reports" / "T04_minimax_smoke.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))
    ok = (
        results["plain"].get("provider_served") == results["plain"].get("provider_requested")
        and results["tool"].get("tool_calls")
    )
    print("SMOKE", "PASS" if ok else "FAIL", "->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
