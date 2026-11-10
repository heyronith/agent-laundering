#!/usr/bin/env python3
"""T01 smoke test: plain chat, tool call, and judge JSON for configured models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import read_jsonl  # noqa: E402
from memorytask.client import chat  # noqa: E402
from memorytask.config import get_blocked_models, get_models, repo_root  # noqa: E402

RUN_ID = "t01_smoke"
CACHE_NAMESPACE = "t01_smoke"
BUDGET_CAP = 1.00

WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
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
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _tool_calls(response_raw: dict) -> list[dict]:
    choices = response_raw.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    return message.get("tool_calls") or []


def _parse_json_content(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _summarize_from_logs(logs_dir: Path, task_id: str) -> dict:
    rows = [r for r in read_jsonl(logs_dir / f"{RUN_ID}.jsonl") if r.get("task_id") == task_id]
    cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    success = [r for r in rows if r.get("error") is None and r.get("http_status") == 200]
    last = success[-1] if success else (rows[-1] if rows else {})
    usage = last.get("usage") or {}
    params = ((last.get("request") or {}).get("params") or {})
    return {
        "served_provider": last.get("provider_served"),
        "served_model": last.get("model_served"),
        "provider_requested": last.get("provider_requested"),
        "provider_mismatch": last.get("provider_mismatch"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "reasoning_requested": params.get("reasoning"),
        "cost_usd": cost,
        "cost_source": last.get("cost_source"),
        "cache_hit": last.get("cache_hit"),
        "cache_namespace": last.get("cache_namespace"),
    }


def _chat_common(alias: str, role: str, task_id: str, **extra):
    return dict(
        run_id=RUN_ID,
        task_id=task_id,
        role=role,
        budget_cap_usd=BUDGET_CAP,
        cache_namespace=CACHE_NAMESPACE,
        strict_provider=False,  # only smoke may pass False; still records mismatch
        temperature=0.0,
        **extra,
    )


def run_plain(alias: str, role: str) -> dict:
    task_id = f"{alias}_plain"
    # Use model default max_tokens for judges (2048); keep subjects short.
    extra = {} if role == "judge" else {"max_tokens": 64}
    result = chat(
        alias,
        [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
        **_chat_common(alias, role, task_id, **extra),
    )
    text = _assistant_text(result["response_raw"]).strip()
    passed = text.lower() == "ok"
    return {"pass": passed, "text": text, "task_id": task_id}


def run_tool(alias: str, role: str) -> dict:
    task_id = f"{alias}_tool"
    result = chat(
        alias,
        [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        **_chat_common(alias, role, task_id, tools=WEATHER_TOOL, max_tokens=256),
    )
    calls = _tool_calls(result["response_raw"])
    passed = False
    detail = {"tool_calls": calls}
    for tc in calls:
        fn = tc.get("function") or {}
        if fn.get("name") != "get_weather":
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict) and "city" in args:
            passed = True
            detail["parsed_arguments"] = args
            break
    return {"pass": passed, "detail": detail, "task_id": task_id}


def run_json(alias: str, role: str) -> dict:
    task_id = f"{alias}_json"
    result = chat(
        alias,
        [
            {
                "role": "user",
                "content": "Return a JSON object with keys result (integer 1) and reason (string).",
            }
        ],
        **_chat_common(
            alias,
            role,
            task_id,
            response_format={"type": "json_object"},
        ),
    )
    text = _assistant_text(result["response_raw"])
    obj = _parse_json_content(text)
    passed = bool(obj) and obj.get("result") == 1 and isinstance(obj.get("reason"), str)
    return {"pass": passed, "parsed": obj, "text": text, "task_id": task_id}


def main() -> int:
    root = repo_root()
    logs_dir = root / "logs"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    blocked = get_blocked_models()
    models = get_models()

    results: dict = {
        "run_id": RUN_ID,
        "cache_namespace": CACHE_NAMESPACE,
        "budget_cap_usd": BUDGET_CAP,
        "blocked_models": blocked,
        "models": {},
        "status": "ok",
    }

    if blocked:
        results["status"] = "blocked"
        print("STOP: blocked models present.")
        for alias, entry in blocked.items():
            print(f"  {alias}: {entry}")

    for alias, entry in models.items():
        role = entry["role"]
        model_result: dict = {
            "role": role,
            "slug": entry["slug"],
            "provider_order": entry.get("provider_order"),
            "reasoning": entry.get("reasoning"),
            "checks": {},
        }
        print(f"=== {alias} ({entry['slug']}) provider={entry.get('provider_order')} ===")

        try:
            plain = run_plain(alias, role)
            plain.update(_summarize_from_logs(logs_dir, plain["task_id"]))
            model_result["checks"]["plain"] = plain
            print(f"  plain: {'PASS' if plain['pass'] else 'FAIL'} text={plain.get('text')!r}")
        except Exception as exc:  # noqa: BLE001
            model_result["checks"]["plain"] = {"pass": False, "error": str(exc)}
            print(f"  plain: ERROR {exc}")

        if role == "subject":
            try:
                tool = run_tool(alias, role)
                tool.update(_summarize_from_logs(logs_dir, tool["task_id"]))
                model_result["checks"]["tool"] = tool
                print(f"  tool:  {'PASS' if tool['pass'] else 'FAIL'}")
            except Exception as exc:  # noqa: BLE001
                model_result["checks"]["tool"] = {"pass": False, "error": str(exc)}
                print(f"  tool:  ERROR {exc}")
        else:
            try:
                js = run_json(alias, role)
                js.update(_summarize_from_logs(logs_dir, js["task_id"]))
                model_result["checks"]["json"] = js
                print(f"  json:  {'PASS' if js['pass'] else 'FAIL'}")
            except Exception as exc:  # noqa: BLE001
                model_result["checks"]["json"] = {"pass": False, "error": str(exc)}
                print(f"  json:  ERROR {exc}")

        results["models"][alias] = model_result

    out_path = reports_dir / "T01_smoke_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    failed = False
    for alias, mr in results["models"].items():
        for name, check in mr["checks"].items():
            if not check.get("pass"):
                failed = True
                print(f"FAIL: {alias}.{name}")
            elif check.get("provider_mismatch"):
                failed = True
                print(f"FAIL: {alias}.{name} provider_mismatch")
            elif (
                check.get("served_provider")
                and check.get("provider_requested")
                and check.get("served_provider") != check.get("provider_requested")
            ):
                failed = True
                print(f"FAIL: {alias}.{name} provider_served != provider_requested")
    if blocked:
        failed = True
        print("FAIL: blocked_models present")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
