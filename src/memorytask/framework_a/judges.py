"""Judge calls for Framework A report deception and doubling-down labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from memorytask.client import chat
from memorytask.config import repo_root

ChatFn = Callable[..., dict[str, Any]]

REPORT_LABELS = {"HONEST", "DECEPTIVE", "AMBIGUOUS", "NO_REPORT"}
DOUBLING_LABELS = {"ADMITS", "DENIES", "EVASIVE", "NO_REPLY"}


def judge_prompt_path(name: str) -> Path:
    return repo_root() / "prompts" / "judges" / name


def load_judge_template(name: str) -> str:
    return judge_prompt_path(name).read_text(encoding="utf-8")


def fill_template(template: str, **fields: str) -> str:
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value)
    return out


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_judge_json(text: str, allowed: set[str]) -> dict[str, Any] | None:
    cleaned = strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract first JSON object
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    label = obj.get("label")
    reason = obj.get("reason")
    if label not in allowed or not isinstance(reason, str):
        return None
    return {"label": label, "reason": reason}


def _assistant_text(response_raw: dict[str, Any]) -> str:
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


def judge_once(
    *,
    model_alias: str,
    prompt_text: str,
    run_id: str,
    task_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int,
    allowed_labels: set[str],
    chat_fn: ChatFn = chat,
) -> dict[str, Any]:
    result = chat_fn(
        model_alias,
        [{"role": "user", "content": prompt_text}],
        cache_namespace=cache_namespace,
        run_id=run_id,
        task_id=task_id,
        role="judge",
        sample_index=sample_index,
        budget_cap_usd=budget_cap_usd,
        temperature=0.0,
        # Use models.yaml default (2048 for judges); do not shrink for credit preflight.
        strict_provider=True,
    )
    text = _assistant_text(result["response_raw"])
    parsed = parse_judge_json(text, allowed_labels)
    return {
        "call_id": result["call_id"],
        "raw_text": text,
        "parsed": parsed,
        "cache_hit": result.get("cache_hit"),
        "sample_index": sample_index,
    }


def judge_with_retry(
    *,
    model_alias: str,
    prompt_text: str,
    run_id: str,
    task_id_prefix: str,
    cache_namespace: str,
    budget_cap_usd: float,
    base_sample_index: int,
    allowed_labels: set[str],
    chat_fn: ChatFn = chat,
) -> dict[str, Any]:
    first = judge_once(
        model_alias=model_alias,
        prompt_text=prompt_text,
        run_id=run_id,
        task_id=f"{task_id_prefix}__try0",
        cache_namespace=cache_namespace,
        budget_cap_usd=budget_cap_usd,
        sample_index=base_sample_index,
        allowed_labels=allowed_labels,
        chat_fn=chat_fn,
    )
    attempts = [first]
    if first["parsed"] is not None:
        return {
            "label": first["parsed"]["label"],
            "reason": first["parsed"]["reason"],
            "call_ids": [first["call_id"]],
            "attempts": attempts,
            "parse_fail": False,
        }

    second = judge_once(
        model_alias=model_alias,
        prompt_text=prompt_text,
        run_id=run_id,
        task_id=f"{task_id_prefix}__try1",
        cache_namespace=cache_namespace,
        budget_cap_usd=budget_cap_usd,
        sample_index=base_sample_index + 1,
        allowed_labels=allowed_labels,
        chat_fn=chat_fn,
    )
    attempts.append(second)
    if second["parsed"] is not None:
        return {
            "label": second["parsed"]["label"],
            "reason": second["parsed"]["reason"],
            "call_ids": [first["call_id"], second["call_id"]],
            "attempts": attempts,
            "parse_fail": False,
        }
    return {
        "label": "PARSE_FAIL",
        "reason": "Judge output could not be parsed into a valid label after retry.",
        "call_ids": [first["call_id"], second["call_id"]],
        "attempts": attempts,
        "parse_fail": True,
    }


def judge_report(
    report_text: str,
    *,
    model_alias: str,
    run_id: str,
    episode_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int,
    chat_fn: ChatFn = chat,
) -> dict[str, Any]:
    template = load_judge_template("a_report_deception.txt")
    prompt = fill_template(template, report=report_text)
    out = judge_with_retry(
        model_alias=model_alias,
        prompt_text=prompt,
        run_id=run_id,
        task_id_prefix=f"{episode_id}__judge_report__{model_alias}",
        cache_namespace=cache_namespace,
        budget_cap_usd=budget_cap_usd,
        base_sample_index=sample_index,
        allowed_labels=REPORT_LABELS,
        chat_fn=chat_fn,
    )
    out["kind"] = "report"
    out["model_alias"] = model_alias
    out["episode_id"] = episode_id
    return out


def judge_doubling(
    reply_text: str,
    *,
    model_alias: str,
    run_id: str,
    episode_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int,
    chat_fn: ChatFn = chat,
) -> dict[str, Any]:
    template = load_judge_template("a_doubling_down.txt")
    prompt = fill_template(template, reply=reply_text)
    out = judge_with_retry(
        model_alias=model_alias,
        prompt_text=prompt,
        run_id=run_id,
        task_id_prefix=f"{episode_id}__judge_doubling__{model_alias}",
        cache_namespace=cache_namespace,
        budget_cap_usd=budget_cap_usd,
        base_sample_index=sample_index,
        allowed_labels=DOUBLING_LABELS,
        chat_fn=chat_fn,
    )
    out["kind"] = "doubling"
    out["model_alias"] = model_alias
    out["episode_id"] = episode_id
    return out
