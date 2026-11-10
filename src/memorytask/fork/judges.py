"""Branch judges: memory_note (M1–M4), recognition (K), plus A report/D wrappers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from memorytask.client import chat as default_chat
from memorytask.config import repo_root
from memorytask.framework_a.judges import (
    judge_doubling,
    judge_report,
    parse_judge_json,
    strip_code_fences,
)

ChatFn = Callable[..., dict[str, Any]]

MEMORY_LABELS = {"FAITHFUL", "ABSTRACTED", "EUPHEMIZED", "OMITTED", "FALSE"}
RECOGNITION_LABELS = {"RECOGNIZED", "PARTIAL", "NOT_RECOGNIZED"}

# Model-family map for same-family assertion (brief §6).
MODEL_FAMILY: dict[str, str] = {
    "deepseek": "deepseek",
    "glm": "zhipu",
    "luna": "openai",
    "minimax": "minimax",
    "judge_primary": "google",
    "judge_secondary": "anthropic",
}


def assert_judge_not_same_family(fork_model: str, judge_model: str) -> None:
    f_fam = MODEL_FAMILY.get(fork_model)
    j_fam = MODEL_FAMILY.get(judge_model)
    if f_fam is None or j_fam is None:
        return
    if f_fam == j_fam:
        raise AssertionError(
            f"Judge {judge_model} (family={j_fam}) same family as fork_model {fork_model}"
        )


def _assistant_text(response_raw: dict[str, Any]) -> str:
    choices = response_raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def load_template(name: str) -> str:
    return (repo_root() / "prompts" / "judges" / name).read_text(encoding="utf-8")


def fill_template(template: str, **fields: str) -> str:
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value)
    return out


def parse_labeled_json(text: str, allowed: set[str]) -> dict[str, Any] | None:
    cleaned = strip_code_fences(text or "")
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
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
    if label not in allowed:
        return None
    evidence = obj.get("evidence", "")
    reason = obj.get("reason", "")
    if not isinstance(evidence, str) or not isinstance(reason, str):
        return None
    return {"label": label, "evidence": evidence, "reason": reason}


def _judge_with_retry(
    *,
    model_alias: str,
    prompt: str,
    allowed: set[str],
    run_id: str,
    task_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int,
    chat_fn: ChatFn,
) -> dict[str, Any]:
    last_raw = ""
    call_ids: list[str] = []
    for attempt in range(2):
        result = chat_fn(
            model_alias=model_alias,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            cache_namespace=cache_namespace,
            run_id=run_id,
            task_id=f"{task_id}__try{attempt}",
            role="judge",
            sample_index=sample_index + attempt,
            budget_cap_usd=budget_cap_usd,
        )
        call_ids.append(result.get("call_id"))
        last_raw = _assistant_text(result.get("response_raw") or {})
        parsed = parse_labeled_json(last_raw, allowed)
        if parsed is not None:
            return {
                "label": parsed["label"],
                "evidence": parsed["evidence"],
                "reason": parsed["reason"],
                "parse_status": "ok",
                "raw_text": last_raw,
                "call_ids": call_ids,
                "model_alias": model_alias,
                "attempt": attempt,
            }
    return {
        "label": "PARSE_FAIL",
        "evidence": "",
        "reason": "failed to parse after retry",
        "parse_status": "PARSE_FAIL",
        "raw_text": last_raw,
        "call_ids": call_ids,
        "model_alias": model_alias,
        "attempt": 1,
    }


def judge_memory_note(
    *,
    note: str,
    event_description: str,
    key_facts: str,
    model_alias: str,
    fork_model: str,
    run_id: str,
    branch_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int = 0,
    chat_fn: ChatFn = default_chat,
) -> dict[str, Any]:
    assert_judge_not_same_family(fork_model, model_alias)
    prompt = fill_template(
        load_template("memory_note.txt"),
        event_description=event_description,
        note=note,
        key_facts=key_facts,
    )
    out = _judge_with_retry(
        model_alias=model_alias,
        prompt=prompt,
        allowed=MEMORY_LABELS,
        run_id=run_id,
        task_id=f"{branch_id}__memory_note__{model_alias}",
        cache_namespace=cache_namespace,
        budget_cap_usd=budget_cap_usd,
        sample_index=sample_index,
        chat_fn=chat_fn,
    )
    out["kind"] = "memory_note"
    out["branch_id"] = branch_id
    return out


def judge_recognition(
    *,
    answer: str,
    event_description: str,
    key_facts: str,
    model_alias: str,
    fork_model: str,
    run_id: str,
    branch_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int = 0,
    chat_fn: ChatFn = default_chat,
) -> dict[str, Any]:
    assert_judge_not_same_family(fork_model, model_alias)
    prompt = fill_template(
        load_template("recognition.txt"),
        event_description=event_description,
        answer=answer,
        key_facts=key_facts,
    )
    out = _judge_with_retry(
        model_alias=model_alias,
        prompt=prompt,
        allowed=RECOGNITION_LABELS,
        run_id=run_id,
        task_id=f"{branch_id}__recognition__{model_alias}",
        cache_namespace=cache_namespace,
        budget_cap_usd=budget_cap_usd,
        sample_index=sample_index,
        chat_fn=chat_fn,
    )
    out["kind"] = "recognition"
    out["branch_id"] = branch_id
    return out
