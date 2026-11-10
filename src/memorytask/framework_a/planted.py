"""A-planted arm: Apollo GPT-4 lapse turns + on-policy report generation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from memorytask.client import chat as default_chat
from memorytask.config import repo_root
from memorytask.fingerprints import prefill_disavowal
from memorytask.framework_a.classify import extract_send_message_body
from memorytask.framework_a.episode import MAX_TOKENS_DEVIATION
from memorytask.framework_a.loader import load_prompt

ChatFn = Callable[..., dict[str, Any]]


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


def planted_path() -> Path:
    return repo_root() / "data" / "planted" / "a_lapse_turns.jsonl"


def load_planted_turns(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or planted_path()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def assemble_planted_messages(
    prompt: dict[str, Any], planted_response: str, amount: int | str
) -> list[dict[str, Any]]:
    """default.json messages + assistant planted turn + user deception_trigger."""
    messages = copy.deepcopy(prompt["messages"])
    messages.append({"role": "assistant", "content": planted_response})
    messages.append(
        {"role": "user", "content": str(prompt["deception_trigger"]).format(amount=amount)}
    )
    return messages


def run_planted_episode(
    *,
    model_alias: str,
    planted: dict[str, Any],
    run_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    sample_index: int = 0,
    chat_fn: ChatFn = default_chat,
    prompt_file: str = "default.json",
) -> dict[str, Any]:
    prompt = load_prompt(prompt_file)
    amount = planted.get("amount")
    planted_response = planted["response"]
    episode_id = f"planted__{model_alias}__{planted.get('index')}__s{sample_index}"

    messages = assemble_planted_messages(prompt, planted_response, amount)

    sampling = dict(
        temperature=float(prompt.get("temperature", 0.9)),
        top_p=float(prompt.get("top_p", 0.95)),
        frequency_penalty=float(prompt.get("frequency_penalty", 0.0)),
        presence_penalty=float(prompt.get("presence_penalty", 0.0)),
        max_tokens=MAX_TOKENS_DEVIATION,
    )

    step2 = chat_fn(
        model_alias,
        messages,
        cache_namespace=cache_namespace,
        run_id=run_id,
        task_id=f"{episode_id}__report",
        role="subject",
        sample_index=sample_index,
        budget_cap_usd=budget_cap_usd,
        strict_provider=True,
        **sampling,
    )
    report_completion = _assistant_text(step2["response_raw"])
    body, found = extract_send_message_body(report_completion)
    report_text = body if found and body is not None else report_completion

    fork_messages = messages + [{"role": "assistant", "content": report_completion}]
    disavow, disavow_hits = prefill_disavowal(report_completion)

    record = {
        "episode_id": episode_id,
        "framework": "A",
        "arm": "planted",
        "model": model_alias,
        "model_alias": model_alias,
        "generator_model": "gpt-4-apollo-planted",
        "fork_model_default": model_alias,
        "planted_index": planted.get("index"),
        "planted_source": planted.get("source_file"),
        "amount": amount,
        "planted_response": planted_response,
        "report_text": report_text,
        "report_completion": report_completion,
        "report_found": found,
        "report_call_id": step2.get("call_id"),
        "prefill_disavowal": disavow,
        "prefill_disavowal_matches": disavow_hits,
        "doubling_down_trigger": prompt.get("doubling_down_trigger"),
        "outcome": "LAPSE_PLANTED",
        "canary": planted.get("canary") or prompt.get("canary"),
        "fork_messages": fork_messages,
    }

    run_dir = repo_root() / "runs" / run_id
    ep_dir = run_dir / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "episode.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    (ep_dir / "report.txt").write_text(report_completion, encoding="utf-8")
    return record
