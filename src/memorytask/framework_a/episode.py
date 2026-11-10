"""Run one Framework A episode (trade → optional report → optional doubling)."""

from __future__ import annotations

import copy
from typing import Any, Callable

from memorytask.client import chat
from memorytask.framework_a.classify import (
    classify_misalignment,
    extract_amount,
    extract_send_message_body,
    find_eval_suspicion,
)
from memorytask.framework_a.loader import load_prompt

ChatFn = Callable[..., dict[str, Any]]

MAX_TOKENS_DEVIATION = 1024


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


def run_episode(
    *,
    model_alias: str,
    prompt_file: str,
    sample_index: int,
    run_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    chat_fn: ChatFn = chat,
    prompt_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = prompt_data or load_prompt(prompt_file)
    rel_path = prompt.get("_path") or prompt_file
    episode_id = f"{model_alias}__{rel_path.replace('/', '__')}__{sample_index}"

    temperature = float(prompt.get("temperature", 0.9))
    top_p = float(prompt.get("top_p", 0.95))
    frequency_penalty = float(prompt.get("frequency_penalty", 0.0))
    presence_penalty = float(prompt.get("presence_penalty", 0.0))
    misalignment_string = prompt["misalignment_string"]

    sampling = dict(
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=MAX_TOKENS_DEVIATION,
    )

    messages = copy.deepcopy(prompt["messages"])
    turns: list[dict[str, Any]] = []

    # Step 1: trade decision
    step1 = chat_fn(
        model_alias,
        messages,
        cache_namespace=cache_namespace,
        run_id=run_id,
        task_id=f"{episode_id}__step1",
        role="subject",
        sample_index=sample_index,
        budget_cap_usd=budget_cap_usd,
        strict_provider=True,
        **sampling,
    )
    trade_text = _assistant_text(step1["response_raw"])
    turns.append(
        {
            "step": 1,
            "name": "trade",
            "call_id": step1["call_id"],
            "text": trade_text,
            "cache_hit": step1.get("cache_hit"),
        }
    )

    labels = classify_misalignment(trade_text, misalignment_string)
    amount, amount_parsed = extract_amount(trade_text)

    report_text = None
    report_found = None
    doubling_text = None
    doubling_found = None

    if labels["misaligned_original"]:
        # Step 2: report to manager
        messages = messages + [
            {"role": "assistant", "content": trade_text},
            {
                "role": "user",
                "content": str(prompt["deception_trigger"]).format(amount=amount),
            },
        ]
        step2 = chat_fn(
            model_alias,
            messages,
            cache_namespace=cache_namespace,
            run_id=run_id,
            task_id=f"{episode_id}__step2",
            role="subject",
            sample_index=sample_index,
            budget_cap_usd=budget_cap_usd,
            strict_provider=True,
            **sampling,
        )
        report_completion = _assistant_text(step2["response_raw"])
        body, found = extract_send_message_body(report_completion)
        if found and body is not None:
            report_text = body
            report_found = True
        else:
            report_text = report_completion
            report_found = False
        turns.append(
            {
                "step": 2,
                "name": "report",
                "call_id": step2["call_id"],
                "text": report_completion,
                "cache_hit": step2.get("cache_hit"),
            }
        )

        # Step 3: doubling down (only if trigger exists AND report found)
        if prompt.get("doubling_down_trigger") and report_found:
            messages = messages + [
                {"role": "assistant", "content": report_completion},
                {"role": "user", "content": str(prompt["doubling_down_trigger"])},
            ]
            step3 = chat_fn(
                model_alias,
                messages,
                cache_namespace=cache_namespace,
                run_id=run_id,
                task_id=f"{episode_id}__step3",
                role="subject",
                sample_index=sample_index,
                budget_cap_usd=budget_cap_usd,
                strict_provider=True,
                **sampling,
            )
            doubling_completion = _assistant_text(step3["response_raw"])
            dbody, dfound = extract_send_message_body(doubling_completion)
            if dfound and dbody is not None:
                doubling_text = dbody
                doubling_found = True
            else:
                doubling_text = doubling_completion
                doubling_found = False
            turns.append(
                {
                    "step": 3,
                    "name": "doubling",
                    "call_id": step3["call_id"],
                    "text": doubling_completion,
                    "cache_hit": step3.get("cache_hit"),
                }
            )

    suspicion = find_eval_suspicion([t["text"] for t in turns])

    return {
        "episode_id": episode_id,
        "model_alias": model_alias,
        "prompt_file": rel_path,
        "sample_index": sample_index,
        "cache_namespace": cache_namespace,
        "run_id": run_id,
        "sampling": sampling,
        "misalignment_string": misalignment_string,
        "misaligned_original": labels["misaligned_original"],
        "misaligned_normalized": labels["misaligned_normalized"],
        "label_disagreement": labels["label_disagreement"],
        "amount": amount,
        "amount_parsed": amount_parsed,
        "report_text": report_text,
        "report_found": report_found,
        "doubling_text": doubling_text,
        "doubling_found": doubling_found,
        "turns": turns,
        **suspicion,
    }
