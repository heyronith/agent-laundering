"""Fork harness: K/M1/M2 in-context; M3/M4 log; D for A."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from memorytask.client import chat as default_chat
from memorytask.config import repo_root
from memorytask.fingerprints import fingerprint, prefill_disavowal
from memorytask.fork.log_render import render_log

ChatFn = Callable[..., dict[str, Any]]

BRANCH_TEMPERATURE = 1.0
BRANCH_MAX_TOKENS = 4096
IN_CONTEXT_BRANCHES = {"K", "M1", "M2", "D"}
LOG_BRANCHES = {"M3", "M4"}


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


def branch_prompt_path(name: str) -> Path:
    return repo_root() / "prompts" / "branches" / f"{name}.txt"


def load_branch_template(name: str) -> str:
    return branch_prompt_path(name).read_text(encoding="utf-8")


def messages_sha256(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fork_messages(
    *,
    branch: str,
    base_messages: list[dict[str, Any]],
    recipient: str | None = None,
    doubling_down_trigger: str | None = None,
    rendered_log: str | None = None,
) -> list[dict[str, Any]]:
    branch = branch.upper()
    if branch in LOG_BRANCHES:
        tmpl = load_branch_template(branch)
        if rendered_log is None:
            rendered_log = render_log(base_messages)
        user_text = tmpl.format(log=rendered_log)
        return [{"role": "user", "content": user_text}]

    if branch == "D":
        if not doubling_down_trigger:
            raise ValueError("D branch requires doubling_down_trigger")
        return list(base_messages) + [{"role": "user", "content": doubling_down_trigger}]

    tmpl = load_branch_template(branch)
    if branch == "M2":
        if not recipient:
            raise ValueError("M2 requires recipient")
        user_text = tmpl.format(recipient=recipient)
    else:
        user_text = tmpl
    return list(base_messages) + [{"role": "user", "content": user_text}]


def fork(
    episode: dict[str, Any],
    branch: str,
    fork_model: str,
    run_id: str,
    namespace: str,
    budget_cap_usd: float,
    *,
    generator_model: str | None = None,
    framework: str = "B",
    base_messages: list[dict[str, Any]] | None = None,
    sample_index: int = 0,
    recipient: str | None = None,
    doubling_down_trigger: str | None = None,
    rendered_log: str | None = None,
    chat_fn: ChatFn = default_chat,
) -> dict[str, Any]:
    branch = branch.upper()
    generator_model = generator_model or episode.get("model") or episode.get("model_alias") or fork_model
    episode_id = episode.get("episode_id") or "unknown"
    if base_messages is None:
        raise ValueError("base_messages required (fork-point message list)")

    if framework.upper() == "B":
        recipient = recipient or "the user who gave you this task"
    else:
        recipient = recipient or "Amy"

    messages = build_fork_messages(
        branch=branch,
        base_messages=base_messages,
        recipient=recipient,
        doubling_down_trigger=doubling_down_trigger,
        rendered_log=rendered_log,
    )
    sha = messages_sha256(messages)
    branch_id = f"{episode_id}__{fork_model}__{branch}__s{sample_index}"

    result = chat_fn(
        model_alias=fork_model,
        messages=messages,
        temperature=BRANCH_TEMPERATURE,
        max_tokens=BRANCH_MAX_TOKENS,
        cache_namespace=namespace,
        run_id=run_id,
        task_id=branch_id,
        role="subject",
        sample_index=sample_index,
        budget_cap_usd=budget_cap_usd,
    )
    output_text = _assistant_text(result.get("response_raw") or {})
    fps = fingerprint(output_text)
    disavow, disavow_hits = prefill_disavowal(output_text)

    return {
        "branch_id": branch_id,
        "episode_id": episode_id,
        "framework": framework.upper(),
        "type": episode.get("task_type") if framework.upper() == "B" else "A",
        "outcome": episode.get("outcome"),
        "arm": episode.get("arm"),
        "branch": branch,
        "generator_model": generator_model,
        "fork_model": fork_model,
        "template_id": branch,
        "sha256": sha,
        "call_id": result.get("call_id"),
        "output_text": output_text,
        "fp_A": fps["fp_A"],
        "fp_A_matches": fps["fp_A_matches"],
        "fp_B": fps["fp_B"],
        "fp_B_matches": fps["fp_B_matches"],
        "eval_suspicion": fps["eval_suspicion"],
        "eval_suspicion_matches": fps["eval_suspicion_matches"],
        "prefill_disavowal": disavow,
        "prefill_disavowal_matches": disavow_hits,
        "cache_hit": result.get("cache_hit"),
        "provider_served": result.get("provider_served"),
        "input_message_count": len(messages),
    }


def load_b_fork_messages(run_id: str, episode_id: str) -> list[dict[str, Any]]:
    """B fork point: last_step_messages from trajectory.json + final assistant turn."""
    root = repo_root()
    ep_dir = root / "runs" / run_id / "episodes" / episode_id
    traj = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
    msgs = copy.deepcopy(traj.get("last_step_messages") or [])
    final = (ep_dir / "final_answer.txt").read_text(encoding="utf-8")
    msgs.append({"role": "assistant", "content": final})
    return msgs
