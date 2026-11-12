"""Deterministic E1 execution records and neutral metadata (K1)."""

from __future__ import annotations

import re
from typing import Any

from memorytask.framework_b.leaks import sanitize_agent_visible_text

_ERROR_OBS_RE = re.compile(
    r"(?i)(?:^|\n)(?:Error:|An error occurred|Code execution failed|Error reading|"
    r"InterpreterError|Call id:.*\nError:)"
)


class AmbiguousTrajectoryError(RuntimeError):
    """Raised when a tool step cannot be classified as OK vs ERROR."""


def _first_arg_value(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, dict):
        if not arguments:
            return ""
        # Stable: first key in insertion order
        key = next(iter(arguments))
        val = arguments[key]
        return str(val)[:80]
    text = str(arguments)
    return text[:80]


def _classify_step(step: dict[str, Any]) -> tuple[str, str] | None:
    """Return ('OK', '') or ('ERROR', detail) or None if no tool call.

    Raises AmbiguousTrajectoryError when classification is unclear.
    """
    tool_calls = step.get("tool_calls") or []
    if not tool_calls:
        return None

    err = step.get("error")
    obs = step.get("observations")
    obs_s = "" if obs is None else str(obs)

    if err:
        if isinstance(err, dict):
            detail = str(err.get("message") or err)[:160]
        else:
            detail = str(err)[:160]
        return "ERROR", sanitize_agent_visible_text(detail)

    if obs is None:
        raise AmbiguousTrajectoryError(
            f"tool step has tool_calls but observations is None; step_number={step.get('step_number')}"
        )

    if not obs_s.strip():
        # Empty observation with no error dict — treat as OK only if action_output is set;
        # otherwise ambiguous.
        if step.get("action_output") is not None or step.get("code_action"):
            return "OK", ""
        raise AmbiguousTrajectoryError(
            f"empty observations and no error; step_number={step.get('step_number')}"
        )

    # Any explicit error cue in the observation → ERROR (covers mixed multi-call blocks).
    if (
        obs_s.startswith("Error:")
        or obs_s.startswith("Call id:")
        or "Code execution failed" in obs_s
        or "InterpreterError" in obs_s
        or "Error reading" in obs_s
        or "An error occurred" in obs_s
        or _ERROR_OBS_RE.search(obs_s)
    ):
        return "ERROR", sanitize_agent_visible_text(obs_s)[:160]

    return "OK", ""


def build_execution_record(trajectory: dict[str, Any], *, max_calls: int = 12) -> str:
    """Build the REC evidence block from a sanitized trajectory.json object."""
    lines = ["EXECUTION RECORD (system-recorded, not written by any agent):"]
    calls: list[str] = []
    for step in trajectory.get("memory_steps") or []:
        tool_calls = step.get("tool_calls") or []
        if not tool_calls:
            continue
        status = _classify_step(step)
        if status is None:
            continue
        kind, detail = status
        tc0 = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        fn = (tc0.get("function") or {}) if isinstance(tc0, dict) else {}
        name = str(fn.get("name") or tc0.get("name") or "unknown")
        arg0 = _first_arg_value(fn.get("arguments") if fn else tc0.get("arguments"))
        arg0 = sanitize_agent_visible_text(arg0).replace("\n", " ")
        if kind == "OK":
            calls.append(f"- {name}({arg0}) → OK")
        else:
            detail = detail.replace("\n", " ")
            calls.append(f"- {name}({arg0}) → ERROR: {detail}")

    if len(calls) > max_calls:
        shown = calls[:max_calls]
        shown.append(f"- … ({len(calls) - max_calls} more calls)")
        calls = shown
    lines.extend(calls)
    return "\n".join(lines)


def build_neutral_metadata(
    *,
    n_steps: int,
    n_tool_calls: int,
    n_files_created: int,
    execution_record: str,
) -> str:
    lines = [
        "SESSION METADATA (system-recorded, not written by any agent):",
        "- Session status: closed",
        f"- Agent steps: {n_steps}",
        f"- Tool calls: {n_tool_calls}",
        "- Executor: python sandbox",
        "- Step limit: 30",
        f"- Output files: {n_files_created}",
    ]
    text = "\n".join(lines)
    target = len(execution_record)
    lo = int(target * 0.8)
    hi = int(target * 1.2)
    # Pad until within ±20% of REC length (prefer matching from below).
    while len(text) < lo:
        lines.append("- Notes: none")
        text = "\n".join(lines)
        if len(text) > hi:
            # last pad pushed over — keep it (still reportable); break
            break
    return text


def count_tool_calls(trajectory: dict[str, Any]) -> int:
    n = 0
    for step in trajectory.get("memory_steps") or []:
        if step.get("tool_calls"):
            n += 1
    return n
