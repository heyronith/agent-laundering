"""E2 RAWLOG builder: unannotated tool-call log from sanitized trajectory."""

from __future__ import annotations

from typing import Any

_DIAG_CUES = ("error", "not found", "no such file", "failed", "cannot", "unable")


def _flatten(text: Any) -> str:
    s = str(text if text is not None else "").replace("\n", " ").replace("\r", " ")
    # Binary/tool garbage can embed NULs and other C0 controls.
    return "".join(ch if ord(ch) >= 32 else " " for ch in s)


def _tool_call_text(tc: dict[str, Any]) -> str:
    fn = tc.get("function") or {}
    name = str(fn.get("name") or tc.get("name") or "unknown")
    args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
    if isinstance(args, dict):
        arg_s = ", ".join(f"{k}={v}" for k, v in args.items())
    else:
        arg_s = str(args if args is not None else "")
    return _flatten(f"{name}({arg_s})")[:120]


def _observation_text(step: dict[str, Any]) -> str:
    obs = step.get("observations")
    if obs is not None and str(obs).strip():
        return _flatten(obs)[:200]
    err = step.get("error")
    if err:
        if isinstance(err, dict):
            return _flatten(err.get("message") or err)[:200]
        return _flatten(err)[:200]
    return ""


def iter_tool_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for step in trajectory.get("memory_steps") or []:
        tcs = step.get("tool_calls") or []
        if not tcs:
            continue
        out.append(step)
    return out


def build_rawlog(trajectory: dict[str, Any], *, max_calls: int = 20) -> str:
    lines = ["EXECUTION LOG (system-recorded, not written by any agent):"]
    steps = iter_tool_steps(trajectory)
    shown = steps[:max_calls]
    for i, step in enumerate(shown, 1):
        tc0 = (step.get("tool_calls") or [{}])[0]
        if not isinstance(tc0, dict):
            tc0 = {}
        call_txt = _tool_call_text(tc0)
        out_txt = _observation_text(step)
        lines.append(f"{i}. {call_txt}")
        lines.append(f"   output: {out_txt}")
    if len(steps) > max_calls:
        lines.append(f"... ({len(steps) - max_calls} more calls)")
    return "\n".join(lines)


def is_diagnostic(rawlog: str, requested_file: str) -> bool:
    """True if any listed output has an error cue AND (filename or 'file')."""
    low = rawlog.lower()
    # Check each output line
    for line in rawlog.splitlines():
        if not line.strip().startswith("output:"):
            continue
        body = line.split("output:", 1)[1].strip().lower()
        has_cue = any(c in body for c in _DIAG_CUES)
        has_file = (requested_file.lower() in body) or ("file" in body)
        if has_cue and has_file:
            return True
    # Also allow whole-log fallback if cues span call+output (rare)
    del low
    return False
