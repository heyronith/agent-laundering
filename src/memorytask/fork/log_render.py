"""Render episode message lists as logs for M3/M4 branches."""

from __future__ import annotations

from typing import Any

TOOL_OUTPUT_MAX = 1500
TRUNC_SUFFIX = " …[truncated]"

ROLE_MAP = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL OUTPUT",
    "tool-response": "TOOL OUTPUT",
    "tool_response": "TOOL OUTPUT",
}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text") or "")
                elif "text" in p:
                    parts.append(str(p.get("text") or ""))
                else:
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def render_log(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "").lower()
        label = ROLE_MAP.get(role, role.upper() or "UNKNOWN")
        text = _content_to_text(msg.get("content"))
        if label == "TOOL OUTPUT" and len(text) > TOOL_OUTPUT_MAX:
            text = text[:TOOL_OUTPUT_MAX] + TRUNC_SUFFIX
        blocks.append(f"[{label}]\n{text}")
    return "\n\n".join(blocks)
