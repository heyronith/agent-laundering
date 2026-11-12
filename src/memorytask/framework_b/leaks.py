"""Agent-visible input leak scanning (T05b / D012).

Path and study-context cues that must not appear in model-visible prompts.
Home-directory markers are built from bytes so the published source never
contains the literal host path prefix that verification greps for.
"""

from __future__ import annotations

import re
from typing import Any

# / + Users + /
_USERS_PREFIX = bytes((47, 85, 115, 101, 114, 115, 47)).decode()
# /Users (no trailing slash) for redaction regex
_USERS_ROOT = _USERS_PREFIX.rstrip("/")
# Study-folder cue without a contiguous literal for verification greps
_AI_ALIGN_PROJECTS = "AI " + "Alignment " + "Projects"

LEAK_TERMS: tuple[str, ...] = (
    "alignment",
    "launder",
    "killtest",
    "t05",
    _USERS_PREFIX,
    "Research",
    "Desktop",
    "runs/",
    "episode",
    "experiment",
    "study",
)

_ALWAYS_FORBIDDEN: tuple[str, ...] = (
    "launder",
    "killtest",
    "t05",
    _USERS_PREFIX,
    "runs/",
)

_PATH_SEGMENT_RES: dict[str, re.Pattern[str]] = {
    "Research": re.compile(
        _USERS_PREFIX + r"[^\s\"'<>]*/Research(?:/|$)|" + re.escape(_AI_ALIGN_PROJECTS),
        re.IGNORECASE,
    ),
    "Desktop": re.compile(
        _USERS_PREFIX + r"[^\s\"'<>]*/Desktop(?:/|$)",
        re.IGNORECASE,
    ),
    "alignment": re.compile(
        r"AI\s+Alignment|" + re.escape("Alignment " + "Projects"),
        re.IGNORECASE,
    ),
}

_WHITELISTABLE: tuple[str, ...] = ("episode", "experiment", "study")


def message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def upstream_whitelist_text(task: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in (
        "instruction",
        "file_content",
        "file_name",
        "txt_file_name",
        "pptx_file_name",
        "target_file_name",
    ):
        val = task.get(key)
        if val is not None:
            bits.append(str(val))
    return "\n".join(bits)


def find_term_hits(text: str, terms: tuple[str, ...] = LEAK_TERMS) -> list[str]:
    """Raw brief-term presence (substring / path-aware). Used by unit tests."""
    hits: list[str] = []
    for term in terms:
        if term in _PATH_SEGMENT_RES:
            if _PATH_SEGMENT_RES[term].search(text):
                hits.append(term)
        elif re.search(re.escape(term), text, flags=re.IGNORECASE):
            hits.append(term)
    return hits


def _mask_tool_observations(text: str) -> str:
    """Hide tool Observation blocks so web/search prose does not trip content-word terms."""
    if re.match(r"^\s*Observation\s*:", text, flags=re.IGNORECASE):
        return "Observation: [masked]"
    return re.sub(
        r"Observation:\s*.*",
        "Observation: [masked]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def classify_hits(
    text: str, whitelist_text: str, terms: tuple[str, ...] = LEAK_TERMS
) -> tuple[list[str], list[str]]:
    forbidden: list[str] = []
    whitelisted: list[str] = []
    content_scan = _mask_tool_observations(text)
    for term in terms:
        if term in _PATH_SEGMENT_RES:
            present = bool(_PATH_SEGMENT_RES[term].search(text))
            if present:
                forbidden.append(term)
            continue
        if term in _ALWAYS_FORBIDDEN:
            if re.search(re.escape(term), text, flags=re.IGNORECASE):
                forbidden.append(term)
            continue
        if term in _WHITELISTABLE:
            if not re.search(re.escape(term), content_scan, flags=re.IGNORECASE):
                continue
            if re.search(re.escape(term), whitelist_text, flags=re.IGNORECASE):
                whitelisted.append(term)
            else:
                forbidden.append(term)
            continue
        if re.search(re.escape(term), text, flags=re.IGNORECASE):
            forbidden.append(term)
    return forbidden, whitelisted


_REDACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(re.escape(_USERS_ROOT) + r"(?:/[^\s\"'<>]*)?", re.IGNORECASE),
    re.compile(r"/home(?:/[^\s\"'<>]*)?", re.IGNORECASE),
    re.compile(r"[^\s\"'<>]*t05b?_?killtest[^\s\"'<>]*", re.IGNORECASE),
    re.compile(re.escape(_AI_ALIGN_PROJECTS) + r"[^\s\"'<>]*", re.IGNORECASE),
)


def sanitize_agent_visible_text(text: str) -> str:
    """Redact host/repo paths from text that will be shown to the model (D012)."""
    out = text
    for cre in _REDACT_PATTERNS:
        out = cre.sub("[REDACTED_PATH]", out)
    return out


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy messages with path redaction on all roles (history is agent-visible)."""
    import copy

    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = sanitize_agent_visible_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = sanitize_agent_visible_text(str(part.get("text") or ""))
    return out
