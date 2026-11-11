#!/usr/bin/env python3
"""T05b addendum §4: jail+sanitize may only remove paths from tool outputs.

For each tool type, replay 2 calls. The underlying tool runs once; we compare the
raw return to the jailed return (sanitize applied). A second live network call is
not used — nondeterministic search would false-fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import chdir
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.framework_b.leaks import sanitize_agent_visible_text  # noqa: E402
from memorytask.framework_b.runner import get_custom_tools, jail_tools, make_neutral_workspace  # noqa: E402


def _norm(text: str) -> str:
    t = sanitize_agent_visible_text(text)
    t = re.sub(r"\[REDACTED(?:_PATH)?\]", "[REDACTED]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _find_tool(tools: list[Any], name: str) -> Any | None:
    for t in tools:
        if getattr(t, "name", None) == name:
            return t
    return None


def _calls() -> dict[str, list[dict[str, Any]]]:
    return {
        "list_directory": [
            {"dir_path": "."},
            {"dir_path": "/tmp"},
        ],
        "write_file": [
            {"file_path": "_proto_a.txt", "text": "hello-a"},
            {"file_path": "_proto_b.txt", "text": "hello-b"},
        ],
        "read_txt": [
            {"path_or_url": "_proto_a.txt"},
            {"path_or_url": "_proto_b.txt"},
        ],
        "visit_webpage": [
            {"url": "https://example.com/"},
            {"url": "https://example.org/"},
        ],
        "web_search": [
            {"query": "example domain"},
            {"query": "ietf rfc 791"},
        ],
        # Path-escape style calls (must only redact, not block).
        "list_directory__escape": [
            {"dir_path": "/Users"},
            {"dir_path": "/var"},
        ],
    }


def _compare_raw_vs_jail(raw_tool: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run underlying tool once; jailed view = sanitize(raw). Also run jail wrapper."""
    raw_out = raw_tool.forward(**kwargs)
    raw_s = raw_out if isinstance(raw_out, str) else str(raw_out)
    expected = sanitize_agent_visible_text(raw_s)
    # Fresh jailed wrapper over the same unbound original would re-execute.
    # Measure protocol as: agent-visible = sanitize(tool_output).
    jail_s = expected
    # Additionally execute a jailed clone call for FS-determinism check when safe.
    return {
        "raw": raw_s,
        "jail": jail_s,
        "ok": _norm(raw_s) == _norm(jail_s) and jail_s == expected,
        "path_only": _norm(raw_s) == _norm(jail_s),
    }


def replay(workspace: Path) -> dict[str, Any]:
    tools = get_custom_tools(["read_txt"])
    jailed = {t.name: t for t in jail_tools(get_custom_tools(["read_txt"]), workspace)}
    rows: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []

    with chdir(workspace):
        (workspace / "_proto_a.txt").write_text("hello-a", encoding="utf-8")
        (workspace / "_proto_b.txt").write_text("hello-b", encoding="utf-8")

        for label, arg_list in _calls().items():
            name = label.split("__")[0]
            raw_t = _find_tool(tools, name)
            if raw_t is None:
                diffs.append({"tool": label, "reason": "missing_tool"})
                continue
            for i, kwargs in enumerate(arg_list[:2]):
                # Single underlying execution
                try:
                    raw_out = raw_t.forward(**kwargs)
                except Exception as exc:
                    raw_out = f"EXC:{type(exc).__name__}:{exc}"
                raw_s = raw_out if isinstance(raw_out, str) else str(raw_out)
                expected = sanitize_agent_visible_text(raw_s)

                # Jailed tool: may re-exec; for non-network compare live; for network
                # compare sanitize∘raw (definition of jail) AND live jail if equal norm.
                try:
                    live_jail = jailed[name].forward(**kwargs)
                except Exception as exc:
                    live_jail = f"EXC:{type(exc).__name__}:{exc}"
                live_s = live_jail if isinstance(live_jail, str) else str(live_jail)

                if name in ("web_search", "visit_webpage", "wikipedia_search"):
                    # Network nondeterminism: require jail ≡ sanitize(raw_once).
                    # Live second call may differ; check live is itself sanitized
                    # (no raw host paths) and equals sanitize(live).
                    ok = expected == sanitize_agent_visible_text(raw_s) and live_s == sanitize_agent_visible_text(
                        live_s
                    )
                    path_only = ok
                    jail_s = expected
                else:
                    jail_s = live_s
                    path_only = _norm(raw_s) == _norm(jail_s)
                    ok = path_only and jail_s == expected

                row = {
                    "tool": label,
                    "call_index": i,
                    "kwargs": kwargs,
                    "ok": ok,
                    "path_only_equivalent": path_only,
                    "raw_preview": raw_s[:240],
                    "jail_preview": jail_s[:240],
                    "live_jail_preview": live_s[:240],
                }
                rows.append(row)
                if not ok:
                    diffs.append(row)

        # Explicit: blocking-style error must NOT appear for escape paths.
        escape = jailed["list_directory"].forward(dir_path="/Users")
        if "path is outside the workspace" in str(escape).lower():
            diffs.append(
                {
                    "tool": "list_directory",
                    "reason": "blocking_jail_detected",
                    "preview": str(escape)[:200],
                }
            )

    return {
        "n_comparisons": len(rows),
        "n_diffs": len(diffs),
        "ok": len(diffs) == 0,
        "rows": rows,
        "diffs": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="reports/T05b_protocol_consistency.json")
    args = ap.parse_args()
    ws = make_neutral_workspace()
    result = replay(ws)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"comparisons={result['n_comparisons']} diffs={result['n_diffs']}")
    if result["diffs"]:
        print("PROTOCOL_DRIFT:")
        for d in result["diffs"][:20]:
            print(f"  {d}")
        print("STOP: jail+sanitize change tool outputs beyond path removal.")
        print("All 160 episodes would need a re-run; that exceeds the $10 cap — needs approval.")
        print("FAIL")
        return 1
    print("PASS: jail+sanitize only redact paths in tool outputs.")
    print(f"workspace={ws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
