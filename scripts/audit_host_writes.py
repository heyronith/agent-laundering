#!/usr/bin/env python3
"""List agent host writes outside /tmp/ws/ (T05b addendum §1). Does not delete."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402

WRITE_PATTERNS = (
    re.compile(
        r"File written successfully to\s+(?P<path>/[^\s\"']+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"write_file\s*\(\s*(?:file_path\s*=\s*)?[\"'](?P<path>/[^\"']+)[\"']",
        re.IGNORECASE,
    ),
    re.compile(
        r"open\(\s*[\"'](?P<path>/[^\"']+)[\"']\s*,\s*[\"'][wa]",
        re.IGNORECASE,
    ),
    re.compile(
        r"Path\(\s*[\"'](?P<path>/[^\"']+)[\"']\s*\)\.(?:write_text|write_bytes|touch)",
        re.IGNORECASE,
    ),
)

LIST_PATTERNS = (
    re.compile(r"list_directory\s*\(\s*(?:dir_path\s*=\s*)?[\"'](?P<path>/[^\"']+)[\"']", re.I),
)


@dataclass
class HostHit:
    path: str
    episode_id: str
    run_id: str
    source: str
    size: int | None = None
    mtime: str | None = None
    exists: bool = False


def _episode_ids(run_id: str) -> list[str]:
    p = repo_root() / "runs" / run_id / "episodes.jsonl"
    if not p.exists():
        return []
    return [
        json.loads(l)["episode_id"]
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def _scan_text(text: str, episode_id: str, run_id: str, source: str) -> list[HostHit]:
    hits: list[HostHit] = []
    seen: set[str] = set()
    for cre in WRITE_PATTERNS:
        for m in cre.finditer(text):
            path = m.group("path")
            if path.startswith("/tmp/ws/"):
                continue
            key = path
            if key in seen:
                continue
            seen.add(key)
            hits.append(HostHit(path=path, episode_id=episode_id, run_id=run_id, source=source))
    return hits


def _enrich(hit: HostHit) -> HostHit:
    p = Path(hit.path)
    try:
        if p.exists() and p.is_file():
            st = p.stat()
            hit.exists = True
            hit.size = int(st.st_size)
            hit.mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        elif p.exists():
            hit.exists = True
            hit.size = None
            st = p.stat()
            hit.mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        pass
    return hit


def audit_run(run_id: str) -> list[HostHit]:
    root = repo_root()
    run = root / "runs" / run_id
    hits: list[HostHit] = []
    for eid in _episode_ids(run_id):
        ep = run / "episodes" / eid
        for name in ("trajectory.json", "agent_stdout.log", "final_answer.txt"):
            path = ep / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hits.extend(_scan_text(text, eid, run_id, f"{name}"))
        # workdir copy stays under runs/ — not a host write outside /tmp/ws from agent cwd,
        # but absolute writes recorded in trajectory still counted above.
    # Dedup by path+episode
    uniq: dict[tuple[str, str], HostHit] = {}
    for h in hits:
        uniq[(h.path, h.episode_id)] = h
    return [_enrich(h) for h in uniq.values()]


def scan_disk_outside_tmp_ws(run_ids: list[str], home: Path | None = None) -> list[HostHit]:
    """Best-effort: files under $HOME mentioned in trajectories as absolute writes.

    Does not walk the whole disk. Only stats paths already extracted from logs.
    """
    del home
    out: list[HostHit] = []
    for rid in run_ids:
        out.extend(audit_run(rid))
    return out


def render_md(hits: list[HostHit], run_ids: list[str]) -> str:
    lines = [
        "# T05b host-write audit",
        "",
        f"Runs: {', '.join(run_ids)}",
        "",
        "Files agents created/changed **outside** `/tmp/ws/` (from trajectory/stdout write evidence).",
        "**Nothing was deleted.**",
        "",
        f"Total distinct path×episode hits: **{len(hits)}**",
        f"Still present on disk: **{sum(1 for h in hits if h.exists)}**",
        "",
        "| run_id | episode_id | path | exists | size | mtime | source |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in sorted(hits, key=lambda x: (x.run_id, x.episode_id, x.path)):
        lines.append(
            f"| {h.run_id} | {h.episode_id} | `{h.path}` | {h.exists} | {h.size} | {h.mtime} | {h.source} |"
        )
    if not hits:
        lines.append("")
        lines.append("_No outside-/tmp/ws/ writes found in trajectory evidence._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", default=["t05_killtest", "t05b_killtest"])
    ap.add_argument("--md-out", default="reports/T05b_host_writes.md")
    ap.add_argument("--json-out", default="reports/T05b_host_writes.json")
    args = ap.parse_args()
    hits = scan_disk_outside_tmp_ws(args.run_ids)
    md = render_md(hits, args.run_ids)
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text(md, encoding="utf-8")
    Path(args.json_out).write_text(
        json.dumps(
            [
                {
                    "run_id": h.run_id,
                    "episode_id": h.episode_id,
                    "path": h.path,
                    "exists": h.exists,
                    "size": h.size,
                    "mtime": h.mtime,
                    "source": h.source,
                }
                for h in hits
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"hits={len(hits)} exists={sum(1 for h in hits if h.exists)} -> {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
