#!/usr/bin/env python3
"""Scan RAW request.messages in logs/*.jsonl for path/study demand cues (T05b).

Addendum §2: does **not** scan sanitized trajectories. Hits are reported per episode.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402
from memorytask.framework_b.leaks import (  # noqa: E402
    LEAK_TERMS,
    classify_hits,
    message_text,
    upstream_whitelist_text,
)
from memorytask.framework_b.runner import load_tasks  # noqa: E402


def _episode_id_from_task_id(task_id: str | None) -> str:
    if not task_id:
        return ""
    bits = task_id.split("__")
    # episode: model__typeN__idxM__s0
    # branch:  model__typeN__idxM__s0__model__K__s0
    if len(bits) >= 4:
        return "__".join(bits[:4])
    return task_id


def _load_discarded(run_id: str) -> set[str]:
    path = repo_root() / "runs" / run_id / "discarded_calls.jsonl"
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("call_id")
        if cid:
            ids.add(cid)
    return ids


def _whitelist_for(eid: str, cache: dict[int, dict[int, dict[str, Any]]]) -> str:
    bits = eid.split("__")
    try:
        ttype = int(bits[1].replace("type", ""))
        idx = int(bits[2].replace("idx", ""))
    except (IndexError, ValueError):
        return ""
    if ttype not in cache:
        cache[ttype] = {int(t["task_index"]): t for t in load_tasks(ttype)}
    return upstream_whitelist_text(cache[ttype].get(idx, {}))


def _broad_hits(text: str) -> list[str]:
    """Pre-narrowing brief matching: plain case-insensitive substring for every term."""
    hits: list[str] = []
    for term in LEAK_TERMS:
        if re.search(re.escape(term), text, flags=re.IGNORECASE):
            hits.append(term)
    return hits


def _context(text: str, term: str, width: int = 40) -> str:
    m = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if not m:
        # path-aware terms may not literal-match
        return text[:80]
    a = max(0, m.start() - width)
    b = min(len(text), m.end() + width)
    return text[a:b]


def collect_raw_inputs(
    run_id: str,
    *,
    call_roles: tuple[str, ...] | None = ("subject", "fork"),
    message_roles: tuple[str, ...] | None = ("user", "system"),
    all_message_roles: bool = False,
) -> list[dict[str, Any]]:
    """Collect raw request message texts for leak scanning.

    K1: pass all_message_roles=True to scan every message role in the request.
    """
    discarded = _load_discarded(run_id)
    task_cache: dict[int, dict[int, dict[str, Any]]] = {}
    items: list[dict[str, Any]] = []
    for r in iter_run_logs(repo_root() / "logs"):
        if r.get("run_id") != run_id:
            continue
        if call_roles is not None and r.get("role") not in call_roles:
            continue
        cid = r.get("call_id")
        if cid in discarded:
            continue
        eid = _episode_id_from_task_id(r.get("task_id"))
        wl = _whitelist_for(eid, task_cache)
        msgs = (r.get("request") or {}).get("messages") or []
        for i, msg in enumerate(msgs):
            role = msg.get("role")
            if not all_message_roles:
                if message_roles is not None and role not in message_roles:
                    continue
            items.append(
                {
                    "source": f"log:{cid}:{i}:{role}",
                    "call_id": cid,
                    "episode_id": eid,
                    "role_call": r.get("role"),
                    "text": message_text(msg),
                    "whitelist": wl,
                }
            )
    return items


def scan_run(
    run_id: str,
    *,
    call_roles: tuple[str, ...] | None = ("subject", "fork"),
    all_message_roles: bool = False,
) -> dict[str, Any]:
    items = collect_raw_inputs(
        run_id, call_roles=call_roles, all_message_roles=all_message_roles
    )
    forbidden_rows: list[dict[str, Any]] = []
    whitelist_rows: list[dict[str, Any]] = []
    exempted_rows: list[dict[str, Any]] = []
    per_episode: dict[str, list[str]] = defaultdict(list)

    for it in items:
        text = it["text"]
        wl = it.get("whitelist") or ""
        forb, white = classify_hits(text, wl)
        broad = set(_broad_hits(text))
        # Exempted by narrowing / observation-mask / path-aware rules:
        # in broad, not in narrow forbidden, and not upstream-whitelist-only noise we still report.
        narrow_set = set(forb) | set(white)
        for term in sorted(broad - narrow_set):
            # Still skip if whitelist absorbed it as allowed upstream term
            if term in white:
                continue
            # If classify would whitelist episode/experiment/study, broad-only exempt
            exempted_rows.append(
                {
                    "episode_id": it.get("episode_id"),
                    "term": term,
                    "context": _context(text, term, 40),
                    "source": it.get("source"),
                }
            )
        if white:
            whitelist_rows.append(
                {"source": it["source"], "episode_id": it.get("episode_id"), "terms": white}
            )
        if forb:
            per_episode[it["episode_id"]].extend(forb)
            forbidden_rows.append(
                {
                    "source": it["source"],
                    "episode_id": it.get("episode_id"),
                    "terms": forb,
                    "snippet": _context(text, forb[0], 40),
                }
            )

    per_ep_summary = {
        eid: dict(Counter(terms)) for eid, terms in sorted(per_episode.items())
    }
    scan_target = "raw request.messages in logs/*.jsonl"
    if call_roles is None:
        scan_target += " (all call roles"
    else:
        scan_target += f" ({'+'.join(call_roles)}"
    scan_target += "; all message roles" if all_message_roles else "; user+system only"
    scan_target += "; discarded excluded)"
    return {
        "run_id": run_id,
        "scan_target": scan_target,
        "leak_terms": list(LEAK_TERMS),
        "n_inputs_scanned": len(items),
        "n_forbidden": len(forbidden_rows),
        "n_whitelist_hits": len(whitelist_rows),
        "n_exempted_narrowing": len(exempted_rows),
        "forbidden": forbidden_rows,
        "forbidden_per_episode": per_ep_summary,
        "whitelist_hits": whitelist_rows,
        "exempted_by_narrowing": exempted_rows,
        "ok": len(forbidden_rows) == 0,
        "all_message_roles": all_message_roles,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--all-roles",
        action="store_true",
        help="Scan all message roles in raw request.messages (not only user+system)",
    )
    ap.add_argument(
        "--call-roles",
        default=None,
        help="Comma-separated call roles to include (default: subject,fork). "
        "Use 'ALL' for every call role.",
    )
    args = ap.parse_args()
    if args.call_roles == "ALL":
        call_roles = None
    elif args.call_roles:
        call_roles = tuple(x.strip() for x in args.call_roles.split(",") if x.strip())
    else:
        call_roles = ("subject", "fork")
    result = scan_run(
        args.run_id,
        call_roles=call_roles,
        all_message_roles=bool(args.all_roles),
    )
    print(f"run_id={result['run_id']}")
    print(f"scan_target={result['scan_target']}")
    print(f"inputs_scanned={result['n_inputs_scanned']}")
    print(
        f"forbidden={result['n_forbidden']} whitelist_term_hits={result['n_whitelist_hits']} "
        f"exempted_narrowing={result['n_exempted_narrowing']}"
    )
    if result["forbidden_per_episode"]:
        print("FORBIDDEN_PER_EPISODE:")
        for eid, terms in result["forbidden_per_episode"].items():
            print(f"  {eid}: {terms}")
    if result["forbidden"]:
        print("FORBIDDEN:")
        for row in result["forbidden"][:40]:
            print(f"  {row.get('source')}: {row.get('terms')} :: {row.get('snippet')!r}")
    if result["exempted_by_narrowing"]:
        print("EXEMPTED_BY_NARROWING (term + ~80ch context + episode):")
        # de-dupe display by episode+term+context
        seen: set[tuple[str, str, str]] = set()
        shown = 0
        for row in result["exempted_by_narrowing"]:
            key = (row.get("episode_id") or "", row.get("term") or "", row.get("context") or "")
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{row.get('episode_id')}] {row.get('term')}: {row.get('context')!r}")
            shown += 1
            if shown >= 200:
                print(f"  ... ({len(result['exempted_by_narrowing'])} total exempted rows)")
                break
    if result["whitelist_hits"]:
        print("WHITELIST (upstream task text):")
        c: Counter[str] = Counter()
        for row in result["whitelist_hits"]:
            for t in row["terms"]:
                c[t] += 1
        for t, n in c.most_common():
            print(f"  {t}: {n}")
    print("PASS" if result["ok"] else "FAIL")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
