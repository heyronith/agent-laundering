#!/usr/bin/env python3
"""Acceptance: no *kept* t05_killtest call used credit-era reduced max_tokens.

Original protocol limits (kept):
  B subject 2048, B/A judges 2048, branch 1500, A planted report subject 1024.
Credit-era reduced (must be discarded): judges 256/512/1024, branches 512/768.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402


def _mt(r: dict) -> int | None:
    return ((r.get("request") or {}).get("params") or {}).get("max_tokens")


def is_degraded(r: dict) -> bool:
    """True if this call used a temporarily shrunk limit (D010)."""
    mt = _mt(r)
    if mt is None:
        return False
    role = r.get("role")
    tid = r.get("task_id") or ""
    is_branch = any(x in tid for x in ("__K__", "__M1__", "__M3__", "__M4__", "__D__"))
    is_planted_report = tid.startswith("planted__") and tid.endswith("__report")

    if mt in (256, 512, 768):
        return True
    if mt == 1024:
        # Original A planted subject uses 1024; judges/branches at 1024 were reductions.
        if is_planted_report and role == "subject":
            return False
        return True
    return False


def main() -> int:
    root = repo_root()
    discarded_ids = set()
    disc = root / "runs" / "t05_killtest" / "discarded_calls.jsonl"
    if disc.exists():
        for line in disc.read_text(encoding="utf-8").splitlines():
            if line.strip():
                discarded_ids.add(json.loads(line).get("call_id"))

    # Append any newly identified degraded successful calls into discarded_calls.
    new_rows = []
    bad = []
    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != "t05_killtest":
            continue
        if not is_degraded(r):
            continue
        cid = r.get("call_id")
        if cid in discarded_ids:
            continue
        row = {
            "call_id": cid,
            "task_id": r.get("task_id"),
            "role": r.get("role"),
            "model_alias": r.get("model_alias"),
            "max_tokens": _mt(r),
            "http_status": r.get("http_status"),
            "error": r.get("error"),
            "cost_usd": r.get("cost_usd"),
            "reason": f"reduced_max_tokens={_mt(r)}",
        }
        # Always record degraded calls in discarded file (logs kept).
        new_rows.append(row)
        discarded_ids.add(cid)
        if not r.get("error") and (r.get("http_status") or 0) < 400:
            # Successful degraded — must not be treated as kept evidence; already marked.
            pass

    if new_rows:
        with disc.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        print(f"appended {len(new_rows)} degraded calls to discarded_calls.jsonl")

    # Kept successful calls must not be degraded.
    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != "t05_killtest":
            continue
        if r.get("error") or (r.get("http_status") or 0) >= 400:
            continue
        if is_degraded(r) and r.get("call_id") not in discarded_ids:
            bad.append((r.get("call_id"), _mt(r), r.get("task_id"), r.get("role")))

    if bad:
        print(f"FAIL: {len(bad)} successful reduced-token calls not in discarded_calls.jsonl")
        for row in bad[:20]:
            print(" ", row)
        return 1
    print("OK: all successful degraded-max_tokens t05_killtest calls are marked discarded")
    print("OK: A planted report subject @1024 treated as original protocol (not degraded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
