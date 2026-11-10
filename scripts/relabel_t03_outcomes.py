#!/usr/bin/env python3
"""Relabel T03 B episodes → outcomes.jsonl and print type 1/5 table."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.framework_b.outcomes import relabel_run  # noqa: E402


def main() -> int:
    path = relabel_run("t03_repro")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Wrote {path} ({len(rows)} rows)")
    # Types 1 and 5 table
    print("\nmodel\ttype\ttask_index\tooutcome\tlapse_cand\tno_answer\tfail_kw\tbroken")
    for r in sorted(rows, key=lambda x: (x["model"], x["task_type"], x["task_index"])):
        if r["task_type"] not in (1, 5):
            continue
        print(
            f"{r['model']}\t{r['task_type']}\t{r['task_index']}\t{r['outcome']}\t"
            f"{r['lapse_candidate']}\t{r['no_answer']}\t{r['fail_keywords']}\t{r['broken_tool_calls']}"
        )
    # Verified list
    verified = [r for r in rows if r["outcome"] == "LAPSE_VERIFIED"]
    print("\nLAPSE_VERIFIED:")
    for r in verified:
        print(f"  {r['episode_id']}")
    noans = [r for r in rows if r["outcome"] == "NO_ANSWER" and r["task_type"] == 5]
    print("\nNO_ANSWER type5:")
    for r in noans:
        print(f"  {r['episode_id']} reasons={r['reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
