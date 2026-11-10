#!/usr/bin/env python3
"""Apply human lapse verification labels → outcomes_verified.jsonl + verified tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t05_killtest")
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()
    root = repo_root()
    run = root / "runs" / args.run_id
    sheet = Path(args.sheet) if args.sheet else root / "coding" / "t05_lapse_verification.csv"
    key_path = root / "coding" / "keys" / "t05_lapse_key.csv"
    if not sheet.exists():
        print(f"Missing sheet {sheet}")
        return 2
    key = {r["item_id"]: r["episode_id"] for r in csv.DictReader(key_path.open(encoding="utf-8"))}
    human = {}
    for r in csv.DictReader(sheet.open(encoding="utf-8")):
        lab = (r.get("label") or "").strip().upper()
        if lab:
            human[key[r["item_id"]]] = lab

    outcomes = []
    for line in (run / "outcomes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        h = human.get(o["episode_id"])
        if h == "LAPSE":
            o = dict(o)
            o["outcome"] = "LAPSE_VERIFIED"
            o["human_label"] = h
        elif h:
            o = dict(o)
            o["human_label"] = h
        outcomes.append(o)

    out = run / "outcomes_verified.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    print(f"Wrote {out} ({sum(1 for o in outcomes if o.get('outcome')=='LAPSE_VERIFIED')} LAPSE_VERIFIED)")

    # Recompute analysis restricted note: full analysis script reads outcomes_verified if present
    sys.path.insert(0, str(ROOT))
    from analysis.t05_killtest import main as analysis_main

    analysis_main()
    # Copy to verified report name
    src = root / "reports" / "T05_killtest.md"
    dst = root / "reports" / "T05_killtest_verified.md"
    text = src.read_text(encoding="utf-8")
    dst.write_text("# T05 Kill Test (LAPSE_VERIFIED human-applied outcomes)\n\n" + text, encoding="utf-8")
    print(f"Wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
