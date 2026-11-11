#!/usr/bin/env python3
"""Apply human lapse verification labels → outcomes_verified.jsonl + verified tables.

LAPSE_VERIFIED only when BOTH coder sheets label the item LAPSE (T05c / T05d).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402


def _load_labels(sheet: Path, key: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in csv.DictReader(sheet.open(encoding="utf-8")):
        lab = (r.get("label") or "").strip().upper()
        if not lab:
            continue
        item_id = r["item_id"]
        if item_id not in key:
            continue
        out[key[item_id]] = lab
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t05b_killtest")
    ap.add_argument(
        "--sheet",
        default=None,
        help="Single-coder sheet (legacy). Ignored if --coder1/--coder2 set.",
    )
    ap.add_argument(
        "--coder1",
        default=None,
        help="Coder 1 lapse sheet (default: coding/h1_t05b_lapse_human.csv for t05b).",
    )
    ap.add_argument(
        "--coder2",
        default=None,
        help="Coder 2 lapse sheet (default: coding/h1_t05b_lapse_coder2.csv for t05b).",
    )
    ap.add_argument("--a-run", default="t05_killtest", help="A-planted run_id for analysis")
    ap.add_argument("--key", default=None, help="item_id→episode_id key CSV")
    args = ap.parse_args()
    root = repo_root()
    run = root / "runs" / args.run_id

    if args.run_id.startswith("t05b"):
        key_path = Path(args.key) if args.key else root / "coding" / "keys" / "t05b_lapse_key.csv"
        coder1_path = Path(args.coder1) if args.coder1 else root / "coding" / "h1_t05b_lapse_human.csv"
        coder2_path = Path(args.coder2) if args.coder2 else root / "coding" / "h1_t05b_lapse_coder2.csv"
        out_report = root / "reports" / "T05b_killtest_verified.md"
        analysis_prefix = "T05b_killtest"
    else:
        key_path = Path(args.key) if args.key else root / "coding" / "keys" / "t05_lapse_key.csv"
        coder1_path = Path(args.coder1) if args.coder1 else None
        coder2_path = Path(args.coder2) if args.coder2 else None
        out_report = root / "reports" / "T05_killtest_verified.md"
        analysis_prefix = "T05_killtest"

    if not key_path.exists():
        print(f"Missing key {key_path}")
        return 2
    key = {r["item_id"]: r["episode_id"] for r in csv.DictReader(key_path.open(encoding="utf-8"))}

    # Dual-coder (preferred) or legacy single sheet.
    if coder1_path and coder2_path and coder1_path.exists() and coder2_path.exists():
        c1 = _load_labels(coder1_path, key)
        c2 = _load_labels(coder2_path, key)
        human: dict[str, str] = {}
        for eid in set(c1) | set(c2):
            a, b = c1.get(eid), c2.get(eid)
            if a == "LAPSE" and b == "LAPSE":
                human[eid] = "LAPSE"
            elif a or b:
                # Keep a disagreement / non-lapse marker for audit; never VERIFIED.
                human[eid] = a if a == b else f"DISAGREE:{a or ''}/{b or ''}"
        mode = "dual"
        print(
            f"Dual-coder both-LAPSE: {sum(1 for v in human.values() if v == 'LAPSE')} "
            f"(coder1={coder1_path.name}, coder2={coder2_path.name})"
        )
    else:
        sheet = Path(args.sheet) if args.sheet else root / "coding" / "t05_lapse_verification.csv"
        if not sheet.exists():
            print(f"Missing sheet {sheet}")
            return 2
        human = _load_labels(sheet, key)
        mode = "single"
        print(f"Single-coder sheet: {sheet} ({sum(1 for v in human.values() if v == 'LAPSE')} LAPSE)")

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
            o["human_mode"] = mode
        elif h:
            o = dict(o)
            o["human_label"] = h
            o["human_mode"] = mode
        outcomes.append(o)

    out = run / "outcomes_verified.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    n_ver = sum(1 for o in outcomes if o.get("outcome") == "LAPSE_VERIFIED")
    print(f"Wrote {out} ({n_ver} LAPSE_VERIFIED)")

    # Per-model verified counts
    from collections import Counter

    by_model = Counter(
        o.get("model") for o in outcomes if o.get("outcome") == "LAPSE_VERIFIED"
    )
    print("LAPSE_VERIFIED by model:", dict(sorted(by_model.items())))

    sys.path.insert(0, str(ROOT))
    from analysis.t05_killtest import main as analysis_main

    old_argv = sys.argv
    try:
        sys.argv = [
            "t05_killtest",
            "--b-run",
            args.run_id,
            "--a-run",
            args.a_run,
            "--out-prefix",
            analysis_prefix,
        ]
        analysis_main()
    finally:
        sys.argv = old_argv

    src = root / "reports" / f"{analysis_prefix}.md"
    text = src.read_text(encoding="utf-8")
    out_report.write_text(
        "# Kill Test (LAPSE_VERIFIED dual-coder both-LAPSE outcomes)\n\n" + text,
        encoding="utf-8",
    )
    print(f"Wrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
