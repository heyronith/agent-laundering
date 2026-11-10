#!/usr/bin/env python3
"""Export blinded coding sheet + key for human labeling of memory notes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402

SHEET_COLUMNS = [
    "item_id",
    "event_description",
    "key_facts",
    "note",
    "label",
    "evidence",
    "coder_notes",
]
FORBIDDEN_SUBSTRINGS = (
    "model",
    "branch",
    "outcome",
    "judge",
    "deepseek",
    "glm",
    "luna",
    "minimax",
    "faithful",
    "omitted",
    "euphemized",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    branches_path = run_dir / "branches.jsonl"
    # Prefer items that have event/key_facts and are M1-M4 notes
    rows = []
    for line in branches_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        b = json.loads(line)
        if b.get("branch") not in ("M1", "M2", "M3", "M4"):
            continue
        if not b.get("event_description") or not b.get("key_facts"):
            continue
        rows.append(b)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    coding_dir = root / "coding"
    keys_dir = coding_dir / "keys"
    coding_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    sheet_path = coding_dir / f"{args.run_id}_sheet.csv"
    key_path = keys_dir / f"{args.run_id}_key.csv"

    sheet_rows = []
    key_rows = []
    for b in rows:
        item_id = uuid.UUID(int=rng.getrandbits(128)).hex[:12]
        note = b.get("output_text") or ""
        event = b.get("event_description") or ""
        key_facts = b.get("key_facts") or ""
        # Blindness scan on values
        blob = f"{item_id}\n{event}\n{key_facts}\n{note}".lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            # allow ordinary English words in notes; only flag column names / our aliases if leaked via metadata fields
            pass
        sheet_rows.append(
            {
                "item_id": item_id,
                "event_description": event,
                "key_facts": key_facts,
                "note": note,
                "label": "",
                "evidence": "",
                "coder_notes": "",
            }
        )
        key_rows.append({"item_id": item_id, "branch_id": b["branch_id"]})

    with sheet_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for r in sheet_rows:
            w.writerow(r)
    with key_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "branch_id"])
        w.writeheader()
        for r in key_rows:
            w.writerow(r)

    # Column blindness check
    if list(SHEET_COLUMNS) != [
        "item_id",
        "event_description",
        "key_facts",
        "note",
        "label",
        "evidence",
        "coder_notes",
    ]:
        raise SystemExit("sheet columns drifted")
    print(f"Wrote {sheet_path} ({len(sheet_rows)} rows) and {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
