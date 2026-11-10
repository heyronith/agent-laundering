#!/usr/bin/env python3
"""Score human coding sheet vs judge labels (κ, confusion, per-label agreement)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float | None, float]:
    if not pairs:
        return None, float("nan")
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        matrix[a][b] += 1
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    row = {a: sum(matrix[a].values()) / n for a in labels}
    col = {b: sum(matrix[a][b] for a in labels) / n for b in labels}
    p_e = sum(row[l] * col[l] for l in labels)
    kappa = None if p_e == 1 else (p_o - p_e) / (1 - p_e)
    return kappa, p_o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--sheet", default=None, help="Path to filled coding sheet CSV")
    args = ap.parse_args()
    root = repo_root()
    sheet_path = Path(args.sheet) if args.sheet else root / "coding" / f"{args.run_id}_sheet.csv"
    key_path = root / "coding" / "keys" / f"{args.run_id}_key.csv"
    judgments_path = root / "runs" / args.run_id / "branch_judgments.jsonl"

    key = {r["item_id"]: r["branch_id"] for r in csv.DictReader(key_path.open(encoding="utf-8"))}
    sheet = list(csv.DictReader(sheet_path.open(encoding="utf-8")))
    j_by: dict[tuple[str, str], dict] = {}
    if judgments_path.exists():
        for line in judgments_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("kind") != "memory_note":
                continue
            j_by[(j["branch_id"], j["model_alias"])] = j

    summary: dict = {"run_id": args.run_id, "n_sheet": len(sheet), "by_judge": {}}
    for judge in ("judge_primary", "judge_secondary"):
        pairs = []
        confusion: dict[str, Counter] = defaultdict(Counter)
        for row in sheet:
            human = (row.get("label") or "").strip().upper()
            if not human:
                continue
            bid = key.get(row["item_id"])
            if not bid:
                continue
            j = j_by.get((bid, judge))
            if not j or j.get("label") in (None, "PARSE_FAIL"):
                continue
            pred = j["label"]
            pairs.append((human, pred))
            confusion[human][pred] += 1
        kappa, agree = cohens_kappa(pairs)
        per_label = {}
        for lab in sorted({a for a, _ in pairs} | {b for _, b in pairs}):
            n = sum(1 for a, _ in pairs if a == lab)
            hit = sum(1 for a, b in pairs if a == lab and a == b)
            per_label[lab] = {"n": n, "agree": (hit / n) if n else None}
        summary["by_judge"][judge] = {
            "n_pairs": len(pairs),
            "agreement": agree,
            "cohens_kappa": kappa,
            "confusion": {a: dict(c) for a, c in confusion.items()},
            "per_label_agreement": per_label,
        }

    out = root / "coding" / f"{args.run_id}_score.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
