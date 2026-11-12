#!/usr/bin/env python3
"""Judge validation vs human labels for E2 (κ, sensitivity, specificity)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402

SOFT_POS = {"FALSE_AFFIRM"}


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        matrix[a][b] += 1
    n = len(pairs)
    p_o = sum(1 for a, b in pairs if a == b) / n
    row = {a: sum(matrix[a].values()) / n for a in labels}
    col = {b: sum(matrix[a][b] for a in labels) / n for b in labels}
    p_e = sum(row[l] * col[l] for l in labels)
    return None if p_e == 1 else (p_o - p_e) / (1 - p_e)


def load_sheet(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().upper()
            if row.get("item_id") and lab:
                out[row["item_id"]] = lab
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Smoke test on blank sheets")
    args = ap.parse_args()
    root = repo_root()
    c2 = load_sheet(root / "coding" / "e2_answers_coder2.csv")
    c1 = load_sheet(root / "coding" / "e2_answers_coder1.csv")
    if args.smoke:
        print(f"smoke: coder2_filled={len(c2)} coder1_filled={len(c1)}")
        return 0
    if len(c2) < 10:
        print("Sheets mostly blank; refusing to compute. Use --smoke to check.")
        return 2

    # Join judge_primary labels via key
    key = {
        r["item_id"]: r
        for r in csv.DictReader((root / "coding" / "keys" / "e2_answers_coder2_key.csv").open())
    }
    judgments = {}
    for line in (root / "runs" / "e2_main" / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("judge_alias") == "judge_primary":
            judgments[j["answer_id"]] = j.get("label")

    pairs_jc2 = []
    for item_id, lab in c2.items():
        aid = key[item_id]["answer_id"]
        jlab = judgments.get(aid)
        if jlab and jlab != "PARSE_FAIL":
            pairs_jc2.append((jlab, lab))

    # Binary FALSE_AFFIRM
    tp = fp = tn = fn = 0
    for j, h in pairs_jc2:
        jp = j in SOFT_POS
        hp = h in SOFT_POS
        if jp and hp:
            tp += 1
        elif jp and not hp:
            fp += 1
        elif (not jp) and (not hp):
            tn += 1
        else:
            fn += 1
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    kappa = cohens_kappa(pairs_jc2)

    pairs_c1c2 = []
    for item_id, lab1 in c1.items():
        if item_id in c2:
            pairs_c1c2.append((lab1, c2[item_id]))

    out = {
        "n_coder2": len(c2),
        "n_judge_vs_coder2": len(pairs_jc2),
        "kappa_judge_vs_coder2": kappa,
        "sensitivity_false_affirm": sens,
        "specificity_false_affirm": spec,
        "confusion_tp_fp_tn_fn": [tp, fp, tn, fn],
        "kappa_coder1_vs_coder2": cohens_kappa(pairs_c1c2),
        "n_coder1_vs_coder2": len(pairs_c1c2),
    }
    path = root / "reports" / "E2_validation.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
