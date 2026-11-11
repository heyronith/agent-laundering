#!/usr/bin/env python3
"""Build reports/T05d_judge_bakeoff.md from committed bake-off judgments + human labels."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402

SOFTENED = {"EUPHEMIZED", "OMITTED", "FALSE"}
JUDGES = ("jb_gem31pro", "jb_opus46", "jb_oss120b", "jb_gem38flash")
PROMPTS = ("P0", "P1")


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


def soft(lab: str) -> str:
    return "Y" if lab in SOFTENED else "N"


def bootstrap_kappa_ci(
    pairs: list[tuple[str, str]], n_boot: int = 2000, seed: int = 5
) -> tuple[float | None, float | None, float | None]:
    if not pairs:
        return None, None, None
    rng = random.Random(seed)
    point = cohens_kappa(pairs)
    stats = []
    n = len(pairs)
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        k = cohens_kappa(sample)
        if k is not None:
            stats.append(k)
    if not stats:
        return point, None, None
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return point, lo, hi


def fmt(x: float | None) -> str:
    return "NA" if x is None else f"{x:.3f}"


def load_labels(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().upper()
            if row.get("item_id") and lab:
                out[row["item_id"]] = lab
    return out


def main() -> int:
    root = repo_root()
    run_id = "t05d_bakeoff"
    rows = []
    path = root / "runs" / run_id / "bakeoff_judgments.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    c1 = load_labels(root / "coding" / "h1_t05_memory_human.csv")
    c2 = load_labels(root / "coding" / "h1_t05_memory_coder2.csv")
    consensus = {i: c2[i] for i in c2 if c1.get(i) == c2[i]}

    # Prefer repeat 0 for primary κ; also compute retest.
    by = defaultdict(dict)  # (judge,prompt,item,repeat)->label
    parse_fail = 0
    for r in rows:
        if r.get("label") == "PARSE_FAIL":
            parse_fail += 1
        by[(r["model_alias"], r["prompt_id"], r["item_id"], int(r["repeat"]))] = r.get("label")

    # costs
    costs = [
        float(r.get("cost_usd") or 0)
        for r in iter_run_logs(root / "logs")
        if r.get("run_id") == run_id and not r.get("cache_hit")
    ]
    total_cost = sum(costs)
    n_calls = len(costs)
    mean_cost = total_cost / n_calls if n_calls else None

    lines = [
        "# T05d judge bake-off",
        "",
        f"Calls logged: {len(rows)}; parse failures: {parse_fail} ({parse_fail/max(1,len(rows)):.1%}).",
        f"Spend: ${total_cost:.4f}; mean cost/call: ${mean_cost:.5f}" if mean_cost else "Spend: NA",
        f"Projected cost per 1,500 items (×1 judge ×1 prompt ×1 call): "
        f"${(mean_cost or 0)*1500:.2f}" if mean_cost else "",
        "",
        "Softened = EUPHEMIZED ∪ OMITTED ∪ FALSE. Bootstrap 95% CI: 2,000 resamples, seed 5.",
        "Primary human reference = coder 2 (D015). No winner selected.",
        "",
        "## 1. κ by judge × prompt (repeat 0)",
        "",
        "| judge | prompt | softκ vs c2 | 95% CI | softκ vs c1 | 95% CI | softκ vs consensus | 95% CI | 5-label κ vs c2 | retest agree |",
        "|---|---|---:|---|---:|---|---:|---|---:|---:|",
    ]

    wrong: dict[str, list] = defaultdict(list)
    summary = {"rows": len(rows), "parse_fail": parse_fail, "total_cost": total_cost, "cells": {}}

    for judge in JUDGES:
        for prompt in PROMPTS:
            def pairs_vs(ref: dict[str, str], repeat: int = 0):
                out = []
                for iid, lab in ref.items():
                    pred = by.get((judge, prompt, iid, repeat))
                    if pred and pred != "PARSE_FAIL":
                        out.append((soft(lab), soft(pred)))
                return out

            def pairs5(ref: dict[str, str], repeat: int = 0):
                out = []
                for iid, lab in ref.items():
                    pred = by.get((judge, prompt, iid, repeat))
                    if pred and pred != "PARSE_FAIL":
                        out.append((lab, pred))
                return out

            p2 = pairs_vs(c2)
            p1 = pairs_vs(c1)
            pc = pairs_vs(consensus)
            k2, lo2, hi2 = bootstrap_kappa_ci(p2)
            k1, lo1, hi1 = bootstrap_kappa_ci(p1)
            kc, loc, hic = bootstrap_kappa_ci(pc)
            k5 = cohens_kappa(pairs5(c2))
            # retest: soft agree between r0 and r1
            both = 0
            agree = 0
            for iid in c2:
                a = by.get((judge, prompt, iid, 0))
                b = by.get((judge, prompt, iid, 1))
                if a and b and a != "PARSE_FAIL" and b != "PARSE_FAIL":
                    both += 1
                    if soft(a) == soft(b):
                        agree += 1
            retest = (agree / both) if both else None
            lines.append(
                f"| {judge} | {prompt} | {fmt(k2)} | [{fmt(lo2)}, {fmt(hi2)}] | "
                f"{fmt(k1)} | [{fmt(lo1)}, {fmt(hi1)}] | {fmt(kc)} | [{fmt(loc)}, {fmt(hic)}] | "
                f"{fmt(k5)} | {fmt(retest)} |"
            )
            summary["cells"][f"{judge}/{prompt}"] = {
                "soft_c2": k2,
                "soft_c2_ci": [lo2, hi2],
                "soft_c1": k1,
                "soft_consensus": kc,
                "kappa5_c2": k5,
                "retest": retest,
            }
            for iid, lab in c2.items():
                pred = by.get((judge, prompt, iid, 0))
                if pred and pred != "PARSE_FAIL" and soft(pred) != soft(lab):
                    wrong[f"{judge}/{prompt}"].append(
                        {"item_id": iid, "coder2": lab, "judge": pred}
                    )

    lines += ["", "## 2. Confusion matrices vs coder 2 (repeat 0)", ""]
    for judge in JUDGES:
        for prompt in PROMPTS:
            lines.append(f"### {judge} / {prompt}")
            # soft
            cm = defaultdict(Counter)
            cm5 = defaultdict(Counter)
            for iid, lab in c2.items():
                pred = by.get((judge, prompt, iid, 0))
                if not pred or pred == "PARSE_FAIL":
                    continue
                cm[soft(lab)][soft(pred)] += 1
                cm5[lab][pred] += 1
            lines.append("Softened yes/no (rows=coder2, cols=judge):")
            lines.append("| | Y | N |")
            lines.append("|---|---:|---:|")
            for a in ("Y", "N"):
                lines.append(f"| {a} | {cm[a]['Y']} | {cm[a]['N']} |")
            labs = sorted({a for a in cm5} | {b for c in cm5.values() for b in c})
            lines.append("")
            lines.append("5-label:")
            lines.append("| |" + "|".join(labs) + "|")
            lines.append("|---|" + "|".join(["---:"] * len(labs)) + "|")
            for a in labs:
                lines.append("| " + a + " | " + " | ".join(str(cm5[a].get(b, 0)) for b in labs) + " |")
            lines.append("")

    lines += ["", "## 3. Items wrong vs coder 2 (softened, repeat 0)", ""]
    for key, items in wrong.items():
        lines.append(f"### {key} (n={len(items)})")
        lines.append("| item_id | coder2 | judge |")
        lines.append("|---|---|---|")
        for it in items:
            lines.append(f"| `{it['item_id']}` | {it['coder2']} | {it['judge']} |")
        lines.append("")

    out = root / "reports" / "T05d_judge_bakeoff.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "reports" / "T05d_judge_bakeoff.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
