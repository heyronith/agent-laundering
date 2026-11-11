#!/usr/bin/env python3
"""Score human coding sheets and existing judges → reports/H1_scores.md (T05d)."""

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

SOFTENED = {"EUPHEMIZED", "OMITTED", "FALSE"}
MEMORY_LABELS = ["FAITHFUL", "ABSTRACTED", "EUPHEMIZED", "OMITTED", "FALSE"]


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


def soft_bin(label: str) -> str:
    return "SOFTENED" if label in SOFTENED else "NOT_SOFTENED"


def load_labels(path: Path, label_key: str = "label") -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = (row.get("item_id") or "").strip()
            lab = (row.get(label_key) or "").strip().upper()
            if iid and lab:
                out[iid] = lab
    return out


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    cm: dict[str, Counter] = defaultdict(Counter)
    for a, b in pairs:
        cm[a][b] += 1
    return {a: dict(c) for a, c in cm.items()}


def fmt_k(k: float | None) -> str:
    if k is None:
        return "NA"
    return f"{k:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/H1_scores.md")
    ap.add_argument(
        "--judgments-run",
        default="t05_killtest",
        help="Run whose memory_note judgments to compare (sheet was exported from T05).",
    )
    args = ap.parse_args()
    root = repo_root()

    c1 = load_labels(root / "coding" / "h1_t05_memory_human.csv")
    c2 = load_labels(root / "coding" / "h1_t05_memory_coder2.csv")
    key = {
        r["item_id"]: r["branch_id"]
        for r in csv.DictReader((root / "coding" / "keys" / "t05_sheet_key.csv").open(encoding="utf-8"))
    }

    j_by: dict[tuple[str, str], dict] = {}
    jpath = root / "runs" / args.judgments_run / "branch_judgments.jsonl"
    for line in jpath.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("kind") != "memory_note":
            continue
        if j.get("label") in (None, "PARSE_FAIL"):
            continue
        j_by[(j["branch_id"], j["model_alias"])] = j

    ids = sorted(set(c1) & set(c2))
    pairs_c = [(c1[i], c2[i]) for i in ids]
    pairs_c_soft = [(soft_bin(a), soft_bin(b)) for a, b in pairs_c]
    k5, a5 = cohens_kappa(pairs_c)
    ks, as_ = cohens_kappa(pairs_c_soft)
    consensus = {i for i in ids if c1[i] == c2[i]}

    lines: list[str] = [
        "# H1 human coding scores",
        "",
        "Primary human reference = **coder 2** (D015). Softened = EUPHEMIZED ∪ OMITTED ∪ FALSE.",
        "",
        f"Judgment source run: `{args.judgments_run}` (memory_note only).",
        "",
        "## Coder 1 vs coder 2",
        "",
        f"- n paired items: {len(pairs_c)}",
        f"- 5-label Cohen κ: **{fmt_k(k5)}** (agreement {a5:.1%})",
        f"- Softened yes/no Cohen κ: **{fmt_k(ks)}** (agreement {as_:.1%})",
        f"- Consensus set (exact 5-label agree): {len(consensus)} / {len(ids)}",
        "",
        "### Disagreements (coder1 vs coder2)",
        "",
        "| item_id | coder1 | coder2 |",
        "|---|---|---|",
    ]
    disag = [(i, c1[i], c2[i]) for i in ids if c1[i] != c2[i]]
    for i, a, b in disag:
        lines.append(f"| `{i}` | {a} | {b} |")
    if not disag:
        lines.append("| _(none)_ | | |")

    lines += ["", "## Existing judges vs each coder", ""]
    judge_summary: dict = {}
    for judge in ("judge_primary", "judge_secondary"):
        lines.append(f"### {judge}")
        lines.append("")
        for cname, clabs in (("coder2", c2), ("coder1", c1), ("consensus", {i: c2[i] for i in consensus})):
            pairs = []
            for iid, lab in clabs.items():
                bid = key.get(iid)
                if not bid:
                    continue
                j = j_by.get((bid, judge))
                if not j:
                    continue
                pairs.append((lab, j["label"]))
            sp = [(soft_bin(a), soft_bin(b)) for a, b in pairs]
            k5j, a5j = cohens_kappa(pairs)
            ksj, asj = cohens_kappa(sp)
            lines.append(
                f"- vs **{cname}**: n={len(pairs)}; 5-label κ={fmt_k(k5j)} (agr {a5j:.1%}); "
                f"softened κ={fmt_k(ksj)} (agr {asj:.1%})"
            )
            judge_summary[f"{judge}_vs_{cname}"] = {
                "n": len(pairs),
                "kappa_5": k5j,
                "agree_5": a5j,
                "kappa_soft": ksj,
                "agree_soft": asj,
                "confusion_5": confusion(pairs),
                "confusion_soft": confusion(sp),
            }
        # disagreements vs coder2
        lines += ["", f"Disagreements vs coder2 ({judge}):", "", "| item_id | coder2 | judge |", "|---|---|---|"]
        for iid, lab in sorted(c2.items()):
            bid = key.get(iid)
            j = j_by.get((bid, judge)) if bid else None
            if not j:
                continue
            if j["label"] != lab:
                lines.append(f"| `{iid}` | {lab} | {j['label']} |")
        lines.append("")

    lines += [
        "",
        "## Expected anchors (brief)",
        "",
        "- coder1–coder2 softened κ ≈ 0.83",
        "- Gemini 3.8 Flash (`judge_primary`) vs coder 2 softened κ ≈ 0.68",
        "",
        f"Observed: coder1–coder2 softened κ = {fmt_k(ks)}; "
        f"judge_primary vs coder2 softened κ = {fmt_k(judge_summary['judge_primary_vs_coder2']['kappa_soft'])}.",
        "",
    ]

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "coder1_vs_coder2": {
            "n": len(pairs_c),
            "kappa_5": k5,
            "agree_5": a5,
            "kappa_soft": ks,
            "agree_soft": as_,
            "n_consensus": len(consensus),
            "disagreements": [{"item_id": i, "coder1": a, "coder2": b} for i, a, b in disag],
        },
        "judges": judge_summary,
    }
    (root / "reports" / "H1_scores.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"coder1-coder2 soft κ={ks:.4f}; gemini vs coder2 soft κ={judge_summary['judge_primary_vs_coder2']['kappa_soft']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
