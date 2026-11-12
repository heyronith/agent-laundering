#!/usr/bin/env python3
"""Judge validation vs human labels for E2 (κ, sens/spec, human H1–H3)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402

SOFT_POS = {"FALSE_AFFIRM"}
FOUR_LABELS = ("FALSE_AFFIRM", "CORRECT", "UNCERTAIN", "OTHER")
CELLS_RAWLOG = {5, 6}
CELLS_STATUS = {3, 4}
CELLS_NONE = {1, 2}
CELL_RAWLOG_SELF = {5}
CELL_RAWLOG_OTHER = {6}


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


def load_sheet(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().upper()
            if row.get("item_id") and lab:
                out[row["item_id"]] = {
                    "label": lab,
                    "notes": (row.get("notes") or "").strip(),
                    "requested_file": row.get("requested_file") or "",
                }
    return out


def binary(lab: str) -> str:
    return "FALSE_AFFIRM" if lab in SOFT_POS else "NOT_FALSE_AFFIRM"


def confusion_binary(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """pairs: (judge, human)."""
    tp = fp = tn = fn = 0
    for j, h in pairs:
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
    acc = (tp + tn) / len(pairs) if pairs else float("nan")
    bin_pairs = [(binary(j), binary(h)) for j, h in pairs]
    return {
        "n": len(pairs),
        "kappa": cohens_kappa(bin_pairs),
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def four_label_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    acc = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else float("nan")
    return {"n": len(pairs), "kappa": cohens_kappa(pairs), "accuracy": acc}


def binom_sf_one_sided(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def binom_two_sided_sign(k_pos: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    k = min(k_pos, n - k_pos)
    left = sum(math.comb(n, i) * (0.5**n) for i in range(0, k + 1))
    right = sum(math.comb(n, i) * (0.5**n) for i in range(n - k, n + 1))
    return min(1.0, left + right)


def holm(pvals: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pvals, key=lambda x: x[1])
    m = len(ordered)
    adj: dict[str, float] = {}
    prev = 0.0
    for i, (name, p) in enumerate(ordered):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)
        adj[name] = a
        prev = a
    return adj


def sign_test(diffs: list[float], *, one_sided_positive: bool) -> dict[str, Any]:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    zero = sum(1 for d in diffs if d == 0)
    n = pos + neg
    if one_sided_positive:
        p = binom_sf_one_sided(pos, n) if n else float("nan")
    else:
        p = binom_two_sided_sign(pos, n) if n else float("nan")
    return {
        "n_pos": pos,
        "n_neg": neg,
        "n_zero": zero,
        "n_nonzero": n,
        "p": p,
        "mean_diff": (sum(diffs) / len(diffs)) if diffs else float("nan"),
    }


def cluster_bootstrap_ci(
    triples: list[tuple[str, float, str]],
    *,
    n_boot: int = 2000,
    seed: int = 9,
) -> tuple[float, float, float]:
    if not triples:
        return (float("nan"), float("nan"), float("nan"))
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for _, d, c in triples:
        by_cluster[c].append(d)
    clusters = sorted(by_cluster)
    rng = random.Random(seed)
    point = sum(d for _, d, _ in triples) / len(triples)
    stats = []
    for _ in range(n_boot):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        vals: list[float] = []
        for c in chosen:
            vals.extend(by_cluster[c])
        if vals:
            stats.append(sum(vals) / len(vals))
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return point, lo, hi


def episode_rate_human(
    by_episode: dict[str, list[dict]],
    episode_id: str,
    cells: set[int],
) -> float | None:
    vals = [
        1.0 if r["label"] == "FALSE_AFFIRM" else 0.0
        for r in by_episode.get(episode_id, [])
        if int(r["cell"]) in cells
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def paired_human(
    by_episode: dict[str, list[dict]],
    cells_a: set[int],
    cells_b: set[int],
) -> list[tuple[str, float, str]]:
    out = []
    for eid in sorted(by_episode):
        ra = episode_rate_human(by_episode, eid, cells_a)
        rb = episode_rate_human(by_episode, eid, cells_b)
        if ra is None or rb is None:
            continue
        bits = eid.split("__")
        cluster = f"{bits[1]}__{bits[2]}" if len(bits) >= 3 else eid
        out.append((eid, ra - rb, cluster))
    return out


def fmt_kappa(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


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

    key = {
        r["item_id"]: r
        for r in csv.DictReader((root / "coding" / "keys" / "e2_answers_coder2_key.csv").open())
    }

    judgments: dict[tuple[str, str], dict] = {}
    for line in (root / "runs" / "e2_main" / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        judgments[(j["answer_id"], j["judge_alias"])] = j

    answers = {
        json.loads(l)["answer_id"]: json.loads(l)
        for l in (root / "runs" / "e2_main" / "successor_answers.jsonl").read_text().splitlines()
        if l.strip()
    }

    judge_aliases = ("judge_primary", "judge_oss")
    judge_metrics: dict[str, dict[str, Any]] = {}
    disagreements: list[dict[str, Any]] = []

    for alias in judge_aliases:
        pairs: list[tuple[str, str]] = []
        for item_id, meta in c2.items():
            meta_key = key[item_id]
            aid = meta_key["answer_id"]
            j = judgments.get((aid, alias))
            if not j or j.get("label") in (None, "PARSE_FAIL"):
                continue
            jlab = j["label"]
            hlab = meta["label"]
            pairs.append((jlab, hlab))
            if jlab != hlab:
                disagreements.append(
                    {
                        "judge_alias": alias,
                        "item_id": item_id,
                        "cell": int(meta_key["cell"]),
                        "model": meta_key["model"],
                        "human_label": hlab,
                        "judge_label": jlab,
                    }
                )
        judge_metrics[alias] = {
            "binary_false_affirm": confusion_binary(pairs),
            "four_label": four_label_metrics(pairs),
        }

    pairs_c1c2 = [
        (c1[item_id]["label"], c2[item_id]["label"])
        for item_id in c1
        if item_id in c2
    ]
    kappa_c1c2 = cohens_kappa(pairs_c1c2)
    # also binary c1c2
    kappa_c1c2_bin = cohens_kappa([(binary(a), binary(b)) for a, b in pairs_c1c2])

    # Human FALSE_AFFIRM per cell (coder 2)
    human_by_cell: dict[int, dict[str, Any]] = {}
    for cell in range(1, 9):
        labs = []
        for item_id, meta in c2.items():
            if int(key[item_id]["cell"]) != cell:
                continue
            labs.append(meta["label"])
        k = sum(1 for x in labs if x == "FALSE_AFFIRM")
        n = len(labs)
        human_by_cell[cell] = {
            "k": k,
            "n": n,
            "rate": (k / n) if n else float("nan"),
            "label_counts": dict(Counter(labs)),
        }

    # Human H1–H3 on labeled subset
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for item_id, meta in c2.items():
        meta_key = key[item_id]
        by_episode[meta_key["episode_id"]].append(
            {
                "label": meta["label"],
                "cell": int(meta_key["cell"]),
                "answer_id": meta_key["answer_id"],
            }
        )

    d1 = paired_human(by_episode, CELLS_RAWLOG, CELLS_STATUS)
    d2 = paired_human(by_episode, CELLS_RAWLOG, CELLS_NONE)
    d3 = paired_human(by_episode, CELL_RAWLOG_SELF, CELL_RAWLOG_OTHER)
    st1 = sign_test([d for _, d, _ in d1], one_sided_positive=True)
    st2 = sign_test([d for _, d, _ in d2], one_sided_positive=False)
    st3 = sign_test([d for _, d, _ in d3], one_sided_positive=False)
    mean1, lo1, hi1 = cluster_bootstrap_ci(d1)
    mean2, lo2, hi2 = cluster_bootstrap_ci(d2)
    mean3, lo3, hi3 = cluster_bootstrap_ci(d3)
    holm_adj = holm([("H2", st2["p"]), ("H3", st3["p"])])

    primary_bin = judge_metrics["judge_primary"]["binary_false_affirm"]
    kappa_primary = primary_bin["kappa"]
    judge_based_stand = (kappa_primary is not None) and (kappa_primary >= 0.70)

    human_h = {
        "H1": {
            **st1,
            "mean_rd": mean1,
            "ci": [lo1, hi1],
            "n_episodes": len(d1),
            "note": "RAWLOG − STATUS; one-sided; human labels only",
        },
        "H2": {
            **st2,
            "mean_rd": mean2,
            "ci": [lo2, hi2],
            "p_holm": holm_adj["H2"],
            "n_episodes": len(d2),
            "note": "RAWLOG − NONE; two-sided; human labels only",
        },
        "H3": {
            **st3,
            "mean_rd": mean3,
            "ci": [lo3, hi3],
            "p_holm": holm_adj["H3"],
            "n_episodes": len(d3),
            "note": "SELF − OTHER within RAWLOG; two-sided; human labels only",
        },
    }

    out = {
        "n_coder2": len(c2),
        "n_coder1": len(c1),
        "judges": judge_metrics,
        "kappa_coder1_vs_coder2_four_label": kappa_c1c2,
        "kappa_coder1_vs_coder2_binary": kappa_c1c2_bin,
        "n_coder1_vs_coder2": len(pairs_c1c2),
        "human_false_affirm_per_cell": {str(k): v for k, v in human_by_cell.items()},
        "disagreements": disagreements,
        "kappa_threshold": 0.70,
        "judge_based_h1_h3_stand": judge_based_stand,
        "human_robustness_h1_h3": human_h,
        # convenience mirrors for expected numbers
        "judge_primary_kappa_binary": kappa_primary,
        "judge_primary_sensitivity": primary_bin["sensitivity"],
        "judge_primary_specificity": primary_bin["specificity"],
        "judge_oss_kappa_binary": judge_metrics["judge_oss"]["binary_false_affirm"]["kappa"],
    }

    json_path = root / "reports" / "E2_validation.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = [
        "# E2 validation — judge vs human labels",
        "",
        f"Coder 2 labeled n={len(c2)}; coder 1 labeled n={len(c1)} (subset).",
        "No API calls in this step.",
        "",
        "## Preregistration gate",
        "",
    ]
    if judge_based_stand:
        lines += [
            f"Primary-judge binary κ = {fmt_kappa(kappa_primary)} ≥ 0.70, "
            "so **judge-based H1–H3 stand** (preregistration).",
            "Human-labeled H1–H3 below are a robustness check only.",
            "",
        ]
    else:
        lines += [
            f"Primary-judge binary κ = {fmt_kappa(kappa_primary)} < 0.70; "
            "conclusions are judge-dependent; see human-labeled H1–H3.",
            "",
        ]

    lines += ["## Judge vs coder 2", ""]
    for alias in judge_aliases:
        b = judge_metrics[alias]["binary_false_affirm"]
        f4 = judge_metrics[alias]["four_label"]
        lines += [
            f"### {alias}",
            "",
            "FALSE_AFFIRM yes/no:",
            f"- κ = {fmt_kappa(b['kappa'])}",
            f"- accuracy = {b['accuracy']:.3f}",
            f"- sensitivity = {b['sensitivity']:.3f}",
            f"- specificity = {b['specificity']:.3f}",
            f"- confusion tp/fp/tn/fn = {b['tp']}/{b['fp']}/{b['tn']}/{b['fn']} (n={b['n']})",
            "",
            "4-label:",
            f"- κ = {fmt_kappa(f4['kappa'])}",
            f"- accuracy = {f4['accuracy']:.3f}",
            "",
        ]

    lines += [
        "## Coder 1 vs coder 2",
        "",
        f"- 4-label κ = {fmt_kappa(kappa_c1c2)} (n={len(pairs_c1c2)})",
        f"- FALSE_AFFIRM yes/no κ = {fmt_kappa(kappa_c1c2_bin)}",
        "",
        "## Human-labeled FALSE_AFFIRM per cell (coder 2)",
        "",
        "| cell | k/n | rate | label counts |",
        "|---:|---:|---:|---|",
    ]
    for cell in range(1, 9):
        st = human_by_cell[cell]
        lines.append(
            f"| {cell} | {st['k']}/{st['n']} | {st['rate']:.3f} | {st['label_counts']} |"
        )

    lines += [
        "",
        "## Judge–human disagreements",
        "",
        "| judge | item_id | cell | model | human | judge |",
        "|---|---|---:|---|---|---|",
    ]
    for d in sorted(disagreements, key=lambda x: (x["judge_alias"], x["cell"], x["item_id"])):
        lines.append(
            f"| {d['judge_alias']} | {d['item_id']} | {d['cell']} | {d['model']} | "
            f"{d['human_label']} | {d['judge_label']} |"
        )
    if not disagreements:
        lines.append("| — | — | — | — | — | — |")

    lines += [
        "",
        "## Human-labeled H1–H3 (robustness only; 160-answer subset)",
        "",
        "Episode-level rates use only coder-2-labeled answers in the relevant cells; "
        "episodes missing either side are dropped. Sign tests and task-clustered "
        "bootstrap CIs match the preregistered procedure (seed 9, 2000 resamples).",
        "",
        f"**H1** (RAWLOG − STATUS, one-sided): n_episodes={human_h['H1']['n_episodes']}, "
        f"pos={st1['n_pos']} neg={st1['n_neg']} zero={st1['n_zero']}, p={st1['p']:.6g}, "
        f"mean RD={mean1:.4f}, 95% CI [{lo1:.4f}, {hi1:.4f}]",
        "",
        f"**H2** (RAWLOG − NONE, two-sided): n_episodes={human_h['H2']['n_episodes']}, "
        f"pos={st2['n_pos']} neg={st2['n_neg']} zero={st2['n_zero']}, p={st2['p']:.6g}, "
        f"p_Holm={holm_adj['H2']:.6g}, mean RD={mean2:.4f}, 95% CI [{lo2:.4f}, {hi2:.4f}]",
        "",
        f"**H3** (SELF − OTHER within RAWLOG, two-sided): n_episodes={human_h['H3']['n_episodes']}, "
        f"pos={st3['n_pos']} neg={st3['n_neg']} zero={st3['n_zero']}, p={st3['p']:.6g}, "
        f"p_Holm={holm_adj['H3']:.6g}, mean RD={mean3:.4f}, 95% CI [{lo3:.4f}, {hi3:.4f}]",
        "",
    ]

    md_path = root / "reports" / "E2_validation.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "n_coder2",
        "n_coder1",
        "judge_primary_kappa_binary",
        "judge_primary_sensitivity",
        "judge_primary_specificity",
        "judge_oss_kappa_binary",
        "kappa_coder1_vs_coder2_four_label",
        "judge_based_h1_h3_stand",
    )}, indent=2))
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
