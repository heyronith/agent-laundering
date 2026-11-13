#!/usr/bin/env python3
"""Recompute E2 cell rates and H1–H3 from results/e2_outcome_matrix.csv alone."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - half) / den, (centre + half) / den)


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


def _hyp_brief(h: dict[str, Any]) -> dict[str, Any]:
    out = {
        "n_pos": h["n_pos"],
        "n_neg": h["n_neg"],
        "n_zero": h["n_zero"],
        "p": h["p"],
        "mean_rd": h.get("mean_rd", h.get("mean_diff")),
        "ci": h.get("ci"),
        "n_episodes": h.get("n_episodes"),
    }
    if "p_holm" in h:
        out["p_holm"] = h["p_holm"]
    return out


def _cell_brief(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "lo": lo, "hi": hi}


def load_matrix(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "episode": r["episode"],
                    "task": r.get("task") or r["episode"],
                    "model": r["model"],
                    "task_type": int(r["task_type"]),
                    "cell": int(r["cell"]),
                    "n_samples": int(r["n_samples"]),
                    "n_labeled_primary": int(r.get("n_labeled_primary") or r["n_samples"]),
                    "n_labeled_oss": int(r.get("n_labeled_oss") or r["n_samples"]),
                    "n_false_primary": int(r["n_false_primary"]),
                    "n_false_oss": int(r["n_false_oss"]),
                    "diagnostic_loose": r["diagnostic_loose"].lower() in ("1", "true", "yes"),
                    "diagnostic_strict": r["diagnostic_strict"].lower() in ("1", "true", "yes"),
                }
            )
    return rows


def episode_rate(
    by_ep_cell: dict[tuple[str, int], dict],
    episode: str,
    cells: set[int],
    *,
    judge: str,
) -> float | None:
    k_key = "n_false_primary" if judge == "judge_primary" else "n_false_oss"
    n_key = "n_labeled_primary" if judge == "judge_primary" else "n_labeled_oss"
    vals = []
    for cell in cells:
        row = by_ep_cell.get((episode, cell))
        if not row:
            continue
        n = row[n_key]
        if n <= 0:
            continue
        # expand to per-sample rates equally weighted as in analysis (mean of 0/1)
        # analysis averages over labeled samples across cells; equivalent to total_k/total_n
        vals.extend([1.0] * row[k_key] + [0.0] * (n - row[k_key]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def recompute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ep_cell = {(r["episode"], r["cell"]): r for r in rows}
    episodes = sorted({r["episode"] for r in rows})
    ep_task = {}
    for r in rows:
        ep_task[r["episode"]] = r["task"]

    out: dict[str, Any] = {"cells": {}, "hypotheses": {}}
    for judge, k_key, n_key in (
        ("judge_primary", "n_false_primary", "n_labeled_primary"),
        ("judge_oss", "n_false_oss", "n_labeled_oss"),
    ):
        cells = {}
        for cell in range(1, 9):
            k = sum(r[k_key] for r in rows if r["cell"] == cell)
            n = sum(r[n_key] for r in rows if r["cell"] == cell)
            cells[str(cell)] = _cell_brief(k, n)
        out["cells"][judge] = cells

        def paired(cells_a: set[int], cells_b: set[int]) -> list[tuple[str, float, str]]:
            triples = []
            for ep in episodes:
                ra = episode_rate(by_ep_cell, ep, cells_a, judge=judge)
                rb = episode_rate(by_ep_cell, ep, cells_b, judge=judge)
                if ra is None or rb is None:
                    continue
                triples.append((ep, ra - rb, ep_task[ep]))
            return triples

        d1 = paired({5, 6}, {3, 4})
        d2 = paired({5, 6}, {1, 2})
        d3 = paired({5}, {6})
        st1 = sign_test([d for _, d, _ in d1], one_sided_positive=True)
        st2 = sign_test([d for _, d, _ in d2], one_sided_positive=False)
        st3 = sign_test([d for _, d, _ in d3], one_sided_positive=False)
        mean1, lo1, hi1 = cluster_bootstrap_ci(d1)
        mean2, lo2, hi2 = cluster_bootstrap_ci(d2)
        mean3, lo3, hi3 = cluster_bootstrap_ci(d3)
        holm_adj = holm([("H2", st2["p"]), ("H3", st3["p"])])
        out["hypotheses"][judge] = {
            "H1": {**_hyp_brief({**st1, "mean_rd": mean1, "ci": [lo1, hi1], "n_episodes": len(d1)})},
            "H2": {
                **_hyp_brief(
                    {
                        **st2,
                        "mean_rd": mean2,
                        "ci": [lo2, hi2],
                        "p_holm": holm_adj["H2"],
                        "n_episodes": len(d2),
                    }
                )
            },
            "H3": {
                **_hyp_brief(
                    {
                        **st3,
                        "mean_rd": mean3,
                        "ci": [lo3, hi3],
                        "p_holm": holm_adj["H3"],
                        "n_episodes": len(d3),
                    }
                )
            },
        }
    return out


def _round_trip(obj: Any) -> Any:
    """Normalize via JSON so float formatting matches summary.json."""
    return json.loads(json.dumps(obj))


def main() -> int:
    root = ROOT
    matrix_path = root / "results" / "e2_outcome_matrix.csv"
    summary_path = root / "results" / "summary.json"
    if not matrix_path.exists():
        print(f"missing {matrix_path}")
        return 2
    rows = load_matrix(matrix_path)
    recomputed = _round_trip(recompute(rows))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "cells": summary["e2"]["cells"],
        "hypotheses": summary["e2"]["hypotheses"],
    }
    # Drop fields not in summary hyp briefs if any extras
    expected = _round_trip(expected)

    # Diff
    exp_s = json.dumps(expected, indent=2, sort_keys=True)
    got_s = json.dumps(recomputed, indent=2, sort_keys=True)
    if exp_s == got_s:
        print("DIFF: (empty)")
        print(f"rows={len(rows)} episodes={len({r['episode'] for r in rows})}")
        return 0

    # Show compact diff of keys
    import difflib

    diff = "".join(
        difflib.unified_diff(
            exp_s.splitlines(keepends=True),
            got_s.splitlines(keepends=True),
            fromfile="summary.json e2 cells/hypotheses",
            tofile="recompute_from_matrix",
        )
    )
    print(diff)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
