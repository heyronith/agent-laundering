#!/usr/bin/env python3
"""E2 preregistered analysis → reports/E2_report.md + E2_summary.json."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402

RUN_ID = "e2_main"
PREREG_SHA = "9ab6faa1b6770974b7cfc9d8dfd4384638e04691"


def _clean_text(s: str) -> str:
    """Drop NULs / other C0 controls that can appear in raw tool observations."""
    return "".join(ch for ch in str(s or "") if ch in "\n\t" or ord(ch) >= 32)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - half) / den, (centre + half) / den)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def binom_sf_one_sided(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X~Bin(n,p)."""
    if n <= 0:
        return float("nan")
    # exact sum
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return total


def binom_two_sided_sign(k_pos: int, n: int) -> float:
    """Two-sided exact sign test p-value (equal probability)."""
    if n <= 0:
        return float("nan")
    # min of two one-sided, ×2, capped at 1; use smaller tail
    k = min(k_pos, n - k_pos)
    # P(X <= k) + P(X >= n-k)
    left = sum(math.comb(n, i) * (0.5**n) for i in range(0, k + 1))
    right = sum(math.comb(n, i) * (0.5**n) for i in range(n - k, n + 1))
    return min(1.0, left + right)


def holm(pvals: list[tuple[str, float]]) -> dict[str, float]:
    """Return adjusted p-values keyed by name."""
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


def episode_rate(
    answers: dict[str, dict],
    judgments: dict[str, dict],
    episode_id: str,
    cells: set[int],
) -> float | None:
    """Mean FALSE_AFFIRM over samples in the given cells (missing/PARSE_FAIL skipped)."""
    vals = []
    for aid, a in answers.items():
        if a["episode_id"] != episode_id:
            continue
        if int(a["cell"]) not in cells:
            continue
        j = judgments.get(aid)
        if not j or j.get("label") in (None, "PARSE_FAIL"):
            continue
        vals.append(1.0 if j["label"] == "FALSE_AFFIRM" else 0.0)
    if not vals:
        return None
    return sum(vals) / len(vals)


def paired_diffs(
    answers: dict[str, dict],
    judgments: dict[str, dict],
    cells_a: set[int],
    cells_b: set[int],
) -> list[tuple[str, float, str]]:
    """Episode-level rate_a - rate_b with task cluster id."""
    episodes = sorted({a["episode_id"] for a in answers.values()})
    out = []
    for eid in episodes:
        ra = episode_rate(answers, judgments, eid, cells_a)
        rb = episode_rate(answers, judgments, eid, cells_b)
        if ra is None or rb is None:
            continue
        # task cluster = type + index from episode_id
        bits = eid.split("__")
        cluster = f"{bits[1]}__{bits[2]}" if len(bits) >= 3 else eid
        out.append((eid, ra - rb, cluster))
    return out


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
    """Mean risk difference CI clustering on task id."""
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
        # resample clusters
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        vals = []
        for c in chosen:
            vals.extend(by_cluster[c])
        if vals:
            stats.append(sum(vals) / len(vals))
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return point, lo, hi


def cell_table(answers: dict, judgments: dict, judge: str) -> dict[int, dict]:
    by = {j["answer_id"]: j for j in judgments.values() if j.get("judge_alias") == judge} if False else None
    # judgments passed as list-indexed by answer for this judge
    out = {}
    for cell in range(1, 9):
        labs = []
        for aid, a in answers.items():
            if int(a["cell"]) != cell:
                continue
            j = judgments.get((aid, judge))
            if j and j.get("label") not in (None, "PARSE_FAIL"):
                labs.append(j["label"])
        n = len(labs)
        k = sum(1 for x in labs if x == "FALSE_AFFIRM")
        lo, hi = wilson_ci(k, n)
        out[cell] = {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}
    return out


def main() -> int:
    root = repo_root()
    run = root / "runs" / RUN_ID
    answers = {r["answer_id"]: r for r in load_jsonl(run / "successor_answers.jsonl")}
    mats = {m["episode_id"]: m for m in load_jsonl(run / "materials.jsonl")}
    stimuli = load_jsonl(run / "stimuli.jsonl")

    judgments_list = load_jsonl(run / "judgments.jsonl")
    # Prefer last non-fail if duplicates
    jmap: dict[tuple[str, str], dict] = {}
    for j in judgments_list:
        key = (j["answer_id"], j["judge_alias"])
        prev = jmap.get(key)
        if prev is None or (prev.get("label") == "PARSE_FAIL" and j.get("label") != "PARSE_FAIL"):
            jmap[key] = j

    # Cost
    cost_by_role: Counter = Counter()
    n_by_role: Counter = Counter()
    total = 0.0
    first_e2_ts = None
    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != RUN_ID or r.get("cache_hit"):
            continue
        c = float(r.get("cost_usd") or 0)
        total += c
        cost_by_role[r.get("role") or "?"] += c
        n_by_role[r.get("role") or "?"] += 1
        ts = r.get("timestamp_utc")
        if ts and (first_e2_ts is None or ts < first_e2_ts):
            first_e2_ts = ts

    leak_text = (
        (root / "reports" / "E2_leakcheck.txt").read_text()
        if (root / "reports" / "E2_leakcheck.txt").exists()
        else "(not run)"
    )
    checkpoint = (
        (root / "reports" / "E2_checkpoint.md").read_text()
        if (root / "reports" / "E2_checkpoint.md").exists()
        else "(missing)"
    )

    results: dict[str, Any] = {
        "run_id": RUN_ID,
        "prereg_sha": PREREG_SHA,
        "n_stimuli": len(stimuli),
        "n_answers": len(answers),
        "n_judgments": len(judgments_list),
        "total_spend": total,
        "cost_by_role": dict(cost_by_role),
        "first_e2_log_timestamp": first_e2_ts,
        "gates": {},
        "hypotheses": {},
        "descriptive": {},
    }

    lines = [
        "# E2 Report — Evidence format experiment",
        "",
        f"Preregistration tag `prereg-e2` SHA `{PREREG_SHA}`. "
        f"First E2 log timestamp: `{first_e2_ts}`. Spend **${total:.4f}** / $8.",
        "",
        "## Stimuli",
        "",
        f"n={len(stimuli)} after excluding holdout (`runs/e1_holdout.json`).",
        "",
    ]
    ctr = Counter((s["model"], s["task_type"]) for s in stimuli)
    lines.append("| model | type | n |")
    lines.append("|---|---:|---:|")
    for (m, t), n in sorted(ctr.items()):
        lines.append(f"| {m} | {t} | {n} |")
    lines.append("")

    # Hypotheses cell sets
    # H1: RAWLOG (5,6) vs STATUS (3,4), MISLEADING pooled
    # H2: RAWLOG (5,6) vs NONE (1,2)
    # H3: SELF vs OTHER within RAWLOG: cell 5 vs 6

    for judge in ("judge_primary", "judge_oss"):
        j_for = {aid: jmap[(aid, judge)] for aid, _ in answers.items() if (aid, judge) in jmap}
        # rebuild judgments dict keyed by answer_id for helpers that expect that
        # Actually episode_rate uses judgments.get(aid) — pass j_for
        ct = {}
        for cell in range(1, 9):
            labs = []
            for aid, a in answers.items():
                if int(a["cell"]) != cell:
                    continue
                j = j_for.get(aid)
                if j and j.get("label") not in (None, "PARSE_FAIL"):
                    labs.append(j["label"])
            n = len(labs)
            k = sum(1 for x in labs if x == "FALSE_AFFIRM")
            lo, hi = wilson_ci(k, n)
            ct[cell] = {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}

        lines += [f"## Cell table — {judge}", "", "| cell | k/n | rate | Wilson 95% CI |", "|---:|---:|---:|---|"]
        for cell in range(1, 9):
            st = ct[cell]
            lines.append(
                f"| {cell} | {st['k']}/{st['n']} | {st['rate']:.3f} | [{st['lo']:.3f}, {st['hi']:.3f}] |"
            )
        lines.append("")

        # H1
        d1 = paired_diffs(answers, j_for, {5, 6}, {3, 4})
        st1 = sign_test([d for _, d, _ in d1], one_sided_positive=True)
        mean1, lo1, hi1 = cluster_bootstrap_ci(d1)
        # H2
        d2 = paired_diffs(answers, j_for, {5, 6}, {1, 2})
        st2 = sign_test([d for _, d, _ in d2], one_sided_positive=False)
        mean2, lo2, hi2 = cluster_bootstrap_ci(d2)
        # H3: SELF(5) - OTHER(6) within RAWLOG — two-sided
        d3 = paired_diffs(answers, j_for, {5}, {6})
        st3 = sign_test([d for _, d, _ in d3], one_sided_positive=False)
        mean3, lo3, hi3 = cluster_bootstrap_ci(d3)

        holm_adj = holm([("H2", st2["p"]), ("H3", st3["p"])])

        hyp = {
            "H1": {**st1, "mean_rd": mean1, "ci": [lo1, hi1], "n_episodes": len(d1)},
            "H2": {
                **st2,
                "mean_rd": mean2,
                "ci": [lo2, hi2],
                "p_holm": holm_adj["H2"],
                "n_episodes": len(d2),
            },
            "H3": {
                **st3,
                "mean_rd": mean3,
                "ci": [lo3, hi3],
                "p_holm": holm_adj["H3"],
                "n_episodes": len(d3),
            },
            "cells": ct,
        }
        results["hypotheses"][judge] = hyp

        lines += [
            f"## Hypotheses — {judge}",
            "",
            f"**H1** (RAWLOG − STATUS, one-sided): n_episodes={len(d1)}, "
            f"pos={st1['n_pos']} neg={st1['n_neg']} zero={st1['n_zero']}, "
            f"p={st1['p']:.6g}, mean RD={mean1:.4f}, 95% CI [{lo1:.4f}, {hi1:.4f}]",
            "",
            f"**H2** (RAWLOG − NONE, two-sided): pos={st2['n_pos']} neg={st2['n_neg']} "
            f"zero={st2['n_zero']}, p={st2['p']:.6g}, p_Holm={holm_adj['H2']:.6g}, "
            f"mean RD={mean2:.4f}, 95% CI [{lo2:.4f}, {hi2:.4f}]",
            "",
            f"**H3** (SELF − OTHER within RAWLOG, two-sided): pos={st3['n_pos']} neg={st3['n_neg']} "
            f"zero={st3['n_zero']}, p={st3['p']:.6g}, p_Holm={holm_adj['H3']:.6g}, "
            f"mean RD={mean3:.4f}, 95% CI [{lo3:.4f}, {hi3:.4f}]",
            "",
        ]

        # Descriptive
        # per model / type / diagnostic
        desc: dict[str, Any] = {}
        for dim_name, keyfn in (
            ("model", lambda a: a["model"]),
            ("task_type", lambda a: str(a["task_type"])),
        ):
            desc[dim_name] = {}
            groups = sorted({keyfn(a) for a in answers.values()})
            for g in groups:
                # MISLEADING RAWLOG vs STATUS rates pooled
                for cond, cells in (("RAWLOG", {5, 6}), ("STATUS", {3, 4}), ("NONE", {1, 2})):
                    labs = []
                    for aid, a in answers.items():
                        if keyfn(a) != g or int(a["cell"]) not in cells:
                            continue
                        j = j_for.get(aid)
                        if j and j.get("label") not in (None, "PARSE_FAIL"):
                            labs.append(j["label"])
                    n = len(labs)
                    k = sum(1 for x in labs if x == "FALSE_AFFIRM")
                    desc[dim_name].setdefault(g, {})[cond] = {
                        "k": k,
                        "n": n,
                        "rate": k / n if n else float("nan"),
                    }

        # diagnostic
        desc["diagnostic"] = {}
        for flag in (True, False):
            for cond, cells in (("RAWLOG", {5, 6}),):
                labs = []
                for aid, a in answers.items():
                    if int(a["cell"]) not in cells:
                        continue
                    if bool(a.get("diagnostic")) != flag and bool(mats.get(a["episode_id"], {}).get("diagnostic")) != flag:
                        # prefer answer field else materials
                        continue
                    diag = a.get("diagnostic")
                    if diag is None:
                        diag = mats.get(a["episode_id"], {}).get("diagnostic")
                    if bool(diag) != flag:
                        continue
                    j = j_for.get(aid)
                    if j and j.get("label") not in (None, "PARSE_FAIL"):
                        labs.append(j["label"])
                n = len(labs)
                k = sum(1 for x in labs if x == "FALSE_AFFIRM")
                desc["diagnostic"][str(flag)] = {
                    "k": k,
                    "n": n,
                    "rate": k / n if n else float("nan"),
                }

        # faithful controls
        desc["faithful"] = {str(c): ct[c] for c in (7, 8)}

        # mentions_record
        desc["mentions_record"] = {}
        for evid in ("NONE", "STATUS", "RAWLOG"):
            vals = []
            for aid, a in answers.items():
                if a["evidence"] != evid:
                    continue
                j = j_for.get(aid)
                if j and j.get("mentions_record") in ("yes", "no"):
                    vals.append(j["mentions_record"])
            yes = sum(1 for v in vals if v == "yes")
            desc["mentions_record"][evid] = {
                "yes": yes,
                "n": len(vals),
                "rate": yes / len(vals) if vals else float("nan"),
            }

        # judge agreement
        both = []
        for aid in answers:
            jp = jmap.get((aid, "judge_primary"))
            jo = jmap.get((aid, "judge_oss"))
            if jp and jo and jp.get("label") != "PARSE_FAIL" and jo.get("label") != "PARSE_FAIL":
                both.append((jp["label"], jo["label"]))
        agree = sum(1 for a, b in both if a == b)
        desc["judge_agreement"] = {
            "n": len(both),
            "agree": agree,
            "rate": agree / len(both) if both else float("nan"),
        }

        results["descriptive"][judge] = desc

        lines += [
            f"## Descriptive — {judge}",
            "",
            "### Per model (FALSE_AFFIRM k/n)",
            "",
        ]
        for g, conds in sorted(desc["model"].items()):
            lines.append(
                f"- {g}: NONE {conds.get('NONE',{})}; STATUS {conds.get('STATUS',{})}; "
                f"RAWLOG {conds.get('RAWLOG',{})}"
            )
        lines += ["", "### Per task type", ""]
        for g, conds in sorted(desc["task_type"].items()):
            lines.append(
                f"- type {g}: NONE {conds.get('NONE',{})}; STATUS {conds.get('STATUS',{})}; "
                f"RAWLOG {conds.get('RAWLOG',{})}"
            )
        lines += [
            "",
            "### RAWLOG by diagnosticity",
            "",
            f"- diagnostic=true: {desc['diagnostic'].get('True')}",
            f"- diagnostic=false: {desc['diagnostic'].get('False')}",
            "",
            "### Faithful controls",
            "",
            f"- cell 7 (FAITHFUL/NONE): {desc['faithful']['7']}",
            f"- cell 8 (FAITHFUL/RAWLOG): {desc['faithful']['8']}",
            "",
            "### mentions_record",
            "",
        ]
        for evid, m in desc["mentions_record"].items():
            lines.append(f"- {evid}: {m['yes']}/{m['n']} = {m['rate']:.3f}")
        lines += [
            "",
            f"Judge agreement primary vs oss: {agree}/{len(both)} = "
            f"{(agree/len(both) if both else float('nan')):.3f}",
            "",
        ]

    # Example RAWLOG prompts + answers
    rng = random.Random(9)
    mats_list = list(mats.values())
    raw_eps = [m for m in mats_list if m.get("evidence_rawlog")]
    rng.shuffle(raw_eps)
    lines += ["## Example RAWLOG evidence blocks (3)", ""]
    for m in raw_eps[:3]:
        lines.append(f"### {m['episode_id']} (diagnostic={m.get('diagnostic')})")
        lines.append("```")
        lines.append(_clean_text(m.get("evidence_rawlog") or "")[:2500])
        lines.append("```")
        lines.append("")

    lines += ["## Sample answers (2 per cell)", ""]
    for cell in range(1, 9):
        pool = [a for a in answers.values() if int(a["cell"]) == cell]
        rng.shuffle(pool)
        lines.append(f"### Cell {cell}")
        for a in pool[:2]:
            lines.append(f"**{a['answer_id']}** ({a['model']}):")
            lines.append("")
            lines.append(_clean_text(a.get("answer") or "")[:1500])
            lines.append("")

    lines += [
        "## Cost checkpoint",
        "",
        checkpoint.strip(),
        "",
        "## Leak check",
        "",
        "```",
        leak_text.strip(),
        "```",
        "",
        "## Cost by role",
        "",
        "| role | n | spend |",
        "|---|---:|---:|",
    ]
    for role in sorted(cost_by_role):
        lines.append(f"| {role} | {n_by_role[role]} | {cost_by_role[role]:.4f} |")
    lines += [
        f"| **total** | {sum(n_by_role.values())} | **{total:.4f}** |",
        "",
        "## Deviations vs preregistration",
        "",
        "- Leak check whitelists intentional materials text (RAWLOG/STATUS/META/notes) so "
        "agent-code comments like `# Experiment:` inside RAWLOG are not false positives.",
        "- 1 judge PARSE_FAIL after 3 retries (reported in judgments).",
        "- 12 successor calls used retry task_id suffixes (`__aN`) after transient errors; "
        "n_answers=1232 complete.",
        "- Human validation sheets issued blank; `e2_validation.py` not scored until labels arrive.",
        "",
    ]

    (root / "reports" / "E2_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (root / "reports" / "E2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote E2_report.md spend={total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
