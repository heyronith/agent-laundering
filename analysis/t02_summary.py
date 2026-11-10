#!/usr/bin/env python3
"""Compute T02 summary tables from committed episodes/judgments/logs."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (low, high)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, dict]:
    """Return (kappa, agreement_rate, confusion matrix as nested dict)."""
    if not pairs:
        return (float("nan"), float("nan"), {})
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        matrix[a][b] += 1
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    # marginals
    row = {a: sum(matrix[a].values()) / n for a in labels}
    col = {b: sum(matrix[a][b] for a in labels) / n for b in labels}
    p_e = sum(row[l] * col[l] for l in labels)
    kappa = float("nan") if p_e == 1 else (p_o - p_e) / (1 - p_e)
    return kappa, p_o, matrix


def main() -> int:
    root = repo_root()
    run_id = "t02_repro"
    run_dir = root / "runs" / run_id
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    logs = [r for r in iter_run_logs(root / "logs") if r.get("run_id") == run_id]

    # Index judgments by episode/kind/judge
    jidx: dict[tuple[str, str, str], dict] = {}
    for j in judgments:
        jidx[(j["episode_id"], j["kind"], j["model_alias"])] = j

    # 1) per model × file
    by_mf: dict[tuple[str, str], list] = defaultdict(list)
    for ep in episodes:
        by_mf[(ep["model_alias"], ep["prompt_file"])].append(ep)

    per_model_file = []
    for (model, prompt_file), rows in sorted(by_mf.items()):
        n = len(rows)
        mis_o = sum(1 for r in rows if r.get("misaligned_original"))
        mis_n = sum(1 for r in rows if r.get("misaligned_normalized"))
        disagreements = sum(1 for r in rows if r.get("label_disagreement"))
        lo, hi = wilson_ci(mis_o, n)
        per_model_file.append(
            {
                "model": model,
                "prompt_file": prompt_file,
                "n": n,
                "misaligned_original_count": mis_o,
                "misalignment_rate_original": mis_o / n if n else None,
                "wilson95_original": [lo, hi],
                "misaligned_normalized_count": mis_n,
                "misalignment_rate_normalized": mis_n / n if n else None,
                "label_disagreement_count": disagreements,
            }
        )

    # 2/3 default-only distributions via judge_primary
    report_dist = defaultdict(Counter)
    doubling_dist = defaultdict(Counter)
    for ep in episodes:
        if ep.get("prompt_file") != "default.json":
            continue
        if not ep.get("misaligned_original"):
            continue
        model = ep["model_alias"]
        jr = jidx.get((ep["episode_id"], "report", "judge_primary"))
        if jr:
            report_dist[model][jr["label"]] += 1
        jd = jidx.get((ep["episode_id"], "doubling", "judge_primary"))
        if jd:
            doubling_dist[model][jd["label"]] += 1

    # 4) judge agreement pooled
    report_pairs = []
    doubling_pairs = []
    for ep in episodes:
        eid = ep["episode_id"]
        rp = jidx.get((eid, "report", "judge_primary"))
        rs = jidx.get((eid, "report", "judge_secondary"))
        if rp and rs:
            report_pairs.append((rp["label"], rs["label"]))
        dp = jidx.get((eid, "doubling", "judge_primary"))
        ds = jidx.get((eid, "doubling", "judge_secondary"))
        if dp and ds:
            doubling_pairs.append((dp["label"], ds["label"]))

    rk, ra, rconf = cohens_kappa(report_pairs)
    dk, da, dconf = cohens_kappa(doubling_pairs)

    # 5) failures / suspicion / amount
    parse_fails = sum(1 for j in judgments if j.get("label") == "PARSE_FAIL")
    parse_fail_rate = parse_fails / len(judgments) if judgments else 0.0
    suspicion = sum(1 for ep in episodes if ep.get("eval_suspicion_keyword"))
    amount_false = sum(1 for ep in episodes if ep.get("amount_parsed") is False)

    # 6) cost by model and role from logs
    cost_by_model = Counter()
    cost_by_role = Counter()
    for r in logs:
        c = float(r.get("cost_usd") or 0.0)
        cost_by_model[r.get("model_alias") or "unknown"] += c
        cost_by_role[r.get("role") or "unknown"] += c

    summary = {
        "run_id": run_id,
        "n_episodes": len(episodes),
        "n_judgments": len(judgments),
        "per_model_file": per_model_file,
        "report_label_distribution_default_misaligned_judge_primary": {
            m: dict(c) for m, c in sorted(report_dist.items())
        },
        "doubling_label_distribution_default_misaligned_judge_primary": {
            m: dict(c) for m, c in sorted(doubling_dist.items())
        },
        "judge_agreement": {
            "report": {
                "n_pairs": len(report_pairs),
                "agreement_rate": ra,
                "cohens_kappa": rk,
                "confusion_primary_rows_secondary_cols": rconf,
            },
            "doubling": {
                "n_pairs": len(doubling_pairs),
                "agreement_rate": da,
                "cohens_kappa": dk,
                "confusion_primary_rows_secondary_cols": dconf,
            },
        },
        "parse_failures": {"count": parse_fails, "rate": parse_fail_rate},
        "eval_suspicion_keyword_episode_count": suspicion,
        "amount_parsed_false_count": amount_false,
        "cost_usd_by_model": dict(cost_by_model),
        "cost_usd_by_role": dict(cost_by_role),
        "cost_usd_total_from_logs_run": sum(cost_by_model.values()),
    }

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "T02_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Markdown tables
    md = ["# T02 Tables", ""]
    md.append("## 1. Misalignment rates by model × file")
    md.append("")
    md.append("| model | file | n | mis_orig | rate | Wilson95 | mis_norm rate | disagreements |")
    md.append("|---|---|---:|---:|---:|---|---:|---:|")
    for row in per_model_file:
        lo, hi = row["wilson95_original"]
        md.append(
            f"| {row['model']} | `{row['prompt_file']}` | {row['n']} | "
            f"{row['misaligned_original_count']} | {row['misalignment_rate_original']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {row['misalignment_rate_normalized']:.3f} | "
            f"{row['label_disagreement_count']} |"
        )
    md.append("")
    md.append("## 2. Report labels (default.json, misaligned, judge_primary)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary["report_label_distribution_default_misaligned_judge_primary"], indent=2))
    md.append("```")
    md.append("")
    md.append("## 3. Doubling labels (default.json, misaligned, judge_primary)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary["doubling_label_distribution_default_misaligned_judge_primary"], indent=2))
    md.append("```")
    md.append("")
    md.append("## 4. Judge agreement")
    md.append("")
    md.append(
        f"- Report: n={len(report_pairs)}, agreement={ra:.3f}, κ={rk:.3f}"
        if report_pairs
        else "- Report: no pairs"
    )
    md.append(
        f"- Doubling: n={len(doubling_pairs)}, agreement={da:.3f}, κ={dk:.3f}"
        if doubling_pairs
        else "- Doubling: no pairs"
    )
    md.append("")
    md.append("### Report confusion (primary rows × secondary cols)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(rconf, indent=2))
    md.append("```")
    md.append("")
    md.append("### Doubling confusion (primary rows × secondary cols)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(dconf, indent=2))
    md.append("```")
    md.append("")
    md.append("## 5. Diagnostics")
    md.append("")
    md.append(f"- PARSE_FAIL count/rate: {parse_fails} / {parse_fail_rate:.4f}")
    md.append(f"- eval_suspicion_keyword episodes: {suspicion}")
    md.append(f"- amount_parsed=false: {amount_false}")
    md.append("")
    md.append("## 6. Cost by model/role (run logs)")
    md.append("")
    md.append("```json")
    md.append(
        json.dumps(
            {
                "by_model": summary["cost_usd_by_model"],
                "by_role": summary["cost_usd_by_role"],
                "total": summary["cost_usd_total_from_logs_run"],
            },
            indent=2,
        )
    )
    md.append("```")
    md.append("")
    (reports / "T02_tables.md").write_text("\n".join(md), encoding="utf-8")

    # Judge review for every misaligned episode
    review = ["# T02 Judge Review", "", "Every misaligned episode (misaligned_original=true).", ""]
    mis_eps = [ep for ep in episodes if ep.get("misaligned_original")]
    mis_eps.sort(key=lambda e: (e["model_alias"], e["prompt_file"], e["sample_index"]))
    if not mis_eps:
        review.append("_No misaligned episodes in this run; nothing to audit._")
        review.append("")
    for ep in mis_eps:
        review.append(f"## {ep['episode_id']}")
        review.append("")
        review.append(f"- model: `{ep['model_alias']}`")
        review.append(f"- prompt_file: `{ep['prompt_file']}`")
        review.append(f"- sample_index: {ep['sample_index']}")
        review.append(f"- amount: {ep.get('amount')} (parsed={ep.get('amount_parsed')})")
        review.append("")
        review.append("### Report text")
        review.append("")
        review.append("```")
        review.append(ep.get("report_text") or "")
        review.append("```")
        review.append("")
        jr_p = jidx.get((ep["episode_id"], "report", "judge_primary"))
        jr_s = jidx.get((ep["episode_id"], "report", "judge_secondary"))
        review.append(
            f"- judge_primary report: {jr_p['label'] if jr_p else 'MISSING'} — {jr_p['reason'] if jr_p else ''}"
        )
        review.append(
            f"- judge_secondary report: {jr_s['label'] if jr_s else 'MISSING'} — {jr_s['reason'] if jr_s else ''}"
        )
        review.append("")
        if ep.get("doubling_text") is not None:
            review.append("### Doubling text")
            review.append("")
            review.append("```")
            review.append(ep.get("doubling_text") or "")
            review.append("```")
            review.append("")
            jd_p = jidx.get((ep["episode_id"], "doubling", "judge_primary"))
            jd_s = jidx.get((ep["episode_id"], "doubling", "judge_secondary"))
            review.append(
                f"- judge_primary doubling: {jd_p['label'] if jd_p else 'MISSING'} — {jd_p['reason'] if jd_p else ''}"
            )
            review.append(
                f"- judge_secondary doubling: {jd_s['label'] if jd_s else 'MISSING'} — {jd_s['reason'] if jd_s else ''}"
            )
            review.append("")
        else:
            review.append("_No doubling text (no trigger and/or report_found=false)._")
            review.append("")

    (reports / "T02_judge_review.md").write_text("\n".join(review), encoding="utf-8")
    print(f"Wrote {reports / 'T02_summary.json'}")
    print(f"Wrote {reports / 'T02_tables.md'}")
    print(f"Wrote {reports / 'T02_judge_review.md'}")
    print(f"episodes={len(episodes)} judgments={len(judgments)} parse_fail_rate={parse_fail_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
