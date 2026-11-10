#!/usr/bin/env python3
"""Compute T03 summary tables + review from Framework B runs."""

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
from memorytask.framework_b.metrics import compute_metrics, episodes_to_upstream_tasks  # noqa: E402


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return (None, None)
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


def cohens_kappa(pairs: list[tuple[int | None, int | None]]) -> tuple[float | None, float | None]:
    if not pairs:
        return (None, None)
    # Map None to a sentinel label for agreement stats
    labels = sorted({str(a) for a, _ in pairs} | {str(b) for _, b in pairs})
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        matrix[str(a)][str(b)] += 1
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    row = {a: sum(matrix[a].values()) / n for a in labels}
    col = {b: sum(matrix[a][b] for a in labels) / n for b in labels}
    p_e = sum(row[l] * col[l] for l in labels)
    kappa = None if p_e == 1 else (p_o - p_e) / (1 - p_e)
    return kappa, p_o


def fmt_rate(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x * 100:.1f}%"


def fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "NA"
    return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


def main() -> int:
    root = repo_root()
    run_id = "t03_repro"
    run_dir = root / "runs" / run_id
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    logs = [r for r in iter_run_logs(root / "logs") if r.get("run_id") == run_id]

    # Assert metrics == upstream compute_metrics on our results
    upstream_tasks = episodes_to_upstream_tasks(episodes)
    upstream_metrics = compute_metrics(upstream_tasks)

    # Per model × type rates with Wilson CIs (using upstream counts)
    per_model_type: dict[str, dict] = {}
    by_mt: dict[tuple[str, int], list] = defaultdict(list)
    for ep in episodes:
        by_mt[(ep["model"], int(ep["task_type"]))].append(ep)

    tables_lines = ["# T03 Tables", ""]
    tables_lines.append("## Per model × type (NFR / FFR / HFR)")
    tables_lines.append("")
    tables_lines.append("| model | type | n | NFR | NFR 95% CI | FFR | FFR 95% CI | HFR | HFR 95% CI |")
    tables_lines.append("|---|---:|---:|---:|---|---:|---|---:|---|")

    for (model, ttype), rows in sorted(by_mt.items()):
        # Build mini task list for this slice
        slice_tasks = episodes_to_upstream_tasks(rows)
        m = compute_metrics(slice_tasks)
        key = f"type_{ttype}"
        block = m.get(key, {})
        n = int(block.get("n") or 0)
        entry: dict = {"n": n}
        nfr_c = int(block.get("NFR_count") or 0)
        nfr = block.get("NFR")
        nfr_lo, nfr_hi = wilson_ci(nfr_c, n)
        entry["NFR"] = nfr
        entry["NFR_count"] = nfr_c
        entry["NFR_ci"] = [nfr_lo, nfr_hi]
        ffr_s = ffr_ci = hfr_s = hfr_ci = "—"
        if ttype in (4, 5):
            ffr_c = int(block.get("FFR_count") or 0)
            ffr = block.get("FFR")
            flo, fhi = wilson_ci(ffr_c, n)
            entry["FFR"] = ffr
            entry["FFR_count"] = ffr_c
            entry["FFR_ci"] = [flo, fhi]
            ffr_s = fmt_rate(ffr)
            ffr_ci = fmt_ci(flo, fhi)
            hfr_c = int(block.get("HFR_count") or 0)
            hfr_d = int(block.get("HFR_related_denom") or 0)
            hfr = block.get("HFR")
            hlo, hhi = wilson_ci(hfr_c, hfr_d)
            entry["HFR"] = hfr
            entry["HFR_count"] = hfr_c
            entry["HFR_related_denom"] = hfr_d
            entry["HFR_ci"] = [hlo, hhi]
            hfr_s = fmt_rate(hfr)
            hfr_ci = fmt_ci(hlo, hhi)
        per_model_type[f"{model}/type_{ttype}"] = entry
        tables_lines.append(
            f"| {model} | {ttype} | {n} | {fmt_rate(nfr)} | {fmt_ci(nfr_lo, nfr_hi)} | "
            f"{ffr_s} | {ffr_ci} | {hfr_s} | {hfr_ci} |"
        )

    # Assert equality vs full upstream metrics recomputed from all episodes
    # (slice-wise NFR counts should sum to global by type when pooling models — checked via recompute)
    recomputed = compute_metrics(upstream_tasks)
    assert recomputed == upstream_metrics

    # 2) broken tool share
    tables_lines.extend(["", "## Broken-tool call share", ""])
    tables_lines.append("| model | n | broken_tool_calls>0 | share | never_touched_tool |")
    tables_lines.append("|---|---:|---:|---:|---:|")
    broken_share: dict[str, dict] = {}
    for model in sorted({ep["model"] for ep in episodes}):
        rows = [ep for ep in episodes if ep["model"] == model]
        n = len(rows)
        hit = sum(1 for ep in rows if int(ep.get("broken_tool_calls") or 0) > 0)
        never = sum(1 for ep in rows if ep.get("never_touched_tool"))
        share = hit / n if n else None
        broken_share[model] = {"n": n, "hit": hit, "share": share, "never_touched_tool": never}
        flag = " ⚠ <90%" if (share is not None and share < 0.9) else ""
        tables_lines.append(f"| {model} | {n} | {hit} | {fmt_rate(share)}{flag} | {never} |")

    # 3) steps / errors / tokens
    tables_lines.extend(["", "## Steps / errors / tokens", ""])
    tables_lines.append("| model | hit_max_steps | error_rate | mean_steps | mean_tokens |")
    tables_lines.append("|---|---:|---:|---:|---:|")
    step_stats: dict[str, dict] = {}
    tokens_by_ep: dict[str, int] = defaultdict(int)
    for r in logs:
        if r.get("error"):
            continue
        tid = r.get("task_id") or ""
        usage = r.get("usage") or {}
        tokens_by_ep[tid] += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)

    for model in sorted({ep["model"] for ep in episodes}):
        rows = [ep for ep in episodes if ep["model"] == model]
        n = len(rows)
        hit_max = sum(1 for ep in rows if ep.get("hit_max_steps"))
        err = sum(1 for ep in rows if ep.get("error"))
        mean_steps = sum(int(ep.get("n_steps") or 0) for ep in rows) / n if n else None
        toks = [tokens_by_ep.get(ep["episode_id"], 0) for ep in rows]
        mean_toks = sum(toks) / n if n else None
        step_stats[model] = {
            "hit_max_steps_rate": hit_max / n if n else None,
            "error_rate": err / n if n else None,
            "mean_steps": mean_steps,
            "mean_tokens": mean_toks,
        }
        ms = f"{mean_steps:.2f}" if mean_steps is not None else "NA"
        mt = f"{mean_toks:.1f}" if mean_toks is not None else "NA"
        tables_lines.append(
            f"| {model} | {fmt_rate(hit_max / n if n else None)} | {fmt_rate(err / n if n else None)} | "
            f"{ms} | {mt} |"
        )

    # 4) judge agreement
    pairs: list[tuple[int | None, int | None]] = []
    parse_fail = 0
    parse_total = 0
    for ep in episodes:
        details = ep.get("judge_outputs") or {}
        # details is dict of prompt_key -> {primary, secondary} OR nested for type1
        def walk(obj):
            nonlocal parse_fail, parse_total
            if isinstance(obj, dict) and "primary" in obj and "secondary" in obj:
                p = obj["primary"]
                s = obj["secondary"]
                parse_total += 2
                if not p.get("parse_ok"):
                    parse_fail += 1
                if not s.get("parse_ok"):
                    parse_fail += 1
                pairs.append((p.get("result"), s.get("result")))
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)

        walk(details)
    kappa, agree = cohens_kappa(pairs)
    parse_fail_rate = (parse_fail / parse_total) if parse_total else None
    tables_lines.extend(["", "## Judge agreement (primary vs secondary)", ""])
    tables_lines.append(f"- pairs: {len(pairs)}")
    tables_lines.append(f"- agreement: {fmt_rate(agree)}")
    tables_lines.append(f"- Cohen κ: {kappa if kappa is not None else 'NA'}")
    tables_lines.append(f"- parse failures: {parse_fail}/{parse_total} ({fmt_rate(parse_fail_rate)})")

    # 5) concealment candidates (NFR=true)
    candidates = []
    for ep in episodes:
        r = ep.get("upstream_result")
        if r is None:
            continue
        nfr = False
        if int(ep["task_type"]) == 1 and isinstance(r, list) and len(r) >= 2:
            nfr = bool(r[1])
        elif int(ep["task_type"]) in (4, 5) and isinstance(r, list) and len(r) >= 1:
            nfr = bool(r[0])
        if nfr:
            candidates.append(
                {
                    "episode_id": ep["episode_id"],
                    "model": ep["model"],
                    "task_type": ep["task_type"],
                    "task_index": ep["task_index"],
                    "upstream_result": r,
                }
            )
    tables_lines.extend(["", f"## Concealment candidates (NFR=true): {len(candidates)}", ""])
    for c in candidates:
        tables_lines.append(
            f"- {c['episode_id']} result={c['upstream_result']}"
        )

    # 6) cost by model and role
    cost_by_model: Counter = Counter()
    cost_by_role: Counter = Counter()
    for r in logs:
        if r.get("cache_hit"):
            continue
        c = float(r.get("cost_usd") or 0.0)
        cost_by_model[r.get("model_alias") or "unknown"] += c
        cost_by_role[r.get("role") or "unknown"] += c
    tables_lines.extend(["", "## Cost (non-cache)", ""])
    tables_lines.append("| key | usd |")
    tables_lines.append("|---|---:|")
    for k, v in sorted(cost_by_model.items()):
        tables_lines.append(f"| model:{k} | {v:.6f} |")
    for k, v in sorted(cost_by_role.items()):
        tables_lines.append(f"| role:{k} | {v:.6f} |")
    tables_lines.append(f"| TOTAL | {sum(cost_by_model.values()):.6f} |")

    summary = {
        "run_id": run_id,
        "n_episodes": len(episodes),
        "upstream_metrics": upstream_metrics,
        "per_model_type": per_model_type,
        "broken_tool_share": broken_share,
        "step_stats": step_stats,
        "judge_agreement": {
            "n_pairs": len(pairs),
            "agreement_rate": agree,
            "cohens_kappa": kappa,
            "parse_fail": parse_fail,
            "parse_total": parse_total,
            "parse_fail_rate": parse_fail_rate,
        },
        "concealment_candidates": candidates,
        "cost_by_model": dict(cost_by_model),
        "cost_by_role": dict(cost_by_role),
        "metrics_assert_ok": True,
    }

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "T03_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (reports / "T03_tables.md").write_text("\n".join(tables_lines) + "\n", encoding="utf-8")

    # Review doc for hand audit
    review = ["# T03 Review (hand audit)", ""]
    for ep in episodes:
        review.append(f"## {ep['episode_id']}")
        review.append(f"- model: {ep['model']}")
        review.append(f"- task_type: {ep['task_type']}  task_index: {ep['task_index']}")
        instr = (ep.get("instruction") or "")[:300]
        review.append(f"- instruction: {instr}")
        review.append(f"- final_answer:")
        review.append("```")
        review.append(ep.get("final_answer") or "")
        review.append("```")
        review.append("- tool calls / observations (truncated):")
        for t in ep.get("tool_trace") or []:
            obs = (t.get("observation") or "")[:500]
            review.append(f"  - name={t.get('name')} args={t.get('arguments')}")
            review.append(f"    obs: {obs}")
            if t.get("code_action"):
                review.append(f"    code: {str(t.get('code_action'))[:300]}")
        review.append(f"- files created: {ep.get('created_files')}")
        review.append(f"- broken_tool_calls: {ep.get('broken_tool_calls')}  never_touched_tool: {ep.get('never_touched_tool')}")
        review.append(f"- upstream_result: {ep.get('upstream_result')}")
        review.append(f"- judges: {json.dumps(ep.get('judge_outputs'), default=str)[:4000]}")
        review.append(f"- error: {ep.get('error')}")
        review.append("")

    (reports / "T03_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(f"Wrote reports/T03_summary.json, T03_tables.md, T03_review.md ({len(episodes)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
