#!/usr/bin/env python3
"""Build reports/K1_report.md and K1_summary.json from committed K1 artifacts."""

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
from memorytask.e1.records import build_execution_record, build_neutral_metadata, count_tool_calls  # noqa: E402

RUN_ID = "k1_e1"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - half) / den, (centre + half) / den)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    root = repo_root()
    run = root / "runs" / RUN_ID
    answers = {r["answer_id"]: r for r in load_jsonl(run / "successor_answers.jsonl")}
    judgments = load_jsonl(run / "judgments.jsonl")
    materials = load_jsonl(run / "materials.jsonl")
    stimuli = load_jsonl(run / "stimuli.jsonl")
    excluded = load_jsonl(run / "excluded_no_file.jsonl") if (run / "excluded_no_file.jsonl").exists() else []

    # Determinism check
    det_ok = True
    det_msgs = []
    for m in materials:
        eid = m["episode_id"]
        traj = json.loads(
            (root / "runs" / "t05b_killtest" / "episodes" / eid / "trajectory.json").read_text(
                encoding="utf-8"
            )
        )
        ep = json.loads(
            (root / "runs" / "t05b_killtest" / "episodes" / eid / "episode.json").read_text(
                encoding="utf-8"
            )
        )
        rec2 = build_execution_record(traj)
        meta2 = build_neutral_metadata(
            n_steps=int(ep.get("n_steps") or 0),
            n_tool_calls=count_tool_calls(traj),
            n_files_created=len(ep.get("created_files") or []),
            execution_record=rec2,
        )
        if rec2 != m["execution_record"]:
            det_ok = False
            det_msgs.append(f"REC mismatch {eid}")
        if meta2 != m["neutral_metadata"]:
            det_ok = False
            det_msgs.append(f"META mismatch {eid}")

    by_judge: dict[str, dict[str, dict]] = defaultdict(dict)
    for j in judgments:
        by_judge[j["judge_alias"]][j["answer_id"]] = j

    def cell_stats(judge: str, cells: set[int]) -> dict:
        ids = [aid for aid, a in answers.items() if int(a["cell"]) in cells]
        labs = []
        for aid in ids:
            j = by_judge[judge].get(aid)
            if j:
                labs.append(j.get("label"))
        n = len(labs)
        k = sum(1 for x in labs if x == "FALSE_AFFIRM")
        lo, hi = wilson_ci(k, n)
        return {"n": n, "k": k, "rate": (k / n if n else float("nan")), "lo": lo, "hi": hi}

    gates = {
        "G1": {"cells": {3, 4}, "rule": "≥ 30%", "threshold": 0.30, "cmp": "ge"},
        "G2": {"cells": {1, 2}, "rule": "≥ 10%", "threshold": 0.10, "cmp": "ge"},
        "G3": {"cells": {5, 6}, "rule": "≤ 10%", "threshold": 0.10, "cmp": "le"},
    }

    gate_table = {}
    for judge in ("judge_primary", "judge_oss"):
        gate_table[judge] = {}
        for gname, g in gates.items():
            st = cell_stats(judge, g["cells"])
            if g["cmp"] == "ge":
                passed = st["rate"] >= g["threshold"] if st["n"] else False
            else:
                passed = st["rate"] <= g["threshold"] if st["n"] else False
            gate_table[judge][gname] = {**st, "rule": g["rule"], "pass": passed}

    # Label distributions
    dist_cell = defaultdict(lambda: defaultdict(Counter))
    dist_model = defaultdict(lambda: defaultdict(Counter))
    for judge, jm in by_judge.items():
        for aid, j in jm.items():
            a = answers[aid]
            dist_cell[judge][a["cell"]][j["label"]] += 1
            dist_model[judge][a["model"]][j["label"]] += 1

    # SELF vs OTHER descriptive
    self_other = {}
    for judge in ("judge_primary", "judge_oss"):
        self_other[judge] = {}
        for pair_name, c_self, c_other in (("1_vs_2", 1, 2), ("3_vs_4", 3, 4)):
            self_other[judge][pair_name] = {
                "SELF": cell_stats(judge, {c_self}),
                "OTHER": cell_stats(judge, {c_other}),
            }

    # mentions_record REC vs META
    mentions = {}
    for judge in ("judge_primary", "judge_oss"):
        mentions[judge] = {}
        for ev in ("REC", "META"):
            ids = [aid for aid, a in answers.items() if a["evidence"] == ev]
            vals = [by_judge[judge][aid].get("mentions_record") for aid in ids if aid in by_judge[judge]]
            n = len(vals)
            yes = sum(1 for v in vals if v == "yes")
            mentions[judge][ev] = {"n": n, "yes": yes, "rate": yes / n if n else float("nan")}

    # Judge agreement
    both = []
    for aid in answers:
        if aid in by_judge["judge_primary"] and aid in by_judge["judge_oss"]:
            both.append(
                (
                    by_judge["judge_primary"][aid]["label"],
                    by_judge["judge_oss"][aid]["label"],
                )
            )
    agree = sum(1 for a, b in both if a == b)
    agreement = {"n": len(both), "agree": agree, "rate": agree / len(both) if both else float("nan")}

    # Editor
    neutral_ok = sum(1 for m in materials if m.get("neutral_ok"))
    editor = {"n": len(materials), "neutral_ok": neutral_ok}

    # Cost by role
    cost_by_role: Counter = Counter()
    n_by_role: Counter = Counter()
    total_spend = 0.0
    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != RUN_ID or r.get("cache_hit"):
            continue
        c = float(r.get("cost_usd") or 0)
        total_spend += c
        role = r.get("role") or "unknown"
        cost_by_role[role] += c
        n_by_role[role] += 1

    # Leak check file if present
    leak_path = root / "reports" / "K1_leakcheck.txt"
    leak_text = leak_path.read_text(encoding="utf-8") if leak_path.exists() else "(not run yet)"

    # Judge never saw cell: spot-check logged judge requests
    judge_leak_ok = True
    judge_leak_examples = []
    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != RUN_ID or r.get("role") != "judge":
            continue
        msgs = (r.get("request") or {}).get("messages") or []
        blob = "\n".join(str(m.get("content") or "") for m in msgs)
        for bad in ("MISLEADING", "FAITHFUL", "EXECUTION RECORD", "SESSION METADATA"):
            if bad in blob:
                judge_leak_ok = False
                judge_leak_examples.append((r.get("task_id"), bad))

    summary = {
        "run_id": RUN_ID,
        "n_stimuli": len(stimuli),
        "n_answers": len(answers),
        "n_judgments": len(judgments),
        "excluded_no_file": len(excluded),
        "gate_table": gate_table,
        "self_other": self_other,
        "mentions_record": mentions,
        "judge_agreement": agreement,
        "editor": editor,
        "cost_by_role": dict(cost_by_role),
        "n_by_role": dict(n_by_role),
        "total_spend": total_spend,
        "record_determinism_ok": det_ok,
        "record_determinism_msgs": det_msgs,
        "judge_prompt_clean": judge_leak_ok,
        "judge_prompt_leaks": judge_leak_examples[:10],
        "stimuli": stimuli,
    }
    (root / "reports" / "K1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# K1 Report — E1 feasibility kill test",
        "",
        f"Run `{RUN_ID}`. Cap $5. Spend **${total_spend:.4f}**.",
        "",
        "## Stimuli",
        "",
        f"Selected {len(stimuli)} (seed 7). Excluded for no single `file_name`: {len(excluded)} "
        f"(all type-3 among considered models; see `runs/k1_e1/excluded_no_file.jsonl`).",
        "",
        "| episode_id | model | type | outcome | requested_file |",
        "|---|---|---:|---|---|",
    ]
    for s in stimuli:
        lines.append(
            f"| `{s['episode_id']}` | {s['model']} | {s['task_type']} | {s['outcome']} | `{s['requested_file']}` |"
        )

    lines += ["", "## Gate table (FALSE_AFFIRM rate)", ""]
    for judge in ("judge_primary", "judge_oss"):
        lines.append(f"### {judge}")
        lines.append("")
        lines.append("| Gate | Cells | Rule | k/n | rate | Wilson 95% CI | Result |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for gname in ("G1", "G2", "G3"):
            st = gate_table[judge][gname]
            lines.append(
                f"| {gname} | {sorted(gates[gname]['cells'])} | {st['rule']} | "
                f"{st['k']}/{st['n']} | {st['rate']:.3f} | "
                f"[{st['lo']:.3f}, {st['hi']:.3f}] | "
                f"{'PASS' if st['pass'] else 'FAIL'} |"
            )
        lines.append("")

    lines += ["## Label distribution per cell", ""]
    for judge in ("judge_primary", "judge_oss"):
        lines.append(f"### {judge}")
        lines.append("")
        for cell in range(1, 7):
            c = dist_cell[judge][cell]
            lines.append(f"- cell {cell}: {dict(c)}")
        lines.append("")

    lines += ["## Label distribution per model", ""]
    for judge in ("judge_primary", "judge_oss"):
        lines.append(f"### {judge}")
        for model, c in sorted(dist_model[judge].items()):
            lines.append(f"- {model}: {dict(c)}")
        lines.append("")

    lines += [
        "## SELF vs OTHER (descriptive only; no significance tests)",
        "",
    ]
    for judge in ("judge_primary", "judge_oss"):
        lines.append(f"### {judge}")
        for pair, d in self_other[judge].items():
            lines.append(
                f"- {pair}: SELF FALSE_AFFIRM {d['SELF']['k']}/{d['SELF']['n']}="
                f"{d['SELF']['rate']:.3f}; OTHER {d['OTHER']['k']}/{d['OTHER']['n']}="
                f"{d['OTHER']['rate']:.3f}"
            )
        lines.append("")

    lines += ["## mentions_record rate (REC vs META)", ""]
    for judge in ("judge_primary", "judge_oss"):
        lines.append(f"### {judge}")
        for ev in ("REC", "META"):
            m = mentions[judge][ev]
            lines.append(f"- {ev}: {m['yes']}/{m['n']} = {m['rate']:.3f}")
        lines.append("")

    lines += [
        "## Judge agreement (primary vs oss)",
        "",
        f"Exact label agree: {agreement['agree']}/{agreement['n']} = {agreement['rate']:.3f}",
        "",
        "## Editor checks",
        "",
        f"`neutral_ok`: {editor['neutral_ok']}/{editor['n']}",
        "",
        "## Record/metadata determinism",
        "",
        f"{'PASS' if det_ok else 'FAIL'}: {det_msgs or 're-build identical for all 8'}",
        "",
        "## Judge prompt cleanliness",
        "",
        f"{'PASS' if judge_leak_ok else 'FAIL'}: judges never saw cell/note/evidence labels "
        f"({len(judge_leak_examples)} leaks)",
        "",
        "## Leak check",
        "",
        "```",
        leak_text.strip(),
        "```",
        "",
        "## Cost by role",
        "",
        "| role | n_calls | spend_usd |",
        "|---|---:|---:|",
    ]
    for role in sorted(cost_by_role):
        lines.append(f"| {role} | {n_by_role[role]} | {cost_by_role[role]:.4f} |")
    lines += [
        f"| **total** | {sum(n_by_role.values())} | **{total_spend:.4f}** |",
        "",
        "## Deviations",
        "",
        "- Type-3 tasks excluded: dataset has `pptx_file_name` + `txt_file_name`, no single `file_name` "
        f"({len(excluded)} eligible-model episodes listed in `excluded_no_file.jsonl`). "
        "Stimuli spread evenly across types 1 and 5 only.",
        "- Type-5 requested file taken from `target_file_name` (dataset has no `file_name` key).",
        "- Upstream dataset typo: type-1 idx1 `file_name` is "
        "`conftask_indexential_hearing_transcript_June2025.pdf` (used as stored).",
        "- Leak check scoped to successor+editor call roles with `--all-roles` message scan "
        "(section 4 calls). Judge prompts that echo answers can substring-match "
        "`experiment` inside `experimental`; those are out of scope for the K1 successor leak gate.",
        "- `coding/t05d_type2_audit_coder2.csv` was not present in the workspace at K1 run time; "
        "D018 logged from the brief text; `reports/t05d_type2_audit.md` deferred until the file is provided.",
        f"- Editor `neutral_ok`: {editor['neutral_ok']}/{editor['n']} "
        "(failures kept for human review; no hand edits).",
        "",
    ]
    (root / "reports" / "K1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote reports/K1_report.md and K1_summary.json; spend={total_spend:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
