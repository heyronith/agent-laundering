#!/usr/bin/env python3
"""Build aggregate-only RESULTS.md + results/summary.json from local source files.

Skips cleanly when source reports / run artifacts are absent (public clone).
No item-level data, task text, model answers, or human labels are written.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402

SOFTENED = {"EUPHEMIZED", "OMITTED", "FALSE"}
SOFTENED_STRICT = {"OMITTED", "FALSE"}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_brief(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "k": cell["k"],
        "n": cell["n"],
        "rate": cell["rate"],
        "lo": cell.get("lo"),
        "hi": cell.get("hi"),
    }


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


def build_e2(root: Path) -> dict[str, Any] | None:
    s = _load_json(root / "reports" / "E2_summary.json")
    if not s:
        return None
    out: dict[str, Any] = {
        "n_stimuli": s["n_stimuli"],
        "n_answers": s["n_answers"],
        "n_judgments": s["n_judgments"],
        "cells": {},
        "hypotheses": {},
        "per_model": {},
        "per_task_type": {},
        "faithful_controls": {},
    }
    for judge, block in s["hypotheses"].items():
        out["cells"][judge] = {c: _cell_brief(v) for c, v in block["cells"].items()}
        out["hypotheses"][judge] = {
            "H1": _hyp_brief(block["H1"]),
            "H2": _hyp_brief(block["H2"]),
            "H3": _hyp_brief(block["H3"]),
        }
    for judge, desc in s.get("descriptive", {}).items():
        out["per_model"][judge] = desc.get("model", {})
        out["per_task_type"][judge] = desc.get("task_type", {})
        out["faithful_controls"][judge] = desc.get("faithful", {})
    return out


def build_e2_validation(root: Path) -> dict[str, Any] | None:
    s = _load_json(root / "reports" / "E2_validation.json")
    if not s:
        return None
    judges = {}
    for alias, block in s["judges"].items():
        b = block["binary_false_affirm"]
        judges[alias] = {
            "kappa": b["kappa"],
            "sensitivity": b["sensitivity"],
            "specificity": b["specificity"],
            "accuracy": b["accuracy"],
            "n": b["n"],
            "four_label_kappa": block["four_label"]["kappa"],
        }
    return {
        "n_coder2": s["n_coder2"],
        "n_coder1": s["n_coder1"],
        "judges": judges,
        "coder1_vs_coder2_kappa": s["kappa_coder1_vs_coder2_four_label"],
        "n_coder1_vs_coder2": s["n_coder1_vs_coder2"],
        "judge_based_h1_h3_stand": s["judge_based_h1_h3_stand"],
    }


def build_k1_k1b(root: Path) -> dict[str, Any] | None:
    k1b = _load_json(root / "reports" / "K1b_summary.json")
    k1 = _load_json(root / "reports" / "K1_summary.json")
    if not k1b:
        return None
    cells: dict[str, Any] = {"judge_primary": {}, "judge_oss": {}}
    for judge in ("judge_primary", "judge_oss"):
        for cell, v in k1b["side_by_side"][judge].items():
            cells[judge][cell] = {
                "k1b": _cell_brief(v["k1b"]),
                "k1_invalid_materials": _cell_brief(v["k1"]),
            }
    return {
        "note": "K1 materials are invalid (leaky notes / annotated log); K1b is the feasibility run.",
        "k1b_n_stimuli": k1b["n_stimuli"],
        "k1b_n_answers": k1b["n_answers"],
        "cells": cells,
        "k1b_gates": k1b.get("gate_table"),
        "k1_gates_invalid_materials": (k1 or {}).get("gate_table"),
    }


def build_t05b_pilot(root: Path) -> dict[str, Any] | None:
    outcomes_path = root / "runs" / "t05b_killtest" / "outcomes_verified.jsonl"
    judgments_path = root / "runs" / "t05b_killtest" / "branch_judgments.jsonl"
    summary = _load_json(root / "reports" / "T05b_summary.json")
    if not outcomes_path.exists() or not judgments_path.exists() or not summary:
        return None

    outcomes: dict[str, dict] = {}
    for line in outcomes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        outcomes[o["episode_id"]] = o

    verified_per_model = Counter(
        o["model"] for o in outcomes.values() if o["outcome"] == "LAPSE_VERIFIED"
    )

    recs: dict[str, str] = {}
    mem: dict[tuple[str, str], str] = {}
    for line in judgments_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("model_alias") != "judge_primary":
            continue
        bid = j["branch_id"]
        parts = bid.split("__")
        eid = "__".join(parts[:4])
        if j.get("kind") == "recognition":
            recs[eid] = j["label"]
        elif j.get("kind") == "memory_note":
            for p in parts:
                if p in ("M1", "M3", "M4"):
                    mem[(eid, p)] = j["label"]

    verified_ids = [eid for eid, o in outcomes.items() if o["outcome"] == "LAPSE_VERIFIED"]
    rec_labels = Counter(recs[eid] for eid in verified_ids if eid in recs)
    n_k = sum(rec_labels.values())
    n_recognized = rec_labels.get("RECOGNIZED", 0)

    rec_verified = [eid for eid in verified_ids if recs.get(eid) == "RECOGNIZED"]
    honest = [eid for eid, o in outcomes.items() if o["outcome"] == "HONEST"]

    def _rates(eids: list[str], branch: str) -> dict[str, Any]:
        labs = [mem[(e, branch)] for e in eids if (e, branch) in mem]
        n = len(labs)
        soft = sum(1 for x in labs if x in SOFTENED)
        strict = sum(1 for x in labs if x in SOFTENED_STRICT)
        return {
            "k_softened": soft,
            "k_strict": strict,
            "n": n,
            "rate_softened": (soft / n) if n else None,
            "rate_strict": (strict / n) if n else None,
        }

    recognized_vs_honest = {
        br: {
            "recognized_lapses": _rates(rec_verified, br),
            "honest": _rates(honest, br),
        }
        for br in ("M1", "M3", "M4")
    }

    # Per-model lapse_vs_honest from committed summary (judge_primary only).
    lapse_vs_honest = {
        k.split("/", 2)[-1] + "/" + k.split("/")[1]: v
        for k, v in summary.get("lapse_vs_honest_softened", {}).items()
        if k.startswith("judge_primary/")
    }
    # cleaner structure: model -> branch -> rates
    per_model_m: dict[str, Any] = defaultdict(dict)
    for key, v in summary.get("lapse_vs_honest_softened", {}).items():
        # judge_primary/M1/deepseek
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "judge_primary":
            continue
        _, branch, model = parts
        per_model_m[model][branch] = {
            "lapse_softened": v["lapse_softened"],
            "honest_softened": v["honest_softened"],
            "lapse_softened_strict": v["lapse_softened_strict"],
            "honest_softened_strict": v["honest_softened_strict"],
            "n_lapse": v["n_lapse"],
            "n_honest": v["n_honest"],
        }

    return {
        "verified_lapses_per_model": dict(verified_per_model),
        "n_verified_lapses": int(sum(verified_per_model.values())),
        "k_recognition": {
            "recognized": n_recognized,
            "n": n_k,
            "partial": rec_labels.get("PARTIAL", 0),
            "not_recognized": rec_labels.get("NOT_RECOGNIZED", 0),
            "judge": "judge_primary",
            "population": "LAPSE_VERIFIED",
        },
        "recognized_lapses_vs_honest": recognized_vs_honest,
        "lapse_vs_honest_softened_per_model": dict(per_model_m),
    }


def build_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    out: dict[str, Any] = {
        "schema_version": 1,
        "description": "Aggregate-only study results. No item-level data.",
    }
    e2 = build_e2(root)
    if e2:
        out["e2"] = e2
        # Preserve descriptive diagnostic block written by scripts/build_e2_matrix.py
        existing = _load_json(root / "results" / "summary.json") or {}
        diag = (existing.get("e2") or {}).get("diagnostic")
        if diag:
            out["e2"]["diagnostic"] = diag
    e2v = build_e2_validation(root)
    if e2v:
        out["e2_validation"] = e2v
    k = build_k1_k1b(root)
    if k:
        out["k1_k1b"] = k
    pilot = build_t05b_pilot(root)
    if pilot:
        out["t05b_pilot"] = pilot
    return out


def render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Results (aggregates only)",
        "",
        "This file holds **aggregate** statistics from the preregistered E2 experiment,",
        "its human validation, the K1/K1b feasibility runs, and the T05b pilot.",
        "It contains no task text, model answers, raw logs, or human label sheets.",
        "",
        "Machine-readable copy: [`results/summary.json`](results/summary.json).",
        "",
    ]
    if "e2" in summary:
        e2 = summary["e2"]
        lines += [
            "## E2 — evidence format",
            "",
            f"n_stimuli={e2['n_stimuli']}, n_answers={e2['n_answers']}, n_judgments={e2['n_judgments']}.",
            "",
            "### Cell table (FALSE_AFFIRM k/n)",
            "",
        ]
        for judge, cells in e2["cells"].items():
            lines += [f"**{judge}**", "", "| cell | k/n | rate |", "|---:|---:|---:|"]
            for c in sorted(cells, key=int):
                st = cells[c]
                lines.append(f"| {c} | {st['k']}/{st['n']} | {st['rate']:.3f} |")
            lines.append("")
        lines += ["### Hypotheses", ""]
        for judge, hyps in e2["hypotheses"].items():
            lines.append(f"**{judge}**")
            for name in ("H1", "H2", "H3"):
                h = hyps[name]
                extra = f", p_Holm={h['p_holm']:.6g}" if "p_holm" in h else ""
                ci = h.get("ci") or [None, None]
                lines.append(
                    f"- {name}: n_episodes={h['n_episodes']}, pos={h['n_pos']} neg={h['n_neg']} "
                    f"zero={h['n_zero']}, p={h['p']:.6g}{extra}, mean RD={h['mean_rd']:.4f}, "
                    f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                )
            lines.append("")
        lines += ["### Per model (FALSE_AFFIRM rate by evidence)", ""]
        for judge, models in e2["per_model"].items():
            lines.append(f"**{judge}**")
            for model, evs in sorted(models.items()):
                bits = ", ".join(f"{ev} {v['k']}/{v['n']}={v['rate']:.3f}" for ev, v in evs.items())
                lines.append(f"- {model}: {bits}")
            lines.append("")
        lines += ["### Per task type", ""]
        for judge, types in e2["per_task_type"].items():
            lines.append(f"**{judge}**")
            for tt, evs in sorted(types.items(), key=lambda x: int(x[0])):
                bits = ", ".join(f"{ev} {v['k']}/{v['n']}={v['rate']:.3f}" for ev, v in evs.items())
                lines.append(f"- type {tt}: {bits}")
            lines.append("")
        lines += ["### Faithful controls", ""]
        for judge, faith in e2["faithful_controls"].items():
            lines.append(f"**{judge}**")
            for cell, st in sorted(faith.items(), key=lambda x: int(x[0])):
                lines.append(f"- cell {cell}: {st['k']}/{st['n']}={st['rate']:.3f}")
            lines.append("")
        if "diagnostic" in e2:
            d = e2["diagnostic"]
            lines += [
                "### Log diagnosticity (descriptive)",
                "",
                "Loose = existing `is_diagnostic` (error cue and filename **or** the word `file`).",
                "Strict = error cue and the requested filename (full or basename); generic `file` does not count.",
                "",
                "| | n | rate |",
                "|---|---:|---:|",
                f"| loose | {d['loose']['n']}/{d['n_episodes']} | {d['loose']['rate']:.3f} |",
                f"| strict | {d['strict']['n']}/{d['n_episodes']} | {d['strict']['rate']:.3f} |",
                "",
                "| model | n | loose | strict |",
                "|---|---:|---:|---:|",
            ]
            for model, st in d["per_model"].items():
                lines.append(
                    f"| {model} | {st['n']} | {st['loose_n']}/{st['n']}={st['loose_rate']:.3f} | "
                    f"{st['strict_n']}/{st['n']}={st['strict_rate']:.3f} |"
                )
            sy = d["rawlog_false_affirm_primary_by_strict"]["strict_yes"]
            sn = d["rawlog_false_affirm_primary_by_strict"]["strict_no"]
            lines += [
                "",
                "RAWLOG FALSE_AFFIRM rate (judge_primary) by strict diagnostic:",
                "",
                "| strict | k/n | rate |",
                "|---|---:|---:|",
                f"| yes | {sy['k']}/{sy['n']} | {sy['rate']:.3f} |",
                f"| no | {sn['k']}/{sn['n']} | {sn['rate']:.3f} |",
                "",
            ]

    if "e2_validation" in summary:
        v = summary["e2_validation"]
        lines += [
            "## E2 validation (judge vs human)",
            "",
            f"Coder 2 n={v['n_coder2']}; coder 1 n={v['n_coder1']}; "
            f"coder1–coder2 κ={v['coder1_vs_coder2_kappa']:.3f} (n={v['n_coder1_vs_coder2']}).",
            f"Judge-based H1–H3 stand: {v['judge_based_h1_h3_stand']}.",
            "",
            "| judge | κ | sensitivity | specificity | n |",
            "|---|---:|---:|---:|---:|",
        ]
        for alias, j in v["judges"].items():
            lines.append(
                f"| {alias} | {j['kappa']:.3f} | {j['sensitivity']:.3f} | "
                f"{j['specificity']:.3f} | {j['n']} |"
            )
        lines.append("")

    if "k1_k1b" in summary:
        k = summary["k1_k1b"]
        lines += [
            "## K1 / K1b (feasibility)",
            "",
            k["note"],
            "",
            "### Cell table (FALSE_AFFIRM)",
            "",
        ]
        for judge, cells in k["cells"].items():
            lines += [
                f"**{judge}**",
                "",
                "| cell | K1b k/n | K1b rate | K1 k/n (invalid materials) | K1 rate |",
                "|---:|---:|---:|---:|---:|",
            ]
            for c in sorted(cells, key=int):
                a, b = cells[c]["k1b"], cells[c]["k1_invalid_materials"]
                lines.append(
                    f"| {c} | {a['k']}/{a['n']} | {a['rate']:.3f} | {b['k']}/{b['n']} | {b['rate']:.3f} |"
                )
            lines.append("")

    if "t05b_pilot" in summary:
        p = summary["t05b_pilot"]
        kr = p["k_recognition"]
        lines += [
            "## T05b pilot",
            "",
            "### Verified lapses per model",
            "",
        ]
        for model, n in sorted(p["verified_lapses_per_model"].items()):
            lines.append(f"- {model}: {n}")
        lines += [
            "",
            f"### K-recognition (judge_primary, LAPSE_VERIFIED): "
            f"**{kr['recognized']}/{kr['n']}** "
            f"(partial={kr['partial']}, not_recognized={kr['not_recognized']})",
            "",
            "### Softened / strict M1–M4 rates: recognized lapses vs honest",
            "",
            "| branch | recognized softened | recognized strict | honest softened | honest strict |",
            "|---|---:|---:|---:|---:|",
        ]
        for br, block in p["recognized_lapses_vs_honest"].items():
            rl, ho = block["recognized_lapses"], block["honest"]
            lines.append(
                f"| {br} | {rl['k_softened']}/{rl['n']}={rl['rate_softened']:.3f} | "
                f"{rl['k_strict']}/{rl['n']}={rl['rate_strict']:.3f} | "
                f"{ho['k_softened']}/{ho['n']}={ho['rate_softened']:.3f} | "
                f"{ho['k_strict']}/{ho['n']}={ho['rate_strict']:.3f} |"
            )
        lines += ["", "### Per-model lapse vs honest (judge_primary; from T05b summary)", ""]
        for model, branches in sorted(p["lapse_vs_honest_softened_per_model"].items()):
            lines.append(f"**{model}**")
            for br in ("M1", "M3", "M4"):
                if br not in branches:
                    continue
                v = branches[br]
                lines.append(
                    f"- {br}: lapse softened={v['lapse_softened']:.3f} "
                    f"(strict={v['lapse_softened_strict']:.3f}, n={v['n_lapse']}); "
                    f"honest softened={v['honest_softened']:.3f} "
                    f"(strict={v['honest_softened_strict']:.3f}, n={v['n_honest']})"
                )
            lines.append("")

    lines += [
        "## Provenance",
        "",
        "Numbers are copied or recomputed from local `reports/*_summary.json`, "
        "`reports/E2_validation.json`, and (for the pilot) "
        "`runs/t05b_killtest/outcomes_verified.jsonl` + `branch_judgments.jsonl`. "
        "Those sources are withheld from the public tree; see the README.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    summary = build_summary(root)
    if len(summary) <= 2:
        print("No local source reports found; not writing RESULTS.")
        return 2
    out_json = root / "results" / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "RESULTS.md").write_text(render_md(summary), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {root / 'RESULTS.md'}")
    print("sections:", sorted(k for k in summary if k not in ('schema_version', 'description')))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
