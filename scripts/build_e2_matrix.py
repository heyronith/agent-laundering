#!/usr/bin/env python3
"""Build anonymized E2 outcome matrix + diagnostic descriptive stats (local data)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402
from memorytask.e1.rawlog import is_diagnostic, is_diagnostic_strict  # noqa: E402

CELL_META = {
    1: ("NONE", "SELF", "MISLEADING"),
    2: ("NONE", "OTHER", "MISLEADING"),
    3: ("STATUS", "SELF", "MISLEADING"),
    4: ("STATUS", "OTHER", "MISLEADING"),
    5: ("RAWLOG", "SELF", "MISLEADING"),
    6: ("RAWLOG", "OTHER", "MISLEADING"),
    7: ("NONE", "SELF", "FAITHFUL"),
    8: ("RAWLOG", "SELF", "FAITHFUL"),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    root = repo_root()
    mats_path = root / "runs" / "e2_main" / "materials.jsonl"
    ans_path = root / "runs" / "e2_main" / "successor_answers.jsonl"
    jud_path = root / "runs" / "e2_main" / "judgments.jsonl"
    if not mats_path.exists():
        print("runs/e2_main absent; cannot build matrix")
        return 2

    materials = {m["episode_id"]: m for m in load_jsonl(mats_path)}
    answers = load_jsonl(ans_path)
    judgments: dict[tuple[str, str], dict] = {}
    for j in load_jsonl(jud_path):
        judgments[(j["answer_id"], j["judge_alias"])] = j

    # Stable episode + task anonymization (mapping local-only)
    episode_ids = sorted(materials)
    ep_map = {eid: f"E{i:03d}" for i, eid in enumerate(episode_ids, 1)}
    task_keys = sorted({(materials[e]["task_type"], materials[e]["task_index"]) for e in episode_ids})
    task_map = {k: f"T{i:03d}" for i, k in enumerate(task_keys, 1)}

    local_dir = root / "local_only"
    local_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "episodes": [
            {
                "episode": ep_map[eid],
                "episode_id": eid,
                "model": materials[eid]["model"],
                "task_type": materials[eid]["task_type"],
                "task_index": materials[eid]["task_index"],
                "task": task_map[(materials[eid]["task_type"], materials[eid]["task_index"])],
            }
            for eid in episode_ids
        ],
        "tasks": [
            {"task": task_map[k], "task_type": k[0], "task_index": k[1]} for k in task_keys
        ],
    }
    (local_dir / "e2_episode_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )

    # Per-episode diagnostic flags
    diag_loose: dict[str, bool] = {}
    diag_strict: dict[str, bool] = {}
    for eid, m in materials.items():
        raw = m["evidence_rawlog"]
        req = m["requested_file"]
        loose = is_diagnostic(raw, req)
        strict = is_diagnostic_strict(raw, req)
        assert loose == bool(m["diagnostic"]), eid
        diag_loose[eid] = loose
        diag_strict[eid] = strict

    # Aggregate answers by episode x cell
    by_ec: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for a in answers:
        by_ec[(a["episode_id"], int(a["cell"]))].append(a)

    rows: list[dict[str, Any]] = []
    for eid in episode_ids:
        m = materials[eid]
        for cell in range(1, 9):
            evidence, attribution, note_type = CELL_META[cell]
            pool = by_ec[(eid, cell)]
            n_samples = 0
            n_false_primary = 0
            n_false_oss = 0
            for a in pool:
                aid = a["answer_id"]
                jp = judgments.get((aid, "judge_primary"))
                jo = judgments.get((aid, "judge_oss"))
                if jp and jp.get("label") not in (None, "PARSE_FAIL"):
                    n_samples += 1  # count primary-labeled samples as n_samples base
                    if jp["label"] == "FALSE_AFFIRM":
                        n_false_primary += 1
                # oss counted separately; n_samples uses primary when both present
                if jo and jo.get("label") not in (None, "PARSE_FAIL"):
                    if jo["label"] == "FALSE_AFFIRM":
                        n_false_oss += 1
            # Prefer max labeled count if oss has different n (parse fails)
            n_pri = sum(
                1
                for a in pool
                if (judgments.get((a["answer_id"], "judge_primary")) or {}).get("label")
                not in (None, "PARSE_FAIL", "")
                and judgments.get((a["answer_id"], "judge_primary"))
            )
            n_oss = sum(
                1
                for a in pool
                if (judgments.get((a["answer_id"], "judge_oss")) or {}).get("label")
                not in (None, "PARSE_FAIL", "")
                and judgments.get((a["answer_id"], "judge_oss"))
            )
            # Store n_samples as primary n (matches cell tables); oss k uses its own n in recompute via per-judge
            # For matrix, n_samples = number of samples attempted (usually 2); use len(pool)
            # and n_false_* only over non-PARSE_FAIL for that judge.
            n_false_primary = sum(
                1
                for a in pool
                if (judgments.get((a["answer_id"], "judge_primary")) or {}).get("label")
                == "FALSE_AFFIRM"
            )
            n_false_oss = sum(
                1
                for a in pool
                if (judgments.get((a["answer_id"], "judge_oss")) or {}).get("label")
                == "FALSE_AFFIRM"
            )
            rows.append(
                {
                    "episode": ep_map[eid],
                    "task": task_map[(m["task_type"], m["task_index"])],
                    "model": m["model"],
                    "task_type": m["task_type"],
                    "cell": cell,
                    "evidence": evidence,
                    "attribution": attribution,
                    "note_type": note_type,
                    "n_samples": len(pool),
                    "n_labeled_primary": n_pri,
                    "n_labeled_oss": n_oss,
                    "n_false_primary": n_false_primary,
                    "n_false_oss": n_false_oss,
                    "diagnostic_loose": diag_loose[eid],
                    "diagnostic_strict": diag_strict[eid],
                }
            )

    out_csv = root / "results" / "e2_outcome_matrix.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "task",
        "model",
        "task_type",
        "cell",
        "evidence",
        "attribution",
        "note_type",
        "n_samples",
        "n_labeled_primary",
        "n_labeled_oss",
        "n_false_primary",
        "n_false_oss",
        "diagnostic_loose",
        "diagnostic_strict",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Descriptive diagnostic stats
    n_ep = len(episode_ids)
    n_loose = sum(1 for eid in episode_ids if diag_loose[eid])
    n_strict = sum(1 for eid in episode_ids if diag_strict[eid])
    per_model: dict[str, Any] = {}
    for model in sorted({materials[e]["model"] for e in episode_ids}):
        eids = [e for e in episode_ids if materials[e]["model"] == model]
        nl = sum(1 for e in eids if diag_loose[e])
        ns = sum(1 for e in eids if diag_strict[e])
        per_model[model] = {
            "n": len(eids),
            "loose_n": nl,
            "loose_rate": nl / len(eids),
            "strict_n": ns,
            "strict_rate": ns / len(eids),
        }

    # RAWLOG false-claim rate (primary) by strict diagnostic
    rawlog_by_strict = {"true": {"k": 0, "n": 0}, "false": {"k": 0, "n": 0}}
    for a in answers:
        if int(a["cell"]) not in (5, 6):
            continue
        j = judgments.get((a["answer_id"], "judge_primary"))
        if not j or j.get("label") in (None, "PARSE_FAIL"):
            continue
        key = "true" if diag_strict[a["episode_id"]] else "false"
        rawlog_by_strict[key]["n"] += 1
        if j["label"] == "FALSE_AFFIRM":
            rawlog_by_strict[key]["k"] += 1
    for v in rawlog_by_strict.values():
        v["rate"] = (v["k"] / v["n"]) if v["n"] else None

    diagnostic = {
        "n_episodes": n_ep,
        "loose": {"n": n_loose, "rate": n_loose / n_ep},
        "strict": {"n": n_strict, "rate": n_strict / n_ep},
        "per_model": per_model,
        "rawlog_false_affirm_primary_by_strict": {
            "strict_yes": rawlog_by_strict["true"],
            "strict_no": rawlog_by_strict["false"],
        },
    }

    summary_path = root / "results" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["e2"]["diagnostic"] = diagnostic
    # Ensure prereg sha note if present elsewhere — not in summary currently
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Patch RESULTS.md: insert diagnostic section after Faithful controls or before E2 validation
    results_md = root / "RESULTS.md"
    text = results_md.read_text(encoding="utf-8")
    section_lines = [
        "### Log diagnosticity (descriptive)",
        "",
        "Loose = existing `is_diagnostic` (error cue and filename **or** the word `file`).",
        "Strict = error cue and the requested filename (full or basename); generic `file` does not count.",
        "",
        "| | n | rate |",
        "|---|---:|---:|",
        f"| loose | {n_loose}/{n_ep} | {n_loose / n_ep:.3f} |",
        f"| strict | {n_strict}/{n_ep} | {n_strict / n_ep:.3f} |",
        "",
        "| model | n | loose | strict |",
        "|---|---:|---:|---:|",
    ]
    for model, st in per_model.items():
        section_lines.append(
            f"| {model} | {st['n']} | {st['loose_n']}/{st['n']}={st['loose_rate']:.3f} | "
            f"{st['strict_n']}/{st['n']}={st['strict_rate']:.3f} |"
        )
    sy = rawlog_by_strict["true"]
    sn = rawlog_by_strict["false"]
    section_lines += [
        "",
        "RAWLOG FALSE_AFFIRM rate (judge_primary) by strict diagnostic:",
        "",
        "| strict | k/n | rate |",
        "|---|---:|---:|",
        f"| yes | {sy['k']}/{sy['n']} | {sy['rate']:.3f} |",
        f"| no | {sn['k']}/{sn['n']} | {sn['rate']:.3f} |",
        "",
    ]
    block = "\n".join(section_lines)
    marker = "## E2 validation"
    if "### Log diagnosticity" in text:
        # replace existing block up to next ##
        import re

        text = re.sub(
            r"### Log diagnosticity \(descriptive\).*?(?=\n## )",
            block + "\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
    elif marker in text:
        text = text.replace(marker, block + "\n" + marker, 1)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    results_md.write_text(text, encoding="utf-8")

    print(f"wrote {out_csv} rows={len(rows)}")
    print(f"wrote {local_dir / 'e2_episode_mapping.json'}")
    print("diagnostic", json.dumps(diagnostic, indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
