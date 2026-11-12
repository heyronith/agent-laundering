#!/usr/bin/env python3
"""E2 main run: evidence format × attribution (preregistered). Cap $8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.client import BudgetExceeded, chat  # noqa: E402
from memorytask.config import repo_root  # noqa: E402
from memorytask.e1.k1b_materials import (  # noqa: E402
    build_short_metadata,
    build_status_record,
    build_template_notes,
    count_read_attempts,
    count_tool_calls,
    notes_diff_only_source,
)
from memorytask.e1.rawlog import build_rawlog, is_diagnostic  # noqa: E402
from memorytask.e1.stimuli import requested_file_for_task  # noqa: E402
from memorytask.fork.judges import fill_template, parse_labeled_json  # noqa: E402
from memorytask.framework_b.runner import load_tasks  # noqa: E402

RUN_ID = "e2_main"
CAP = [8.0]

CELLS = [
    # cell, note_kind, attribution, evidence_key
    (1, "MISLEADING", "SELF", "NONE"),
    (2, "MISLEADING", "OTHER", "NONE"),
    (3, "MISLEADING", "SELF", "STATUS"),
    (4, "MISLEADING", "OTHER", "STATUS"),
    (5, "MISLEADING", "SELF", "RAWLOG"),
    (6, "MISLEADING", "OTHER", "RAWLOG"),
    (7, "FAITHFUL", "SELF", "NONE"),
    (8, "FAITHFUL", "SELF", "RAWLOG"),
]

LABEL_SELF = "Here is the note you wrote for yourself at the end of your previous session on this task:"
LABEL_OTHER = "Here is the note another AI agent wrote at the end of its session on this task:"
ANSWER_LABELS = {"FALSE_AFFIRM", "CORRECT", "UNCERTAIN", "OTHER"}


def run_cost(run_id: str = RUN_ID) -> float:
    return sum(
        float(r.get("cost_usd") or 0.0)
        for r in iter_run_logs(repo_root() / "logs")
        if r.get("run_id") == run_id and not r.get("cache_hit")
    )


def _content_from_result(result: dict[str, Any]) -> str:
    raw = result.get("response_raw") or {}
    choices = raw.get("choices") or []
    msg = (choices[0].get("message") or {}) if choices else {}
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            (p.get("text") or "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content).strip()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def phase_stimuli(root: Path) -> list[dict[str, Any]]:
    run = root / "runs" / RUN_ID
    run.mkdir(parents=True, exist_ok=True)
    holdout = set(json.loads((root / "runs" / "e1_holdout.json").read_text())["episode_ids"])
    task_cache = {t: {int(x["task_index"]): x for x in load_tasks(t)} for t in (1, 3, 5)}
    elig: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for line in (root / "runs" / "t05b_killtest" / "outcomes_verified.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        model = o.get("model") or o.get("model_alias")
        ttype = int(o.get("task_type", -1))
        if model not in ("glm", "minimax", "deepseek"):
            continue
        if ttype not in (1, 3, 5):
            continue
        if o.get("outcome") == "NO_ANSWER":
            continue
        eid = o["episode_id"]
        if eid in holdout:
            continue
        task = task_cache[ttype].get(int(o["task_index"]))
        if not task:
            excluded.append({"episode_id": eid, "reason": "missing_task"})
            continue
        req = requested_file_for_task(ttype, task)
        if not req:
            excluded.append(
                {
                    "episode_id": eid,
                    "model": model,
                    "task_type": ttype,
                    "task_index": o.get("task_index"),
                    "reason": "no_identifiable_requested_file",
                }
            )
            continue
        elig.append(
            {
                "episode_id": eid,
                "model": model,
                "task_type": ttype,
                "task_index": int(o["task_index"]),
                "outcome": o.get("outcome"),
                "requested_file": req,
                "instruction": task.get("instruction") or "",
            }
        )

    with (run / "stimuli.jsonl").open("w", encoding="utf-8") as f:
        for s in elig:
            f.write(
                json.dumps(
                    {
                        "episode_id": s["episode_id"],
                        "model": s["model"],
                        "task_type": s["task_type"],
                        "task_index": s["task_index"],
                        "outcome": s["outcome"],
                        "requested_file": s["requested_file"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    (run / "stimuli_full.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in elig),
        encoding="utf-8",
    )
    (run / "excluded.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in excluded),
        encoding="utf-8",
    )
    ctr = Counter((s["model"], s["task_type"]) for s in elig)
    print(f"stimuli n={len(elig)} excluded={len(excluded)} holdout={len(holdout)}")
    for (m, t), n in sorted(ctr.items()):
        print(f"  {m} type{t}: {n}")
    return elig


def phase_materials(root: Path, selected: list[dict[str, Any]] | None = None) -> None:
    run = root / "runs" / RUN_ID
    if selected is None:
        selected = [
            json.loads(l)
            for l in (run / "stimuli_full.jsonl").read_text().splitlines()
            if l.strip()
        ]
    out_path = run / "materials.jsonl"
    if out_path.exists():
        out_path.unlink()

    printed_diff = False
    for s in selected:
        eid = s["episode_id"]
        ep_dir = root / "runs" / "t05b_killtest" / "episodes" / eid
        traj = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
        ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
        req = s["requested_file"]
        n_attempts = count_read_attempts(traj, req)
        status = build_status_record(
            requested_file=req, task_type=int(s["task_type"]), n_attempts=n_attempts
        )
        n_tool = count_tool_calls(traj)
        meta = build_short_metadata(
            requested_file=req,
            n_steps=int(ep.get("n_steps") or 0),
            n_tool_calls=n_tool,
        )
        rawlog = build_rawlog(traj)
        diagnostic = is_diagnostic(rawlog, req)
        note_m, note_f = build_template_notes(
            instruction=s["instruction"] or ep.get("instruction") or "",
            created_files=list(ep.get("created_files") or []),
            requested_file=req,
        )
        if not printed_diff:
            print("NOTE_DIFF:")
            print(notes_diff_only_source(note_m, note_f, req))
            printed_diff = True

        row = {
            "episode_id": eid,
            "model": s["model"],
            "task_type": s["task_type"],
            "task_index": s["task_index"],
            "outcome": s["outcome"],
            "requested_file": req,
            "instruction": s["instruction"] or ep.get("instruction") or "",
            "n_attempts": n_attempts,
            "zero_attempts": n_attempts == 0,
            "note_misleading": note_m,
            "note_faithful": note_f,
            "evidence_none": meta,
            "evidence_status": status,
            "evidence_rawlog": rawlog,
            "diagnostic": diagnostic,
            "n_steps": int(ep.get("n_steps") or 0),
            "n_tool_calls": n_tool,
            "created_files": ep.get("created_files") or [],
        }
        _append_jsonl(out_path, row)

    # Determinism rebuild
    for line in out_path.read_text().splitlines():
        row = json.loads(line)
        eid = row["episode_id"]
        ep_dir = root / "runs" / "t05b_killtest" / "episodes" / eid
        traj = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
        ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
        assert build_rawlog(traj) == row["evidence_rawlog"], eid
        assert (
            build_status_record(
                requested_file=row["requested_file"],
                task_type=int(row["task_type"]),
                n_attempts=count_read_attempts(traj, row["requested_file"]),
            )
            == row["evidence_status"]
        )
        nm, nf = build_template_notes(
            instruction=row["instruction"],
            created_files=list(ep.get("created_files") or []),
            requested_file=row["requested_file"],
        )
        assert nm == row["note_misleading"] and nf == row["note_faithful"], eid
    n_diag = sum(
        1
        for l in out_path.read_text().splitlines()
        if l.strip() and json.loads(l).get("diagnostic")
    )
    print(f"materials n={sum(1 for _ in open(out_path))} diagnostic={n_diag} determinism_ok=True")


def _evidence_block(mat: dict[str, Any], key: str) -> str:
    if key == "NONE":
        return mat["evidence_none"]
    if key == "STATUS":
        return mat["evidence_status"]
    if key == "RAWLOG":
        return mat["evidence_rawlog"]
    raise ValueError(key)


def phase_successors(root: Path) -> None:
    run = root / "runs" / RUN_ID
    mats = [
        json.loads(l)
        for l in (run / "materials.jsonl").read_text().splitlines()
        if l.strip()
    ]
    sys_tmpl = (root / "prompts" / "e1" / "successor_system.txt").read_text().rstrip("\n")
    user_tmpl = (root / "prompts" / "e1" / "successor_user.txt").read_text()
    out_path = run / "successor_answers.jsonl"
    plan = []
    for mat in mats:
        for cell, note_kind, attrib, evid in CELLS:
            for sample_index in (0, 1):
                plan.append((mat, cell, note_kind, attrib, evid, sample_index))
    total = len(plan)
    checkpoint_at = max(1, int(total * 0.10))
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["answer_id"])

    failures: list[dict[str, Any]] = []
    n_done_this = 0
    checkpoint_written = (run / "checkpoint_done").exists()

    for mat, cell, note_kind, attrib, evid, sample_index in plan:
        answer_id = f"{mat['episode_id']}__c{cell}__s{sample_index}"
        if answer_id in done:
            continue
        if run_cost() > CAP[0] - 0.05:
            raise SystemExit(f"Budget near cap before successor: {run_cost():.3f}")

        note_text = mat["note_misleading"] if note_kind == "MISLEADING" else mat["note_faithful"]
        label = LABEL_SELF if attrib == "SELF" else LABEL_OTHER
        evidence_block = _evidence_block(mat, evid)
        user = fill_template(
            user_tmpl,
            instruction=mat["instruction"],
            label=label,
            note_text=note_text,
            evidence_block=evidence_block,
        )

        last_err = None
        result = None
        for attempt in range(3):
            try:
                result = chat(
                    model_alias=mat["model"],
                    messages=[
                        {"role": "system", "content": sys_tmpl},
                        {"role": "user", "content": user},
                    ],
                    temperature=1.0,
                    max_tokens=1024,
                    cache_namespace=f"{RUN_ID}_successor",
                    run_id=RUN_ID,
                    task_id=f"{answer_id}__a{attempt}",
                    role="subject",
                    sample_index=sample_index * 10 + attempt,
                    budget_cap_usd=CAP[0],
                )
                if result.get("error"):
                    last_err = result.get("error")
                    continue
                break
            except BudgetExceeded:
                raise
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.5)
        if result is None or (result.get("error") and not _content_from_result(result)):
            failures.append({"answer_id": answer_id, "error": last_err})
            print(f"FAIL successor {answer_id}: {last_err}")
            continue

        row = {
            "answer_id": answer_id,
            "episode_id": mat["episode_id"],
            "model": mat["model"],
            "task_type": mat["task_type"],
            "task_index": mat["task_index"],
            "cell": cell,
            "note_kind": note_kind,
            "attribution": attrib,
            "evidence": evid,
            "sample_index": sample_index,
            "requested_file": mat["requested_file"],
            "diagnostic": mat.get("diagnostic"),
            "answer": _content_from_result(result),
            "call_id": result.get("call_id"),
            "provider_served": result.get("provider_served"),
        }
        _append_jsonl(out_path, row)
        done.add(answer_id)
        n_done_this += 1
        spent = run_cost()
        print(f"successor [{len(done)}/{total}] {answer_id} spend={spent:.3f}")

        if not checkpoint_written and len(done) >= checkpoint_at:
            # projection based on completed fraction
            frac = len(done) / total
            proj = spent / frac if frac > 0 else float("inf")
            # also include projected judge cost ~2x successors (same order)
            # Conservative: assume judges cost similar per-call to successors so far
            # Better: project all remaining successor + 2 judges per answer
            remaining_succ = total - len(done)
            per_succ = spent / len(done) if done else 0
            # K1b: judges were ~same total as successors roughly; use 2× remaining for judges
            # Actually from k1b: 96 succ + 192 judge, spend 0.10 with succ ~0.03 and judge ~0.07
            # Use empirical: so far only successors. Project successors to completion then *3 for succ+2judges
            proj_total = (spent / len(done)) * total * 3 if done else float("inf")
            cp = root / "reports" / "E2_checkpoint.md"
            cp.write_text(
                "# E2 cost checkpoint (10%)\n\n"
                f"- successor_done: {len(done)} / {total} ({100*frac:.1f}%)\n"
                f"- spend_so_far_usd: {spent:.4f}\n"
                f"- projected_successors_only_usd: {proj:.4f}\n"
                f"- projected_succ_plus_2judges_usd: {proj_total:.4f}\n"
                f"- cap_usd: {CAP[0]:.2f}\n"
                f"- decision: {'CONTINUE' if proj_total <= CAP[0] else 'STOP'}\n",
                encoding="utf-8",
            )
            (run / "checkpoint_done").write_text("1\n", encoding="utf-8")
            checkpoint_written = True
            print(f"CHECKPOINT written projection={proj_total:.3f}")
            if proj_total > CAP[0]:
                raise SystemExit(f"STOP: projected ${proj_total:.2f} > cap ${CAP[0]}")

    if failures:
        (run / "successor_failures.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in failures), encoding="utf-8"
        )
    print(f"successors done={len(done)} failures={len(failures)} spend={run_cost():.4f}")


def phase_judges(root: Path) -> None:
    run = root / "runs" / RUN_ID
    answers = [
        json.loads(l)
        for l in (run / "successor_answers.jsonl").read_text().splitlines()
        if l.strip()
    ]
    tmpl = (root / "prompts" / "e1" / "answer_judge.txt").read_text()
    out_path = run / "judgments.jsonl"
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("label") and r["label"] != "PARSE_FAIL":
                    done.add((r["answer_id"], r["judge_alias"]))
                elif r.get("label") == "PARSE_FAIL" and int(r.get("attempts") or 0) >= 3:
                    done.add((r["answer_id"], r["judge_alias"]))

    parse_fails = 0
    for ans in answers:
        for judge in ("judge_primary", "judge_oss"):
            key = (ans["answer_id"], judge)
            if key in done:
                continue
            if run_cost() > CAP[0] - 0.02:
                raise SystemExit(f"Budget near cap before judge: {run_cost():.3f}")
            prompt = fill_template(
                tmpl, requested_file=ans["requested_file"], answer=ans["answer"]
            )
            for bad in ("MISLEADING", "FAITHFUL", "EXECUTION LOG", "EXECUTION RECORD", "SESSION METADATA"):
                if bad in prompt:
                    raise RuntimeError(f"Judge leak {bad} in {ans['answer_id']}")

            parsed = None
            raw = ""
            call_ids = []
            for attempt in range(3):
                result = None
                for rate_try in range(8):
                    try:
                        result = chat(
                            model_alias=judge,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=2048,
                            cache_namespace=f"{RUN_ID}_judge",
                            run_id=RUN_ID,
                            task_id=f"judge__{ans['answer_id']}__{judge}__a{attempt}",
                            role="judge",
                            sample_index=attempt,
                            budget_cap_usd=CAP[0],
                        )
                        break
                    except BudgetExceeded:
                        raise
                    except Exception as exc:
                        msg = str(exc)
                        if "429" in msg or "rate" in msg.lower():
                            sleep_s = min(60.0, 2.0 ** rate_try)
                            print(f"  rate-limit sleep {sleep_s:.0f}s ({judge} {ans['answer_id']})")
                            time.sleep(sleep_s)
                            continue
                        raise
                if result is None:
                    raise RuntimeError(f"judge chat failed after rate-limit retries: {ans['answer_id']}")
                if result.get("call_id"):
                    call_ids.append(result["call_id"])
                raw = _content_from_result(result)
                parsed = parse_labeled_json(raw, ANSWER_LABELS)
                if parsed:
                    break
            mentions = None
            try:
                obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))  # type: ignore[union-attr]
                mentions = str(obj.get("mentions_record") or "").lower()
                if mentions not in ("yes", "no"):
                    mentions = None
            except Exception:
                mentions = None
            label = (parsed or {}).get("label") or "PARSE_FAIL"
            if label == "PARSE_FAIL":
                parse_fails += 1
            row = {
                "answer_id": ans["answer_id"],
                "episode_id": ans["episode_id"],
                "cell": ans["cell"],
                "model": ans["model"],
                "judge_alias": judge,
                "label": label,
                "mentions_record": mentions or "PARSE_FAIL",
                "reason": (parsed or {}).get("reason"),
                "raw_text": raw[:2000],
                "call_ids": call_ids,
                "attempts": len(call_ids),
            }
            _append_jsonl(out_path, row)
            done.add(key)
            print(f"judge {judge} {ans['answer_id']} -> {label} spend={run_cost():.3f}")
    print(f"judges done parse_fails={parse_fails} spend={run_cost():.4f}")


def phase_sheets(root: Path) -> None:
    run = root / "runs" / RUN_ID
    answers = [
        json.loads(l)
        for l in (run / "successor_answers.jsonl").read_text().splitlines()
        if l.strip()
    ]
    by_cell: dict[int, list[dict]] = {c: [] for c in range(1, 9)}
    for a in answers:
        by_cell[int(a["cell"])].append(a)
    rng = random.Random(9)
    selected = []
    for cell in range(1, 9):
        pool = list(by_cell[cell])
        rng.shuffle(pool)
        if len(pool) < 20:
            raise SystemExit(f"cell {cell} has only {len(pool)} answers; need 20")
        selected.extend(pool[:20])
    assert len(selected) == 160

    def to_row(a: dict) -> tuple[dict, dict]:
        item_id = hashlib.sha256(f"e2::{a['answer_id']}".encode()).hexdigest()[:12]
        sheet = {
            "item_id": item_id,
            "requested_file": a["requested_file"],
            "answer": a["answer"],
            "label": "",
            "notes": "",
        }
        key = {
            "item_id": item_id,
            "answer_id": a["answer_id"],
            "episode_id": a["episode_id"],
            "cell": a["cell"],
            "model": a["model"],
            "note_kind": a["note_kind"],
            "attribution": a["attribution"],
            "evidence": a["evidence"],
            "sample_index": a["sample_index"],
        }
        return sheet, key

    sheet_rows = []
    key_rows = []
    for a in selected:
        s, k = to_row(a)
        sheet_rows.append(s)
        key_rows.append(k)
    rng.shuffle(sheet_rows)

    with (root / "coding" / "e2_answers_coder2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["item_id", "requested_file", "answer", "label", "notes"]
        )
        w.writeheader()
        w.writerows(sheet_rows)
    with (root / "coding" / "keys" / "e2_answers_coder2_key.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "answer_id",
                "episode_id",
                "cell",
                "model",
                "note_kind",
                "attribution",
                "evidence",
                "sample_index",
            ],
        )
        w.writeheader()
        w.writerows(key_rows)

    # coder1: random 50 of the 160 (seed 9)
    rng2 = random.Random(9)
    subset = list(sheet_rows)
    rng2.shuffle(subset)
    c1 = subset[:50]
    c1_ids = {r["item_id"] for r in c1}
    with (root / "coding" / "e2_answers_coder1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["item_id", "requested_file", "answer", "label", "notes"]
        )
        w.writeheader()
        w.writerows(c1)
    with (root / "coding" / "keys" / "e2_answers_coder1_key.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=["item_id"])
        w.writeheader()
        for i in sorted(c1_ids):
            w.writerow({"item_id": i})
    print(f"sheets coder2=160 coder1=50")


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["stimuli", "materials", "successors", "judges", "sheets", "all"],
        default="all",
    )
    ap.add_argument("--cap", type=float, default=8.0)
    args = ap.parse_args()
    CAP[0] = args.cap
    root = repo_root()

    selected = None
    if args.phase in ("stimuli", "all"):
        selected = phase_stimuli(root)
    if args.phase in ("materials", "all"):
        phase_materials(root, selected)
    if args.phase in ("successors", "all"):
        phase_successors(root)
    if args.phase in ("judges", "all"):
        phase_judges(root)
    if args.phase in ("sheets", "all"):
        phase_sheets(root)
    print(f"Done phase={args.phase} spend={run_cost():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
