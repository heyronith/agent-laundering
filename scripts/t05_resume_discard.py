#!/usr/bin/env python3
"""T05-RESUME §3: mark degraded (reduced max_tokens) calls; prune artifacts for re-run."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402

# Limits temporarily used during credit exhaustion (must not be kept).
REDUCED_MAX_TOKENS = {256, 512, 768, 1024}
SUBJECT_OK = 2048
JUDGE_OK = 2048


def _mt(r: dict) -> int | None:
    return ((r.get("request") or {}).get("params") or {}).get("max_tokens")


def main() -> int:
    root = repo_root()
    run = root / "runs" / "t05_killtest"
    run.mkdir(parents=True, exist_ok=True)

    discarded: list[dict] = []
    subject_by_ep: dict[str, list[dict]] = defaultdict(list)
    judge_by_ep: dict[str, list[dict]] = defaultdict(list)

    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != "t05_killtest":
            continue
        mt = _mt(r)
        tid = r.get("task_id") or ""
        role = r.get("role")
        is_branch = any(x in tid for x in ("__K__", "__M1__", "__M3__", "__M4__", "__D__"))
        is_planted = tid.startswith("planted__")
        is_b_subject = role == "subject" and not is_branch and not is_planted
        is_b_judge = role == "judge" and not is_planted and "__report" not in tid and not is_branch

        if mt in REDUCED_MAX_TOKENS:
            discarded.append(
                {
                    "call_id": r.get("call_id"),
                    "task_id": tid,
                    "role": role,
                    "model_alias": r.get("model_alias"),
                    "max_tokens": mt,
                    "http_status": r.get("http_status"),
                    "error": r.get("error"),
                    "cost_usd": r.get("cost_usd"),
                    "reason": f"reduced_max_tokens={mt}",
                }
            )

        if is_b_subject:
            # task_id == episode_id for agent steps
            subject_by_ep[tid].append(r)
        if is_b_judge:
            # judge task_ids are episode_id__judge__...
            eid = tid.split("__judge")[0] if "__judge" in tid else tid
            judge_by_ep[eid].append(r)

    out_disc = run / "discarded_calls.jsonl"
    with out_disc.open("w", encoding="utf-8") as f:
        for row in discarded:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    print(f"discarded_calls={len(discarded)} -> {out_disc}")

    eps_path = run / "episodes.jsonl"
    eps = [json.loads(l) for l in eps_path.read_text(encoding="utf-8").splitlines() if l.strip()] if eps_path.exists() else []
    keep: list[dict] = []
    drop_ids: list[str] = []
    rejudge_ids: list[str] = []
    for ep in eps:
        eid = ep["episode_id"]
        steps = subject_by_ep.get(eid, [])
        ok_subject = bool(steps) and all(
            _mt(s) == SUBJECT_OK
            and not s.get("error")
            and (s.get("http_status") is None or int(s["http_status"]) < 400)
            for s in steps
        )
        if ep.get("error") and ("402" in str(ep["error"]) or "credits" in str(ep["error"]).lower()):
            ok_subject = False
        if not ok_subject:
            drop_ids.append(eid)
            continue
        # Judges: keep episode row only if all successful judges used 2048
        jcalls = judge_by_ep.get(eid, [])
        reduced_j = [j for j in jcalls if _mt(j) in REDUCED_MAX_TOKENS]
        if reduced_j or not jcalls:
            # Drop from episodes.jsonl so B phase re-runs (subject cache hits @2048; judges re-call @2048)
            rejudge_ids.append(eid)
            drop_ids.append(eid)
            continue
        bad_j = [
            j
            for j in jcalls
            if j.get("error") or (j.get("http_status") is not None and int(j["http_status"]) >= 400)
        ]
        if bad_j:
            rejudge_ids.append(eid)
            drop_ids.append(eid)
            continue
        keep.append(ep)

    with eps_path.open("w", encoding="utf-8") as f:
        for ep in keep:
            f.write(json.dumps(ep, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    print(f"episodes kept={len(keep)} dropped_for_rerun={len(drop_ids)} (rejudge={len(rejudge_ids)})")

    # Clear derived artifacts that depended on degraded judges/branches/A
    for name in (
        "outcomes.jsonl",
        "branches.jsonl",
        "branch_judgments.jsonl",
        "planted_episodes.jsonl",
        "outcomes_verified.jsonl",
    ):
        p = run / name
        if p.exists():
            # keep a backup copy under discarded/
            bak = run / "discarded_artifacts" / name
            bak.parent.mkdir(parents=True, exist_ok=True)
            bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            p.write_text("", encoding="utf-8")
            print(f"cleared {name} (backup {bak})")

    # token_stats: keep B subject stats for kept episodes only
    ts = run / "token_stats.jsonl"
    if ts.exists():
        rows = [json.loads(l) for l in ts.read_text(encoding="utf-8").splitlines() if l.strip()]
        kept_ids = {e["episode_id"] for e in keep}
        kept_ts = [
            r
            for r in rows
            if r.get("episode_id") in kept_ids and r.get("kind") == "b_last_step_prompt_tokens"
        ]
        ts.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in kept_ts),
            encoding="utf-8",
        )
        print(f"token_stats kept={len(kept_ts)}")

    summary = {
        "discarded_calls": len(discarded),
        "episodes_kept": len(keep),
        "episodes_rerun": drop_ids,
        "rejudge_episodes": rejudge_ids,
        "reduced_max_tokens": sorted(REDUCED_MAX_TOKENS),
    }
    (run / "resume_discard_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
