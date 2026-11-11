#!/usr/bin/env python3
"""T05b: re-run B arm in neutral /tmp/ws workspaces (namespace t05b_killtest)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from memorytask.budget import total_spend_usd  # noqa: E402
from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import repo_root  # noqa: E402
from memorytask.fork.events import event_for_b, extract_title  # noqa: E402
from memorytask.fork.harness import fork, load_b_fork_messages  # noqa: E402
from memorytask.fork.judges import judge_memory_note, judge_recognition  # noqa: E402
from memorytask.fork.log_render import render_log  # noqa: E402
from memorytask.framework_b.outcomes import (  # noqa: E402
    enrich_final_answer_call_from_episode_dir,
    label_episode,
    load_judge_details_for_episode,
)
from memorytask.framework_b.runner import load_tasks, run_episode  # noqa: E402


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("\x00", "").replace("\u2028", " ").replace("\u2029", " ")
    return obj


def _jsonl_dumps(row: dict) -> str:
    return json.dumps(_scrub(row), ensure_ascii=True, sort_keys=True, default=str)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_jsonl_dumps(row) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _done(path: Path, key: str) -> set[str]:
    return {r[key] for r in _load_jsonl(path) if key in r}


def run_cost(run_id: str) -> float:
    root = repo_root()
    return sum(
        float(r.get("cost_usd") or 0.0)
        for r in iter_run_logs(root / "logs")
        if r.get("run_id") == run_id and not r.get("cache_hit")
    )


def last_step_prompt_tokens(run_id: str, episode_id: str) -> int | None:
    root = repo_root()
    toks = None
    for r in iter_run_logs(root / "logs"):
        if r.get("run_id") != run_id or r.get("task_id") != episode_id or r.get("role") != "subject":
            continue
        usage = r.get("usage") or {}
        pt = usage.get("prompt_tokens")
        if pt is not None:
            toks = int(pt)
    return toks


def judge_branch(br: dict, event: str, key_facts: str, run_id: str, ns: str, cap: float, jpath: Path) -> None:
    fm = br["fork_model"]
    have = {
        j.get("model_alias")
        for j in _load_jsonl(jpath)
        if j.get("branch_id") == br["branch_id"] and j.get("label") not in (None, "PARSE_FAIL")
    }
    if br["branch"] == "K":
        for alias, si in (("judge_primary", 0), ("judge_secondary", 10)):
            if alias in have:
                continue
            j = judge_recognition(
                answer=br["output_text"],
                event_description=event,
                key_facts=key_facts,
                model_alias=alias,
                fork_model=fm,
                run_id=run_id,
                branch_id=br["branch_id"],
                cache_namespace=ns,
                budget_cap_usd=cap,
                sample_index=si,
            )
            _append(jpath, j)
    elif br["branch"] in ("M1", "M3", "M4"):
        for alias, si in (("judge_primary", 0), ("judge_secondary", 10)):
            if alias in have:
                continue
            j = judge_memory_note(
                note=br["output_text"],
                event_description=event,
                key_facts=key_facts,
                model_alias=alias,
                fork_model=fm,
                run_id=run_id,
                branch_id=br["branch_id"],
                cache_namespace=ns,
                budget_cap_usd=cap,
                sample_index=si,
            )
            _append(jpath, j)


def b_event(ep: dict, outcome: str) -> tuple[str, str]:
    kind = "LAPSE" if "LAPSE" in (outcome or "") else "HONEST"
    t = int(ep["task_type"])
    tasks = {int(x["task_index"]): x for x in load_tasks(t)}
    task = tasks.get(int(ep["task_index"]), {})
    if t == 1:
        return event_for_b(1, kind, file_name=task.get("file_name"))
    if t == 2:
        return event_for_b(2, kind, file_name=task.get("file_name"))
    if t == 3:
        return event_for_b(
            3, kind, pptx_file_name=task.get("pptx_file_name"), txt_file_name=task.get("txt_file_name")
        )
    if t == 5:
        return event_for_b(
            5,
            kind,
            title=extract_title(task.get("instruction") or ep.get("instruction") or ""),
            target_file_name=task.get("target_file_name"),
            instruction=task.get("instruction") or ep.get("instruction"),
        )
    return event_for_b(1, kind, file_name=task.get("file_name") or "file")


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t05b_killtest")
    ap.add_argument("--namespace", default="t05b_killtest")
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--phase", choices=["all", "b", "label", "fork", "smoke"], default="all")
    ap.add_argument("--smoke-type", type=int, default=1, help="task type for smoke (default 1)")
    args = ap.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    eps_path = run_dir / "episodes.jsonl"
    outcomes_path = run_dir / "outcomes.jsonl"
    branches_path = run_dir / "branches.jsonl"
    judgments_path = run_dir / "branch_judgments.jsonl"
    token_stats_path = run_dir / "token_stats.jsonl"

    models = ["deepseek", "glm", "luna", "minimax"]
    types = [1, 2, 3, 5]
    n_per = 10

    if args.phase == "smoke":
        # One episode per model, first task of smoke-type
        tasks = load_tasks(args.smoke_type)[:1]
        plan = [(m, tasks[0]) for m in models]
    elif args.phase in ("all", "b"):
        plan = []
        for t in types:
            for task in load_tasks(t)[:n_per]:
                for model in models:
                    plan.append((model, task))
    else:
        plan = []

    if args.phase in ("all", "b", "smoke"):
        done = _done(eps_path, "episode_id")
        total = len(plan)
        for i, (model, task) in enumerate(plan, 1):
            eid = f"{model}__type{int(task['task_type'])}__idx{int(task['task_index'])}__s0"
            if eid in done:
                print(f"[{i}/{total}] SKIP {eid}")
                continue
            spent = run_cost(args.run_id)
            # Soft reserve leaves headroom for forks/judges without aborting late B fills.
            if spent > args.cap - 0.35 and args.phase != "smoke":
                print(f"STOP B: spend {spent:.3f} exceeds soft reserve (cap {args.cap})")
                break
            print(f"[{i}/{total}] RUN {eid} (spent={spent:.3f})")
            try:
                ep = run_episode(
                    model_alias=model,
                    task=task,
                    episode_sample=0,
                    run_id=args.run_id,
                    cache_namespace=args.namespace,
                    budget_cap_usd=args.cap,
                    seed=int(task["task_index"]) * 1009,
                )
            except Exception as exc:
                msg = str(exc)
                print(f"  FATAL {eid}: {type(exc).__name__}: {msg[:300]}")
                if "402" in msg or "credits" in msg.lower():
                    print(f"STOP B: credits at {eid}")
                    break
                raise
            if isinstance(ep.get("error"), str) and (
                "402" in ep["error"] or "credits" in ep["error"].lower()
            ):
                print(f"STOP B: credits error on {eid}")
                break
            _append(eps_path, ep)
            done.add(eid)
            _append(
                token_stats_path,
                {
                    "episode_id": eid,
                    "kind": "b_last_step_prompt_tokens",
                    "prompt_tokens": last_step_prompt_tokens(args.run_id, eid),
                },
            )
            print(
                f"  result={ep.get('upstream_result')} steps={ep.get('n_steps')} "
                f"broken={ep.get('broken_tool_calls')} err={ep.get('error')}"
            )

    if args.phase in ("all", "label", "fork", "smoke"):
        rows = []
        for ep in _load_jsonl(eps_path):
            ep = enrich_final_answer_call_from_episode_dir(ep, run_dir)
            details = load_judge_details_for_episode(ep, run_dir)
            rows.append(label_episode(ep, details))
        with outcomes_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(_jsonl_dumps(r) + "\n")
        from collections import Counter

        print("outcomes", Counter(r["outcome"] for r in rows))

    if args.phase in ("all", "fork"):
        eps = {e["episode_id"]: e for e in _load_jsonl(eps_path)}
        outcomes = {o["episode_id"]: o for o in _load_jsonl(outcomes_path)}
        done_b = _done(branches_path, "branch_id")
        skipped_no = 0
        order = {"LAPSE_CANDIDATE": 0, "LAPSE_UNCONFIRMED": 1, "HONEST": 2}
        fork_eps = []
        for eid, ep in eps.items():
            oc = outcomes.get(eid, {}).get("outcome")
            if oc == "NO_ANSWER":
                skipped_no += 1
                continue
            if oc not in order:
                continue
            fork_eps.append((order[oc], eid, ep, oc))
        fork_eps.sort(key=lambda x: (x[0], x[1]))
        for _, eid, ep, oc in fork_eps:
            ep = dict(ep)
            ep["outcome"] = oc
            event, key_facts = b_event(ep, oc)
            try:
                base = load_b_fork_messages(args.run_id, eid)
            except Exception as exc:
                print(f"SKIP forks {eid}: {exc}")
                continue
            rendered = render_log(base)
            gen = ep.get("model") or ep.get("model_alias")
            stop_forks = False
            for branch in ("K", "M1", "M3", "M4"):
                bid = f"{eid}__{gen}__{branch}__s0"
                if bid in done_b:
                    continue
                spent = run_cost(args.run_id)
                if spent > args.cap - 0.15:
                    print(f"STOP forks: spend {spent:.3f}")
                    stop_forks = True
                    break
                print(f"FORK {bid}")
                try:
                    br = fork(
                        ep,
                        branch,
                        fork_model=gen,
                        run_id=args.run_id,
                        namespace=args.namespace,
                        budget_cap_usd=args.cap,
                        generator_model=gen,
                        framework="B",
                        base_messages=base,
                        rendered_log=rendered,
                    )
                    judge_branch(br, event, key_facts, args.run_id, args.namespace, args.cap, judgments_path)
                except Exception as exc:
                    msg = str(exc)
                    if "402" in msg or "credits" in msg.lower():
                        print(f"STOP forks: credits at {bid}")
                        stop_forks = True
                        break
                    raise
                br["event_description"] = event
                br["key_facts"] = key_facts
                br["outcome"] = oc
                _append(branches_path, br)
                done_b.add(bid)
            if stop_forks:
                break
        print(f"NO_ANSWER episodes skipped for forks: {skipped_no}")

    # T05b addendum §1: automatic host-write audit after every run.
    import subprocess

    audit = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_host_writes.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print((audit.stdout or audit.stderr or "").strip() or f"host_write_audit_exit={audit.returncode}")

    print(f"Done phase={args.phase}. spend={run_cost(args.run_id):.4f} ledger={total_spend_usd(root / 'logs'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
