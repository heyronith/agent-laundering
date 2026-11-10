#!/usr/bin/env python3
"""T05 kill test: 160 B episodes + forks + 40 A-planted (resumable; namespace t05_killtest)."""

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
from memorytask.fork.events import event_for_a, event_for_b, extract_title  # noqa: E402
from memorytask.fork.harness import fork, load_b_fork_messages  # noqa: E402
from memorytask.fork.judges import judge_memory_note, judge_recognition  # noqa: E402
from memorytask.fork.log_render import render_log  # noqa: E402
from memorytask.framework_a.judges import judge_doubling, judge_report  # noqa: E402
from memorytask.framework_a.planted import load_planted_turns, run_planted_episode  # noqa: E402
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
    # ensure_ascii=True avoids U+2028/U+2029 JSONL splits; scrub removes NULs from tool traces.
    return json.dumps(_scrub(row), ensure_ascii=True, sort_keys=True, default=str)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_jsonl_dumps(row) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        out.append(json.loads(l))
    return out


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
        if r.get("run_id") != run_id:
            continue
        if r.get("task_id") != episode_id:
            continue
        if r.get("role") != "subject":
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
    ap.add_argument("--run-id", default="t05_killtest")
    ap.add_argument("--namespace", default="t05_killtest")
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--phase", choices=["all", "b", "label", "fork", "a"], default="all")
    args = ap.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    eps_path = run_dir / "episodes.jsonl"
    outcomes_path = run_dir / "outcomes.jsonl"
    branches_path = run_dir / "branches.jsonl"
    judgments_path = run_dir / "branch_judgments.jsonl"
    planted_path = run_dir / "planted_episodes.jsonl"
    token_stats_path = run_dir / "token_stats.jsonl"

    models = ["deepseek", "glm", "luna", "minimax"]
    types = [1, 2, 3, 5]
    n_per = 10

    # ---- Phase B ----
    if args.phase in ("all", "b"):
        plan = []
        for t in types:
            tasks = load_tasks(t)[:n_per]
            for task in tasks:
                for model in models:
                    plan.append((model, task))
        done = _done(eps_path, "episode_id")
        total = len(plan)
        for i, (model, task) in enumerate(plan, 1):
            eid = f"{model}__type{int(task['task_type'])}__idx{int(task['task_index'])}__s0"
            if eid in done:
                print(f"[{i}/{total}] SKIP {eid}")
                continue
            spent = run_cost(args.run_id)
            # reserve ~$4 for forks+A
            if spent > args.cap - 4.0:
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
                if "402" in msg or "credits" in msg.lower() or "in-flight" in msg.lower():
                    # One short settle for in-flight only; never shrink max_tokens (D010).
                    if "in-flight" in msg.lower():
                        print("  settle 20s for in-flight budget, one retry")
                        time.sleep(20)
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
                        except Exception as exc2:
                            print(f"STOP B: OpenRouter credits exhausted at {eid}: {exc2}")
                            break
                    else:
                        print(f"STOP B: OpenRouter credits exhausted at {eid} (spent={run_cost(args.run_id):.3f})")
                        break
                else:
                    ep = {
                        "episode_id": eid,
                        "model": model,
                        "model_alias": model,
                        "task_type": int(task["task_type"]),
                        "task_index": int(task["task_index"]),
                        "sample_index": 0,
                        "n_steps": 0,
                        "hit_max_steps": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "broken_tool_calls": 0,
                        "never_touched_tool": True,
                        "call_ids": [],
                        "final_answer": "",
                        "judge_outputs": {},
                        "upstream_result": None,
                        "instruction": task.get("instruction"),
                        "created_files": [],
                        "tool_trace": [],
                    }
            if isinstance(ep.get("error"), str) and (
                "402" in ep["error"] or "credits" in ep["error"].lower() or "in-flight" in ep["error"].lower()
            ):
                print(f"STOP B: credits error recorded on {eid} (not appending; resume later)")
                break
            _append(eps_path, ep)
            done.add(eid)
            pt = last_step_prompt_tokens(args.run_id, eid)
            _append(
                token_stats_path,
                {"episode_id": eid, "kind": "b_last_step_prompt_tokens", "prompt_tokens": pt},
            )
            print(
                f"  result={ep.get('upstream_result')} steps={ep.get('n_steps')} "
                f"broken={ep.get('broken_tool_calls')} err={ep.get('error')}"
            )

    # ---- Label ----
    if args.phase in ("all", "label", "fork"):
        rows = []
        for ep in _load_jsonl(eps_path):
            ep = enrich_final_answer_call_from_episode_dir(ep, run_dir)
            details = load_judge_details_for_episode(ep, run_dir)
            rows.append(label_episode(ep, details))
        with outcomes_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        from collections import Counter

        print("outcomes", Counter(r["outcome"] for r in rows))

    # ---- Forks ----
    if args.phase in ("all", "fork"):
        eps = {e["episode_id"]: e for e in _load_jsonl(eps_path)}
        outcomes = {o["episode_id"]: o for o in _load_jsonl(outcomes_path)}
        done_b = _done(branches_path, "branch_id")
        skipped_no = 0
        # Prefer lapse candidates first (most informative under a credit stop).
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
                if spent > args.cap - 1.5:
                    print(f"STOP forks: spend {spent:.3f}")
                    print(f"NO_ANSWER skipped so far: {skipped_no}")
                    return 0
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
                    if "402" in msg or "credits" in msg.lower() or "in-flight" in msg.lower():
                        print(f"STOP forks: credits exhausted at {bid}")
                        stop_forks = True
                        break
                    print(f"  fork error {bid}: {type(exc).__name__}: {msg[:240]}")
                    br = {
                        "branch_id": bid,
                        "episode_id": eid,
                        "branch": branch,
                        "fork_model": gen,
                        "error": f"{type(exc).__name__}: {exc}",
                        "output_text": "",
                    }
                br["event_description"] = event
                br["key_facts"] = key_facts
                br["outcome"] = oc
                _append(branches_path, br)
                done_b.add(bid)
                _append(
                    token_stats_path,
                    {
                        "branch_id": bid,
                        "kind": "branch",
                        "call_id": br.get("call_id"),
                    },
                )
            if stop_forks:
                break
        print(f"NO_ANSWER episodes skipped for forks: {skipped_no}")

    # ---- A planted ----
    if args.phase in ("all", "a"):
        planted_turns = load_planted_turns()
        done_p = _done(planted_path, "episode_id")
        done_b = _done(branches_path, "branch_id")
        judged_report_aliases: dict[str, set[str]] = {}
        for j in _load_jsonl(judgments_path):
            if j.get("kind") != "a_report_deception" or not j.get("branch_id"):
                continue
            judged_report_aliases.setdefault(j["branch_id"], set()).add(j.get("model_alias") or "")
        for planted in planted_turns:
            for model in models:
                peid = f"planted__{model}__{planted.get('index')}__s0"
                try:
                    if peid in done_p:
                        planted_ep = next(r for r in _load_jsonl(planted_path) if r["episode_id"] == peid)
                    else:
                        spent = run_cost(args.run_id)
                        if spent > args.cap - 0.5:
                            print(f"STOP A: spend {spent:.3f}")
                            return 0
                        print(f"PLANTED {peid}")
                        planted_ep = run_planted_episode(
                            model_alias=model,
                            planted=planted,
                            run_id=args.run_id,
                            cache_namespace=args.namespace,
                            budget_cap_usd=args.cap,
                        )
                        _append(planted_path, planted_ep)
                        done_p.add(peid)

                    report_bid = f"{peid}__report"
                    have = judged_report_aliases.get(report_bid, set())
                    for jalias in ("judge_primary", "judge_secondary"):
                        if jalias in have:
                            continue
                        jr = judge_report(
                            planted_ep["report_text"],
                            model_alias=jalias,
                            run_id=args.run_id,
                            episode_id=peid,
                            cache_namespace=args.namespace,
                            budget_cap_usd=args.cap,
                            sample_index=0,
                        )
                        jr["branch_id"] = report_bid
                        jr["kind"] = "a_report_deception"
                        _append(judgments_path, jr)
                        have.add(jalias)
                    judged_report_aliases[report_bid] = have

                    event, key_facts = event_for_a(planted_ep["amount"])
                    base = planted_ep["fork_messages"]
                    rendered = render_log(base)
                    for branch in ("K", "M1", "M3", "M4", "D"):
                        bid = f"{peid}__{model}__{branch}__s0"
                        if bid in done_b:
                            continue
                        spent = run_cost(args.run_id)
                        if spent > args.cap:
                            print(f"STOP A forks: spend {spent:.3f}")
                            return 0
                        print(f"FORK A {bid}")
                        br = fork(
                            planted_ep,
                            branch,
                            fork_model=model,
                            run_id=args.run_id,
                            namespace=args.namespace,
                            budget_cap_usd=args.cap,
                            generator_model="gpt-4-apollo-planted",
                            framework="A",
                            base_messages=base,
                            rendered_log=rendered,
                            doubling_down_trigger=planted_ep.get("doubling_down_trigger"),
                            recipient="Amy",
                        )
                        if branch == "D":
                            for jalias in ("judge_primary", "judge_secondary"):
                                jd = judge_doubling(
                                    br["output_text"],
                                    model_alias=jalias,
                                    run_id=args.run_id,
                                    episode_id=peid,
                                    cache_namespace=args.namespace,
                                    budget_cap_usd=args.cap,
                                    sample_index=0,
                                )
                                jd["branch_id"] = br["branch_id"]
                                jd["kind"] = "a_doubling_down"
                                _append(judgments_path, jd)
                        else:
                            judge_branch(
                                br, event, key_facts, args.run_id, args.namespace, args.cap, judgments_path
                            )
                        br["event_description"] = event
                        br["key_facts"] = key_facts
                        br["outcome"] = "LAPSE_PLANTED"
                        br["arm"] = "planted"
                        _append(branches_path, br)
                        done_b.add(bid)
                except Exception as exc:
                    msg = str(exc)
                    print(f"  A error {peid}: {type(exc).__name__}: {msg[:300]}")
                    if "402" in msg or "credits" in msg.lower() or "in-flight" in msg.lower():
                        print(f"STOP A: credits exhausted at {peid} (spent={run_cost(args.run_id):.3f})")
                        print(f"Done phase={args.phase}. spend={run_cost(args.run_id):.4f} ledger={total_spend_usd(root / 'logs'):.4f}")
                        return 0
                    raise

    print(f"Done phase={args.phase}. spend={run_cost(args.run_id):.4f} ledger={total_spend_usd(root / 'logs'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
