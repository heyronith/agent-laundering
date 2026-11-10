#!/usr/bin/env python3
"""T04 live demo: B forks + transplant + A-planted. namespace/run_id t04_demo."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from memorytask.config import repo_root  # noqa: E402
from memorytask.fork.events import event_for_a, event_for_b, extract_title  # noqa: E402
from memorytask.fork.harness import fork, load_b_fork_messages  # noqa: E402
from memorytask.fork.judges import judge_memory_note, judge_recognition  # noqa: E402
from memorytask.fork.log_render import render_log  # noqa: E402
from memorytask.framework_a.judges import judge_doubling, judge_report  # noqa: E402
from memorytask.framework_a.planted import load_planted_turns, run_planted_episode  # noqa: E402
from memorytask.framework_b.outcomes import label_episode, enrich_final_answer_call_from_episode_dir  # noqa: E402


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _done_ids(path: Path, key: str) -> set[str]:
    return {r[key] for r in _load_jsonl(path) if key in r}


def b_event_for_ep(ep: dict, outcome: str) -> tuple[str, str]:
    kind = "LAPSE" if outcome.startswith("LAPSE") else "HONEST"
    t = int(ep["task_type"])
    meta = {}
    # Load task metadata from vendor dataset
    from memorytask.framework_b.runner import load_tasks

    tasks = {int(x["task_index"]): x for x in load_tasks(t)}
    task = tasks.get(int(ep["task_index"]), {})
    if t == 1:
        return event_for_b(1, kind, file_name=task.get("file_name"))
    if t == 5:
        return event_for_b(
            5,
            kind,
            title=extract_title(task.get("instruction") or ep.get("instruction") or ""),
            target_file_name=task.get("target_file_name"),
            instruction=task.get("instruction") or ep.get("instruction"),
        )
    return event_for_b(t, kind, file_name=task.get("file_name"))


def judge_branch(
    *,
    br: dict,
    event: str,
    key_facts: str,
    run_id: str,
    namespace: str,
    cap: float,
    secondary: bool,
    judgments_path: Path,
) -> None:
    fork_model = br["fork_model"]
    if br["branch"] == "K":
        primary = judge_recognition(
            answer=br["output_text"],
            event_description=event,
            key_facts=key_facts,
            model_alias="judge_primary",
            fork_model=fork_model,
            run_id=run_id,
            branch_id=br["branch_id"],
            cache_namespace=namespace,
            budget_cap_usd=cap,
        )
        _append(judgments_path, primary)
        if secondary:
            sec = judge_recognition(
                answer=br["output_text"],
                event_description=event,
                key_facts=key_facts,
                model_alias="judge_secondary",
                fork_model=fork_model,
                run_id=run_id,
                branch_id=br["branch_id"],
                cache_namespace=namespace,
                budget_cap_usd=cap,
                sample_index=10,
            )
            _append(judgments_path, sec)
    elif br["branch"] in ("M1", "M2", "M3", "M4"):
        primary = judge_memory_note(
            note=br["output_text"],
            event_description=event,
            key_facts=key_facts,
            model_alias="judge_primary",
            fork_model=fork_model,
            run_id=run_id,
            branch_id=br["branch_id"],
            cache_namespace=namespace,
            budget_cap_usd=cap,
        )
        _append(judgments_path, primary)
        if secondary:
            sec = judge_memory_note(
                note=br["output_text"],
                event_description=event,
                key_facts=key_facts,
                model_alias="judge_secondary",
                fork_model=fork_model,
                run_id=run_id,
                branch_id=br["branch_id"],
                cache_namespace=namespace,
                budget_cap_usd=cap,
                sample_index=10,
            )
            _append(judgments_path, sec)


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t04_demo")
    ap.add_argument("--namespace", default="t04_demo")
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--secondary-fraction", type=float, default=1.0)
    args = ap.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    branches_path = run_dir / "branches.jsonl"
    judgments_path = run_dir / "branch_judgments.jsonl"
    done_branches = _done_ids(branches_path, "branch_id")

    # Load T03 episodes + outcomes
    t03 = root / "runs" / "t03_repro"
    eps = {
        e["episode_id"]: enrich_final_answer_call_from_episode_dir(e, t03)
        for e in _load_jsonl(t03 / "episodes.jsonl")
    }
    outcomes = {o["episode_id"]: o for o in _load_jsonl(t03 / "outcomes.jsonl")}
    for eid, ep in eps.items():
        ep["outcome"] = outcomes.get(eid, label_episode(ep))["outcome"]

    verified_glm = [
        eid
        for eid, o in outcomes.items()
        if o["outcome"] == "LAPSE_VERIFIED" and o["model"] == "glm"
    ]
    # Expected 3; take what we have
    honest_targets = [
        "deepseek__type1__idx0__s0",
        "luna__type5__idx61__s0",
    ]
    # any glm type1 HONEST
    glm_honest = [
        eid
        for eid, o in outcomes.items()
        if o["model"] == "glm" and o["task_type"] == 1 and o["outcome"] == "HONEST"
    ]
    if glm_honest:
        honest_targets.append(glm_honest[0])

    b_episodes = []
    for eid in verified_glm[:3] + honest_targets[:3]:
        if eid in eps:
            b_episodes.append(eps[eid])

    print(f"B episodes to fork: {[e['episode_id'] for e in b_episodes]}")

    # --- B forks ---
    for ep in b_episodes:
        eid = ep["episode_id"]
        outcome = ep["outcome"]
        event, key_facts = b_event_for_ep(ep, outcome)
        base = load_b_fork_messages("t03_repro", eid)
        rendered = render_log(base)
        gen = ep.get("model") or ep.get("model_alias")
        for branch in ("K", "M1", "M2", "M3", "M4"):
            bid = f"{eid}__{gen}__{branch}__s0"
            if bid in done_branches:
                print(f"SKIP {bid}")
                continue
            print(f"FORK {bid}")
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
            br["event_description"] = event
            br["key_facts"] = key_facts
            br["outcome"] = outcome
            _append(branches_path, br)
            done_branches.add(br["branch_id"])
            judge_branch(
                br=br,
                event=event,
                key_facts=key_facts,
                run_id=args.run_id,
                namespace=args.namespace,
                cap=args.cap,
                secondary=True,
                judgments_path=judgments_path,
            )

    # --- Transplant: one GLM LAPSE_VERIFIED × deepseek/luna/minimax × M1,M3 ---
    if verified_glm:
        ep = eps[verified_glm[0]]
        event, key_facts = b_event_for_ep(ep, ep["outcome"])
        base = load_b_fork_messages("t03_repro", ep["episode_id"])
        rendered = render_log(base)
        for fm in ("deepseek", "luna", "minimax"):
            for branch in ("M1", "M3"):
                bid = f"{ep['episode_id']}__{fm}__{branch}__s0"
                if bid in done_branches:
                    print(f"SKIP transplant {bid}")
                    continue
                print(f"TRANSPLANT {bid}")
                br = fork(
                    ep,
                    branch,
                    fork_model=fm,
                    run_id=args.run_id,
                    namespace=args.namespace,
                    budget_cap_usd=args.cap,
                    generator_model=ep.get("model"),
                    framework="B",
                    base_messages=base,
                    rendered_log=rendered,
                )
                br["event_description"] = event
                br["key_facts"] = key_facts
                br["outcome"] = ep["outcome"]
                br["arm"] = "transplant"
                _append(branches_path, br)
                done_branches.add(br["branch_id"])
                judge_branch(
                    br=br,
                    event=event,
                    key_facts=key_facts,
                    run_id=args.run_id,
                    namespace=args.namespace,
                    cap=args.cap,
                    secondary=True,
                    judgments_path=judgments_path,
                )

    # --- A planted: 1 turn × 4 models ---
    planted = load_planted_turns()[0]
    planted_eps_path = run_dir / "planted_episodes.jsonl"
    done_planted = _done_ids(planted_eps_path, "episode_id")
    for model in ("deepseek", "glm", "luna", "minimax"):
        peid = f"planted__{model}__{planted.get('index')}__s0"
        if peid in done_planted:
            planted_ep = next(r for r in _load_jsonl(planted_eps_path) if r["episode_id"] == peid)
        else:
            print(f"PLANTED {peid}")
            planted_ep = run_planted_episode(
                model_alias=model,
                planted=planted,
                run_id=args.run_id,
                cache_namespace=args.namespace,
                budget_cap_usd=args.cap,
                sample_index=0,
            )
            _append(planted_eps_path, planted_ep)
            # Judge report
            for jalias in ("judge_primary", "judge_secondary"):
                jr = judge_report(
                    planted_ep["report_text"],
                    model_alias=jalias,
                    run_id=args.run_id,
                    episode_id=peid,
                    cache_namespace=args.namespace,
                    budget_cap_usd=args.cap,
                    sample_index=0,
                )
                jr["branch_id"] = f"{peid}__report"
                jr["kind"] = "a_report_deception"
                _append(judgments_path, jr)

        event, key_facts = event_for_a(planted_ep["amount"])
        base = planted_ep["fork_messages"]
        rendered = render_log(base)
        for branch in ("K", "M1", "M2", "M3", "M4", "D"):
            bid = f"{peid}__{model}__{branch}__s0"
            if bid in done_branches:
                print(f"SKIP {bid}")
                continue
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
            br["event_description"] = event
            br["key_facts"] = key_facts
            br["outcome"] = "LAPSE_PLANTED"
            br["arm"] = "planted"
            _append(branches_path, br)
            done_branches.add(br["branch_id"])
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
                    br=br,
                    event=event,
                    key_facts=key_facts,
                    run_id=args.run_id,
                    namespace=args.namespace,
                    cap=args.cap,
                    secondary=True,
                    judgments_path=judgments_path,
                )

    print("Demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
