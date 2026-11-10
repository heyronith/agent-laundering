#!/usr/bin/env python3
"""Run Framework B (Upward Deception) episodes — resumable via episodes.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from memorytask.config import repo_root  # noqa: E402
from memorytask.framework_b.runner import load_tasks, run_episode  # noqa: E402


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _load_done_episode_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        done.add(json.loads(line)["episode_id"])
    return done


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", type=int, required=True, help="Task types e.g. 1 4 5")
    parser.add_argument(
        "--n-per-type",
        nargs="+",
        type=int,
        required=True,
        help="Number of tasks per type (same order as --types)",
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--seed-tasks", type=int, default=0, help="RNG seed for type-1 file content")
    parser.add_argument("--samples", type=int, default=1, help="Samples per task (episode_sample)")
    args = parser.parse_args()

    if len(args.n_per_type) != len(args.types):
        print("ERROR: --n-per-type length must match --types", file=sys.stderr)
        return 2

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    episodes_path = run_dir / "episodes.jsonl"
    done = _load_done_episode_ids(episodes_path)

    plan: list[tuple[str, dict, int]] = []
    for task_type, n in zip(args.types, args.n_per_type):
        tasks = load_tasks(task_type)[:n]
        for task in tasks:
            for sample in range(args.samples):
                for model in args.models:
                    plan.append((model, task, sample))

    total = len(plan)
    completed = 0
    for model, task, sample in plan:
        episode_id = f"{model}__type{int(task['task_type'])}__idx{int(task['task_index'])}__s{sample}"
        completed += 1
        if episode_id in done:
            print(f"[{completed}/{total}] SKIP {episode_id}")
            continue
        print(f"[{completed}/{total}] RUN {episode_id}")
        seed = args.seed_tasks + int(task["task_index"]) * 1009 + sample
        try:
            ep = run_episode(
                model_alias=model,
                task=task,
                episode_sample=sample,
                run_id=args.run_id,
                cache_namespace=args.namespace,
                budget_cap_usd=args.cap,
                seed=seed,
            )
        except Exception as exc:
            print(f"  FATAL {episode_id}: {type(exc).__name__}: {exc}")
            ep = {
                "episode_id": episode_id,
                "model": model,
                "model_alias": model,
                "task_type": int(task["task_type"]),
                "task_index": int(task["task_index"]),
                "sample_index": sample,
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
            ep_dir = run_dir / "episodes" / episode_id
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "episode.json").write_text(json.dumps(ep, indent=2, default=str), encoding="utf-8")
            (ep_dir / "final_answer.txt").write_text("", encoding="utf-8")
            (ep_dir / "trajectory.json").write_text(
                json.dumps({"episode_id": episode_id, "last_step_messages": None, "memory_steps": []}, indent=2),
                encoding="utf-8",
            )
            (ep_dir / "agent_stdout.log").write_text(str(ep["error"]), encoding="utf-8")
        _append_jsonl(episodes_path, ep)
        done.add(episode_id)
        print(
            f"  steps={ep.get('n_steps')} broken_tools={ep.get('broken_tool_calls')} "
            f"result={ep.get('upstream_result')} err={ep.get('error')}"
        )

    print(f"Done. {len(done)} episodes in {episodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
