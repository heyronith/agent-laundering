#!/usr/bin/env python3
"""Run Framework A episodes + judges (resumable via cache)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import repo_root  # noqa: E402
from memorytask.framework_a.episode import run_episode  # noqa: E402
from memorytask.framework_a.judges import judge_doubling, judge_report  # noqa: E402
from memorytask.framework_a.loader import load_prompt  # noqa: E402


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_done_episode_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        done.add(json.loads(line)["episode_id"])
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cap", type=float, required=True)
    args = parser.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    episodes_path = run_dir / "episodes.jsonl"
    judgments_path = run_dir / "judgments.jsonl"
    done = _load_done_episode_ids(episodes_path)

    total = len(args.files) * len(args.models) * args.n
    completed = 0

    for prompt_file in args.files:
        prompt = load_prompt(prompt_file)
        for model in args.models:
            for sample_index in range(args.n):
                episode_id = (
                    f"{model}__{prompt['_path'].replace('/', '__')}__{sample_index}"
                )
                completed += 1
                if episode_id in done:
                    print(f"[{completed}/{total}] SKIP cached episode {episode_id}")
                    continue
                print(f"[{completed}/{total}] RUN {episode_id}")
                ep = run_episode(
                    model_alias=model,
                    prompt_file=prompt_file,
                    sample_index=sample_index,
                    run_id=args.run_id,
                    cache_namespace=args.namespace,
                    budget_cap_usd=args.cap,
                    prompt_data=prompt,
                )
                _append_jsonl(episodes_path, ep)

                if ep.get("misaligned_original") and ep.get("report_text") is not None:
                    for judge_alias in ("judge_primary", "judge_secondary"):
                        jr = judge_report(
                            ep["report_text"],
                            model_alias=judge_alias,
                            run_id=args.run_id,
                            episode_id=ep["episode_id"],
                            cache_namespace=args.namespace,
                            budget_cap_usd=args.cap,
                            sample_index=sample_index,
                        )
                        jr["prompt_file"] = ep["prompt_file"]
                        jr["subject_model"] = ep["model_alias"]
                        jr["sample_index"] = sample_index
                        _append_jsonl(judgments_path, jr)

                    if ep.get("doubling_text") is not None and ep.get("doubling_found") is not None:
                        # Judge doubling whenever step 3 produced text (report was found).
                        for judge_alias in ("judge_primary", "judge_secondary"):
                            jd = judge_doubling(
                                ep["doubling_text"],
                                model_alias=judge_alias,
                                run_id=args.run_id,
                                episode_id=ep["episode_id"],
                                cache_namespace=args.namespace,
                                budget_cap_usd=args.cap,
                                sample_index=sample_index,
                            )
                            jd["prompt_file"] = ep["prompt_file"]
                            jd["subject_model"] = ep["model_alias"]
                            jd["sample_index"] = sample_index
                            _append_jsonl(judgments_path, jd)

                done.add(episode_id)

    print(f"Done. Episodes at {episodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
