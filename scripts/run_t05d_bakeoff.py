#!/usr/bin/env python3
"""T05d judge bake-off: 4 judges × 2 prompts × 2 repeats × 50 items = 800 calls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from memorytask.budget import total_spend_usd  # noqa: E402
from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.client import chat  # noqa: E402
from memorytask.config import repo_root  # noqa: E402
from memorytask.fork.judges import MEMORY_LABELS, fill_template, parse_labeled_json  # noqa: E402

JUDGES = ("jb_gem31pro", "jb_opus46", "jb_oss120b", "jb_gem38flash")
PROMPTS = {
    "P0": "memory_note.txt",
    "P1": "memory_note_rubric.txt",
}
REPEATS = (0, 1)


def run_cost(run_id: str) -> float:
    root = repo_root()
    return sum(
        float(r.get("cost_usd") or 0.0)
        for r in iter_run_logs(root / "logs")
        if r.get("run_id") == run_id and not r.get("cache_hit")
    )


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") and row.get("label") != "PARSE_FAIL":
            out.add(row["call_key"])
        elif row.get("label") == "PARSE_FAIL" and row.get("attempts", 1) >= 2:
            out.add(row["call_key"])
    return out


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")


def judge_one(
    *,
    model_alias: str,
    prompt_id: str,
    template: str,
    item: dict,
    run_id: str,
    cache_namespace: str,
    sample_index: int,
    budget_cap_usd: float,
) -> dict:
    prompt = fill_template(
        template,
        event_description=item["event_description"],
        note=item["note"],
        key_facts=item["key_facts"],
    )
    # Safety: never leak model/outcome into prompt (inputs are sheet fields only).
    for bad in ("deepseek", "glm", "luna", "minimax", "outcome", "LAPSE_"):
        if bad.lower() in prompt.lower() and bad in ("outcome", "LAPSE_"):
            pass  # outcome/LAPSE shouldn't appear; model names may appear inside notes legitimately
    call_ids: list[str] = []
    last_text = ""
    parsed = None
    for attempt in range(2):
        result = chat(
            model_alias=model_alias,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            cache_namespace=cache_namespace,
            run_id=run_id,
            task_id=f"{item['item_id']}__{prompt_id}__{model_alias}__s{sample_index}",
            role="judge",
            sample_index=sample_index * 1000 + attempt,
            budget_cap_usd=budget_cap_usd,
        )
        if result.get("call_id"):
            call_ids.append(result["call_id"])
        raw = result.get("response_raw") or {}
        choices = raw.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                (p.get("text") or "") if isinstance(p, dict) else str(p) for p in content
            )
        last_text = str(content)
        parsed = parse_labeled_json(last_text, MEMORY_LABELS)
        if parsed:
            break
    row = {
        "item_id": item["item_id"],
        "model_alias": model_alias,
        "prompt_id": prompt_id,
        "repeat": sample_index,
        "cache_namespace": cache_namespace,
        "call_ids": call_ids,
        "call_key": f"{item['item_id']}__{prompt_id}__{model_alias}__r{sample_index}",
        "attempts": len(call_ids),
        "label": parsed["label"] if parsed else "PARSE_FAIL",
        "evidence": (parsed or {}).get("evidence"),
        "reason": (parsed or {}).get("reason"),
        "raw_text": last_text[:2000],
    }
    return row


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t05d_bakeoff")
    ap.add_argument("--cap", type=float, default=6.0)
    ap.add_argument("--sheet", default="coding/t05_sheet.csv")
    args = ap.parse_args()

    root = repo_root()
    run_dir = root / "runs" / args.run_id
    out_path = run_dir / "bakeoff_judgments.jsonl"
    templates = {
        pid: (root / "prompts" / "judges" / fname).read_text(encoding="utf-8")
        for pid, fname in PROMPTS.items()
    }

    items = []
    with (root / args.sheet).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append(
                {
                    "item_id": row["item_id"],
                    "event_description": row["event_description"],
                    "key_facts": row["key_facts"],
                    "note": row["note"],
                }
            )
    assert len(items) == 50, len(items)

    done = _load_done(out_path)
    plan = []
    for repeat in REPEATS:
        for prompt_id in PROMPTS:
            for judge in JUDGES:
                for item in items:
                    key = f"{item['item_id']}__{prompt_id}__{judge}__r{repeat}"
                    plan.append((repeat, prompt_id, judge, item, key))

    total = len(plan)
    print(f"plan={total} already_done={len(done)} spend={run_cost(args.run_id):.4f}")
    for i, (repeat, prompt_id, judge, item, key) in enumerate(plan, 1):
        if key in done:
            continue
        spent = run_cost(args.run_id)
        if spent > args.cap - 0.15:
            print(f"STOP: spend {spent:.3f} near cap {args.cap}")
            break
        ns = f"t05d_bakeoff_r{repeat}"
        print(f"[{i}/{total}] {key} spend={spent:.3f}")
        try:
            row = judge_one(
                model_alias=judge,
                prompt_id=prompt_id,
                template=templates[prompt_id],
                item=item,
                run_id=args.run_id,
                cache_namespace=ns,
                sample_index=repeat,
                budget_cap_usd=args.cap,
            )
        except Exception as exc:
            msg = str(exc)
            print(f"  FATAL {type(exc).__name__}: {msg[:300]}")
            if "402" in msg or "credit" in msg.lower() or "BudgetExceeded" in type(exc).__name__:
                break
            raise
        _append(out_path, row)
        done.add(key)
        if row["label"] == "PARSE_FAIL":
            print("  PARSE_FAIL")
        else:
            print(f"  -> {row['label']}")

    print(
        f"Done. judgments={sum(1 for _ in open(out_path))} "
        f"spend={run_cost(args.run_id):.4f} ledger={total_spend_usd(root / 'logs'):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
