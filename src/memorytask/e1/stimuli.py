"""K1 / E1 stimulus selection (seed 7)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from memorytask.framework_b.runner import load_tasks


def requested_file_for_task(task_type: int, task: dict[str, Any]) -> str | None:
    """Single requested source file, or None if the task has no single file_name."""
    if task_type == 1:
        name = task.get("file_name")
        return str(name) if name else None
    if task_type == 5:
        # Dataset uses target_file_name (no `file_name` key); treat as the single requested file.
        name = task.get("file_name") or task.get("target_file_name")
        return str(name) if name else None
    if task_type == 3:
        # pptx + txt — no single requested file name → exclude.
        return None
    return task.get("file_name")


def load_m1_notes(branches_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in branches_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        b = json.loads(line)
        if b.get("branch") != "M1":
            continue
        text = (b.get("output_text") or "").strip()
        if text:
            out[str(b["episode_id"])] = text
    return out


def eligible_pool(
    *,
    outcomes_path: Path,
    branches_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (eligible stimuli dicts, excluded-no-file rows)."""
    m1 = load_m1_notes(branches_path)
    task_cache = {t: {int(x["task_index"]): x for x in load_tasks(t)} for t in (1, 3, 5)}
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for line in outcomes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        model = o.get("model") or o.get("model_alias")
        ttype = int(o.get("task_type", -1))
        outcome = o.get("outcome")
        eid = o.get("episode_id")
        if model not in ("glm", "minimax", "deepseek"):
            continue
        if ttype not in (1, 3, 5):
            continue
        if outcome == "NO_ANSWER":
            continue
        if eid not in m1:
            continue
        task = task_cache[ttype].get(int(o["task_index"]))
        if not task:
            continue
        req = requested_file_for_task(ttype, task)
        if not req:
            excluded.append(
                {
                    "episode_id": eid,
                    "model": model,
                    "task_type": ttype,
                    "task_index": o.get("task_index"),
                    "reason": "no_single_requested_file_name",
                }
            )
            continue
        eligible.append(
            {
                "episode_id": eid,
                "model": model,
                "task_type": ttype,
                "task_index": int(o["task_index"]),
                "outcome": outcome,
                "requested_file": req,
                "m1_note": m1[eid],
                "instruction": task.get("instruction") or "",
            }
        )
    return eligible, excluded


def select_k1_stimuli(
    eligible: list[dict[str, Any]],
    *,
    seed: int = 7,
    n_glm: int = 3,
    n_minimax: int = 3,
    n_deepseek: int = 2,
) -> list[dict[str, Any]]:
    """Select 8 episodes: model quotas; types as even as possible; ≥3 VERIFIED + ≥3 HONEST."""
    rng = random.Random(seed)
    quotas = {"glm": n_glm, "minimax": n_minimax, "deepseek": n_deepseek}
    total = sum(quotas.values())
    assert total == 8

    types_present = sorted({e["task_type"] for e in eligible})
    # Even spread across available types.
    type_quota: dict[int, int] = {t: 0 for t in types_present}
    if types_present:
        base, rem = divmod(total, len(types_present))
        for i, t in enumerate(types_present):
            type_quota[t] = base + (1 if i < rem else 0)

    def is_verified(o: str) -> bool:
        return o == "LAPSE_VERIFIED"

    def is_honest(o: str) -> bool:
        return o == "HONEST"

    # Multi-start shuffle until constraints satisfied (or best effort after many tries).
    best: list[dict[str, Any]] | None = None
    best_score = -1
    for attempt in range(5000):
        pool = list(eligible)
        rng.shuffle(pool)
        chosen: list[dict[str, Any]] = []
        model_counts = {m: 0 for m in quotas}
        type_counts = {t: 0 for t in types_present}

        # Pass 1: fill quotas preferring underfilled types
        for e in pool:
            if len(chosen) >= total:
                break
            m, t = e["model"], e["task_type"]
            if model_counts[m] >= quotas[m]:
                continue
            if type_counts.get(t, 0) >= type_quota.get(t, 0):
                continue
            chosen.append(e)
            model_counts[m] += 1
            type_counts[t] = type_counts.get(t, 0) + 1

        # Pass 2: fill remaining model quotas ignoring type soft caps
        if len(chosen) < total:
            chosen_ids = {c["episode_id"] for c in chosen}
            for e in pool:
                if len(chosen) >= total:
                    break
                if e["episode_id"] in chosen_ids:
                    continue
                m = e["model"]
                if model_counts[m] >= quotas[m]:
                    continue
                chosen.append(e)
                model_counts[m] += 1
                chosen_ids.add(e["episode_id"])

        if len(chosen) < total:
            continue

        n_ver = sum(1 for c in chosen if is_verified(c["outcome"]))
        n_hon = sum(1 for c in chosen if is_honest(c["outcome"]))
        # Prefer meeting both floors; then even types.
        type_spread = len({c["task_type"] for c in chosen})
        score = (
            (10 if n_ver >= 3 else n_ver)
            + (10 if n_hon >= 3 else n_hon)
            + type_spread
        )
        if score > best_score:
            best_score = score
            best = chosen
        if n_ver >= 3 and n_hon >= 3 and all(
            model_counts[m] == quotas[m] for m in quotas
        ):
            # Good enough type spread if we used available types
            return chosen

    if best is None:
        raise RuntimeError("Could not select K1 stimuli")
    return best
