"""Budget guard and global spend ledger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetExceeded(Exception):
    """Raised when a paid call would exceed a run or global budget cap."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ledger_path(logs_dir: Path) -> Path:
    return logs_dir / "budget_ledger.jsonl"


def read_ledger(logs_dir: Path) -> list[dict[str, Any]]:
    path = ledger_path(logs_dir)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def total_spend_usd(logs_dir: Path) -> float:
    return sum(float(r.get("cost_usd") or 0.0) for r in read_ledger(logs_dir))


def run_spend_usd(logs_dir: Path, run_id: str) -> float:
    return sum(
        float(r.get("cost_usd") or 0.0)
        for r in read_ledger(logs_dir)
        if r.get("run_id") == run_id
    )


def append_ledger(
    logs_dir: Path,
    *,
    call_id: str,
    run_id: str,
    task_id: str,
    cost_usd: float,
    timestamp: str | None = None,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "call_id": call_id,
        "run_id": run_id,
        "task_id": task_id,
        "cost_usd": float(cost_usd),
        "timestamp": timestamp or _utc_now(),
    }
    with ledger_path(logs_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def estimate_worst_case_cost_usd(
    *,
    prompt_tokens_estimate: int,
    max_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Worst-case estimate: assumed prompt size + full max_tokens completion."""
    in_cost = (prompt_tokens_estimate / 1_000_000.0) * input_price_per_million
    out_cost = (max_tokens / 1_000_000.0) * output_price_per_million
    return in_cost + out_cost


def check_budget(
    logs_dir: Path,
    *,
    run_id: str,
    run_cap_usd: float,
    global_cap_usd: float,
    estimate_usd: float,
) -> None:
    run_spent = run_spend_usd(logs_dir, run_id)
    global_spent = total_spend_usd(logs_dir)
    if (run_spent + estimate_usd) > run_cap_usd:
        raise BudgetExceeded(
            f"Run budget exceeded: run_spend={run_spent:.6f} + estimate={estimate_usd:.6f} "
            f"> run_cap={run_cap_usd:.6f} (run_id={run_id})"
        )
    if (global_spent + estimate_usd) > global_cap_usd:
        raise BudgetExceeded(
            f"Global budget exceeded: global_spend={global_spent:.6f} + estimate={estimate_usd:.6f} "
            f"> global_cap={global_cap_usd:.6f}"
        )
