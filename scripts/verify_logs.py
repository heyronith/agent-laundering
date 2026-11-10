#!/usr/bin/env python3
"""Recompute spend from logs and compare to the budget ledger."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.budget import read_ledger, total_spend_usd  # noqa: E402
from memorytask.calllog import iter_run_logs  # noqa: E402
from memorytask.config import load_run_defaults, repo_root  # noqa: E402


def main() -> int:
    root = repo_root()
    defaults = load_run_defaults()
    logs_dir = root / defaults.get("logs_dir", "logs")
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = iter_run_logs(logs_dir)
    ledger = read_ledger(logs_dir)

    by_model: Counter = Counter()
    by_provider: Counter = Counter()
    by_role: Counter = Counter()
    errors = []
    mismatches = []
    cache_hits = 0
    log_cost = 0.0

    for r in rows:
        by_model[r.get("model_alias") or "unknown"] += 1
        by_provider[r.get("provider_served") or r.get("provider_requested") or "unknown"] += 1
        by_role[r.get("role") or "unknown"] += 1
        if r.get("error") is not None:
            errors.append(
                {
                    "call_id": r.get("call_id"),
                    "run_id": r.get("run_id"),
                    "task_id": r.get("task_id"),
                    "error": r.get("error"),
                    "http_status": r.get("http_status"),
                }
            )
        if r.get("provider_mismatch"):
            mismatches.append(
                {
                    "call_id": r.get("call_id"),
                    "provider_requested": r.get("provider_requested"),
                    "provider_served": r.get("provider_served"),
                }
            )
        if r.get("cache_hit"):
            cache_hits += 1
        # Paid cost: only non-cache rows contribute; cache hits are 0.
        log_cost += float(r.get("cost_usd") or 0.0)

    ledger_cost = sum(float(r.get("cost_usd") or 0.0) for r in ledger)
    ledger_total_fn = total_spend_usd(logs_dir)

    # Paid calls in logs (cost > 0 or non-cache successful) should match ledger entries.
    paid_log_cost = sum(
        float(r.get("cost_usd") or 0.0)
        for r in rows
        if not r.get("cache_hit") and float(r.get("cost_usd") or 0.0) > 0
    )
    # Also include zero-cost paid? Ledger only gets append on successful paid calls.
    # Compare ledger sum to sum of non-cache costs in logs.
    non_cache_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows if not r.get("cache_hit"))

    summary = {
        "calls_total": len(rows),
        "calls_per_model": dict(by_model),
        "calls_per_provider": dict(by_provider),
        "calls_per_role": dict(by_role),
        "errors": errors,
        "provider_mismatches": mismatches,
        "cache_hits": cache_hits,
        "total_cost_usd_from_logs": round(log_cost, 10),
        "total_cost_usd_from_logs_non_cache": round(non_cache_cost, 10),
        "total_cost_usd_from_logs_paid_positive": round(paid_log_cost, 10),
        "total_cost_usd_from_ledger": round(ledger_cost, 10),
        "total_cost_usd_from_ledger_fn": round(ledger_total_fn, 10),
        "ledger_entries": len(ledger),
    }

    out_path = reports_dir / "T01_log_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")

    mismatches_found = []
    # Ledger should match non-cache costs (cache hits log cost_usd=0 and are not ledgered).
    if abs(non_cache_cost - ledger_cost) > 1e-9:
        mismatches_found.append(
            f"cost mismatch: logs_non_cache={non_cache_cost} ledger={ledger_cost}"
        )
    if abs(ledger_cost - ledger_total_fn) > 1e-9:
        mismatches_found.append(
            f"ledger internal mismatch: sum={ledger_cost} fn={ledger_total_fn}"
        )
    if mismatches:
        mismatches_found.append(f"provider_mismatch count={len(mismatches)}")

    if mismatches_found:
        for m in mismatches_found:
            print(f"VERIFY FAIL: {m}")
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
