from __future__ import annotations

from pathlib import Path

import pytest

from memorytask.budget import BudgetExceeded, append_ledger, check_budget


def test_budget_raises_on_run_cap(tmp_path: Path):
    append_ledger(
        tmp_path,
        call_id="c1",
        run_id="r1",
        task_id="t",
        cost_usd=0.80,
        timestamp="2026-11-09T00:00:00Z",
    )
    with pytest.raises(BudgetExceeded, match="Run budget exceeded"):
        check_budget(
            tmp_path,
            run_id="r1",
            run_cap_usd=1.0,
            global_cap_usd=95.0,
            estimate_usd=0.30,
        )


def test_budget_raises_on_global_cap(tmp_path: Path):
    append_ledger(
        tmp_path,
        call_id="c1",
        run_id="other",
        task_id="t",
        cost_usd=94.5,
        timestamp="2026-11-09T00:00:00Z",
    )
    with pytest.raises(BudgetExceeded, match="Global budget exceeded"):
        check_budget(
            tmp_path,
            run_id="r2",
            run_cap_usd=10.0,
            global_cap_usd=95.0,
            estimate_usd=1.0,
        )


def test_budget_allows_when_under_caps(tmp_path: Path):
    append_ledger(
        tmp_path,
        call_id="c1",
        run_id="r1",
        task_id="t",
        cost_usd=0.10,
        timestamp="2026-11-09T00:00:00Z",
    )
    check_budget(
        tmp_path,
        run_id="r1",
        run_cap_usd=1.0,
        global_cap_usd=95.0,
        estimate_usd=0.20,
    )
