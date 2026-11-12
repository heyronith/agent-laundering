"""Recompute results/summary.json from local sources when present; skip when absent."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from memorytask.config import repo_root


def _load_builder():
    path = repo_root() / "scripts" / "build_results_summary.py"
    spec = importlib.util.spec_from_file_location("build_results_summary", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources_present(root: Path) -> bool:
    return (root / "reports" / "E2_summary.json").exists()


@pytest.mark.skipif(not _sources_present(repo_root()), reason="local source reports absent")
def test_summary_json_matches_recompute():
    brs = _load_builder()
    root = repo_root()
    summary = brs.build_summary(root)
    committed = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    assert committed == summary


def test_summary_json_is_aggregates_only():
    root = repo_root()
    path = root / "results" / "summary.json"
    if not path.exists():
        pytest.skip("results/summary.json absent")
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert "schema_version" in blob
    assert "description" in blob
    # No item-level payload keys at top level
    for k in ("answer", "raw_text", "item_id", "trajectory", "messages", "disagreements"):
        assert k not in blob
