from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest

from memorytask.calllog import REQUIRED_FIELDS, append_call_log


def _sample_record(**overrides):
    base = {
        "call_id": str(uuid.uuid4()),
        "run_id": "test_run",
        "task_id": "t",
        "role": "subject",
        "timestamp_utc": "2026-11-09T00:00:00Z",
        "attempt": 1,
        "model_alias": "deepseek",
        "model_slug_requested": "deepseek/deepseek-v4.1-flash-1017",
        "model_served": "deepseek/deepseek-v4.1-flash-1017",
        "provider_requested": "Nebius",
        "provider_served": "Nebius",
        "request": {"messages": [{"role": "user", "content": "hi"}], "tools": None, "params": {}},
        "response_raw": {"id": "gen_1", "choices": []},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "cost_usd": 0.0,
        "cost_source": "response",
        "latency_ms": 10,
        "http_status": 200,
        "error": None,
        "cache_key": "abc",
        "cache_hit": False,
        "cache_namespace": "t01_smoke",
        "sample_index": 0,
        "seed": None,
        "provider_mismatch": False,
    }
    base.update(overrides)
    return base


def test_logger_writes_valid_jsonl_with_required_fields(tmp_path: Path):
    record = _sample_record()
    path = append_call_log(tmp_path, "test_run", record)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    for field in REQUIRED_FIELDS:
        assert field in loaded


def test_logger_rejects_api_key_material(tmp_path: Path):
    record = _sample_record(request={"messages": [], "api_key": "sk_or_v1-00deadbeef"})
    with pytest.raises(ValueError, match="API key"):
        append_call_log(tmp_path, "test_run", record)


def test_logger_rejects_authorization_bearer(tmp_path: Path):
    record = _sample_record(
        request={"headers": {"Authorization": "Bearer sk_or_v1-00deadbeef"}}
    )
    with pytest.raises(ValueError, match="Authorization"):
        append_call_log(tmp_path, "test_run", record)


def test_repo_has_no_sk_or_in_committed_files():
    """Grep for OpenRouter key-shaped secrets in git-tracked files only."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "grep",
            "-nE",
            r"sk_or_v1-[a-fA-F0-9]{32,}",
            "--",
            ".",
            ":(exclude).env",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)
    assert result.stdout.strip() == "", f"Found secret-shaped sk_or_ in tracked files:\n{result.stdout}"
