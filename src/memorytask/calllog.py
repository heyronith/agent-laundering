"""JSONL call logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "call_id",
    "run_id",
    "task_id",
    "role",
    "timestamp_utc",
    "attempt",
    "model_alias",
    "model_slug_requested",
    "model_served",
    "provider_requested",
    "provider_served",
    "request",
    "response_raw",
    "usage",
    "cost_usd",
    "cost_source",
    "latency_ms",
    "http_status",
    "error",
    "cache_key",
    "cache_hit",
    "cache_namespace",
    "sample_index",
    "seed",
    "provider_mismatch",
)


def _assert_no_secrets(obj: Any) -> None:
    blob = json.dumps(obj, ensure_ascii=False)
    lowered = blob.lower()
    if "authorization" in lowered and "bearer" in lowered:
        raise ValueError("Refusing to log Authorization bearer credentials")
    if ("sk-or" + "-") in blob:
        raise ValueError("Refusing to log OpenRouter API key material")


def append_call_log(logs_dir: Path, run_id: str, record: dict[str, Any]) -> Path:
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"Missing required log field: {field}")
    _assert_no_secrets(record)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{run_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def iter_run_logs(logs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not logs_dir.exists():
        return rows
    for path in sorted(logs_dir.glob("*.jsonl")):
        if path.name == "budget_ledger.jsonl":
            continue
        rows.extend(read_jsonl(path))
    return rows
