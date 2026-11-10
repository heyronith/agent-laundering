"""Resume/cache index built from committed JSONL call logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from memorytask.calllog import iter_run_logs


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_cache_key(
    *,
    model_slug: str,
    provider_order: list[str],
    messages: list[dict[str, Any]],
    tools: Any,
    response_format: Any,
    temperature: float | None,
    max_tokens: int | None,
    seed: int | None,
    sample_index: int,
    cache_namespace: str,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    stop: list[str] | None = None,
) -> str:
    payload = {
        "cache_namespace": cache_namespace,
        "model_slug": model_slug,
        "provider_order": provider_order,
        "messages": messages,
        "tools": tools,
        "response_format": response_format,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "sample_index": sample_index,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def is_successful_call(record: dict[str, Any]) -> bool:
    if record.get("error") is not None:
        return False
    status = record.get("http_status")
    if status is None:
        return False
    if int(status) >= 400:
        return False
    if record.get("response_raw") is None:
        return False
    return True


def build_cache_index(logs_dir: Path) -> dict[str, dict[str, Any]]:
    """Map cache_key -> first successful call record (logs are source of truth)."""
    index: dict[str, dict[str, Any]] = {}
    for record in iter_run_logs(logs_dir):
        if not is_successful_call(record):
            continue
        key = record.get("cache_key")
        if not key:
            continue
        existing = index.get(key)
        if existing is None:
            index[key] = record
            continue
        if existing.get("cache_hit") and not record.get("cache_hit"):
            index[key] = record
    return index


class ProcessCacheIndex:
    """In-memory cache index: scan logs once per logs_dir per process, then update in place."""

    def __init__(self) -> None:
        self._indexes: dict[str, dict[str, dict[str, Any]]] = {}
        self.build_counts: dict[str, int] = {}

    def _key(self, logs_dir: Path) -> str:
        return str(logs_dir.resolve())

    def get(self, logs_dir: Path) -> dict[str, dict[str, Any]]:
        key = self._key(logs_dir)
        if key not in self._indexes:
            self._indexes[key] = build_cache_index(logs_dir)
            self.build_counts[key] = self.build_counts.get(key, 0) + 1
        return self._indexes[key]

    def update_after_success(self, logs_dir: Path, record: dict[str, Any]) -> None:
        if not is_successful_call(record):
            return
        key = record.get("cache_key")
        if not key:
            return
        index = self.get(logs_dir)
        existing = index.get(key)
        if existing is None:
            index[key] = record
            return
        if existing.get("cache_hit") and not record.get("cache_hit"):
            index[key] = record

    def reset(self) -> None:
        self._indexes.clear()
        self.build_counts.clear()


PROCESS_CACHE_INDEX = ProcessCacheIndex()


def reset_process_cache() -> None:
    PROCESS_CACHE_INDEX.reset()


def write_cache_index(cache_dir: Path, index: dict[str, dict[str, Any]]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "index.json"
    slim = {
        key: {
            "call_id": rec.get("call_id"),
            "run_id": rec.get("run_id"),
            "model_alias": rec.get("model_alias"),
            "response_raw": rec.get("response_raw"),
            "usage": rec.get("usage"),
            "model_served": rec.get("model_served"),
            "provider_served": rec.get("provider_served"),
        }
        for key, rec in index.items()
    }
    path.write_text(canonical_json(slim) + "\n", encoding="utf-8")
    return path
