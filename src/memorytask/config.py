"""Load YAML configs and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def load_env(env_path: Path | None = None) -> None:
    path = env_path or (REPO_ROOT / ".env")
    load_dotenv(path, override=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_run_defaults(path: Path | None = None) -> dict[str, Any]:
    return _read_yaml(path or (REPO_ROOT / "configs" / "run_defaults.yaml"))


def load_models_config(path: Path | None = None) -> dict[str, Any]:
    return _read_yaml(path or (REPO_ROOT / "configs" / "models.yaml"))


def get_models(path: Path | None = None) -> dict[str, dict[str, Any]]:
    cfg = load_models_config(path)
    models = cfg.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("configs/models.yaml: 'models' must be a mapping")
    return models


def get_blocked_models(path: Path | None = None) -> dict[str, dict[str, Any]]:
    cfg = load_models_config(path)
    blocked = cfg.get("blocked_models") or {}
    if not isinstance(blocked, dict):
        raise ValueError("configs/models.yaml: 'blocked_models' must be a mapping")
    return blocked


def get_model(alias: str, path: Path | None = None) -> dict[str, Any]:
    models = get_models(path)
    if alias not in models:
        raise KeyError(f"Unknown model alias: {alias}")
    return models[alias]


def validate_models_pinned(path: Path | None = None) -> None:
    models = get_models(path)
    for alias, entry in models.items():
        order = entry.get("provider_order")
        if not order or not isinstance(order, list) or not all(isinstance(x, str) for x in order):
            raise ValueError(f"Model {alias}: provider_order must be a non-empty list of strings")
        if entry.get("allow_fallbacks") is not False:
            raise ValueError(f"Model {alias}: allow_fallbacks must be false")
        if not entry.get("slug"):
            raise ValueError(f"Model {alias}: slug is required")


def get_api_key() -> str:
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env")
    return key
