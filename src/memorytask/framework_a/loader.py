"""Load vendored Apollo insider-trading prompt JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memorytask.config import repo_root

PROMPTS_ROOT = repo_root() / "vendor" / "insider-trading" / "prompts"


def prompts_root() -> Path:
    return PROMPTS_ROOT


def list_prompt_files(root: Path | None = None) -> list[Path]:
    base = root or PROMPTS_ROOT
    return sorted(p for p in base.rglob("*.json") if p.is_file())


def resolve_prompt_path(rel_or_name: str, root: Path | None = None) -> Path:
    base = root or PROMPTS_ROOT
    candidate = Path(rel_or_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    path = base / rel_or_name
    if path.exists():
        return path
    raise FileNotFoundError(f"Prompt file not found: {rel_or_name} (under {base})")


def load_prompt(rel_or_name: str, root: Path | None = None) -> dict[str, Any]:
    path = resolve_prompt_path(rel_or_name, root=root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "messages" not in data:
        raise ValueError(f"{path}: missing messages")
    if "misalignment_string" not in data:
        raise ValueError(f"{path}: missing misalignment_string")
    data["_path"] = str(path.relative_to(root or PROMPTS_ROOT)).replace("\\", "/")
    data["_abspath"] = str(path)
    return data
