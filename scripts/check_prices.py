#!/usr/bin/env python3
"""Fetch OpenRouter models + endpoints for configured models; write price snapshots."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memorytask.config import (  # noqa: E402
    get_api_key,
    get_blocked_models,
    get_models,
    load_run_defaults,
    repo_root,
)


def _per_million(price_per_token: str | float | None) -> float | None:
    if price_per_token is None:
        return None
    return float(price_per_token) * 1_000_000.0


def main() -> int:
    root = repo_root()
    defaults = load_run_defaults()
    prices_dir = root / defaults.get("prices_dir", "data/prices")
    prices_dir.mkdir(parents=True, exist_ok=True)
    base = defaults.get("openrouter_base_url", "https://openrouter.ai/api/v1").rstrip("/")
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    models = get_models()
    blocked = get_blocked_models()
    if blocked:
        print("BLOCKED MODELS (not fetched; standing rule: no silent substitute):")
        for alias, entry in blocked.items():
            print(f"  - {alias}: expected={entry.get('expected_slug')} reason={entry.get('reason')}")
            alts = entry.get("alternatives") or []
            if alts:
                print("    alternatives:")
                for a in alts:
                    print(f"      - {a}")
        print()

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    with httpx.Client(timeout=60.0) as client:
        models_resp = client.get(f"{base}/models", headers=headers)
        models_resp.raise_for_status()
        models_json = models_resp.json()
        by_id = {m["id"]: m for m in models_json.get("data", [])}

        missing = []
        for alias, entry in models.items():
            slug = entry["slug"]
            if slug not in by_id:
                missing.append((alias, slug))
        if missing:
            print("STOP: configured model slug(s) not found on OpenRouter:")
            for alias, slug in missing:
                print(f"  - {alias}: {slug}")
            return 2

        endpoints_out: dict[str, Any] = {"fetched_utc": datetime.now(timezone.utc).isoformat(), "endpoints": {}}
        prices_out: dict[str, Any] = {
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "by_slug": {},
            "models": {},
        }

        rows = []
        for alias, entry in models.items():
            slug = entry["slug"]
            author, name = slug.split("/", 1)
            ep_resp = client.get(f"{base}/models/{author}/{name}/endpoints", headers=headers)
            ep_resp.raise_for_status()
            ep_json = ep_resp.json()
            endpoints_out["endpoints"][slug] = ep_json

            payload = ep_json.get("data", ep_json)
            eps = payload.get("endpoints", []) if isinstance(payload, dict) else []
            pinned = entry["provider_order"][0]
            pinned_eps = [e for e in eps if e.get("provider_name") == pinned]
            if not pinned_eps:
                available = sorted({e.get("provider_name") for e in eps})
                print(f"STOP: first-party provider {pinned!r} not available for {slug}")
                print(f"  available providers: {available}")
                return 3

            chosen = pinned_eps[0]
            pricing = chosen.get("pricing") or {}
            inp = _per_million(pricing.get("prompt"))
            out = _per_million(pricing.get("completion"))
            quant = chosen.get("quantization")
            ctx = chosen.get("context_length") or by_id[slug].get("context_length")

            prices_out["models"][alias] = {
                "slug": slug,
                "pinned_provider": pinned,
                "input_per_million": inp,
                "output_per_million": out,
                "context_length": ctx,
                "quantization": quant,
                "allow_fallbacks": entry.get("allow_fallbacks"),
            }
            prices_out["by_slug"][slug] = prices_out["models"][alias]
            rows.append((alias, slug, pinned, inp, out, ctx, quant))

    prices_path = prices_dir / f"prices_{day}.json"
    endpoints_path = prices_dir / f"endpoints_{day}.json"
    prices_path.write_text(json.dumps(prices_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    endpoints_path.write_text(json.dumps(endpoints_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {prices_path}")
    print(f"Wrote {endpoints_path}")
    print()
    header = f"{'alias':16} {'provider':18} {'in$/1M':>10} {'out$/1M':>10} {'context':>10} {'quant':12}"
    print(header)
    print("-" * len(header))
    for alias, slug, pinned, inp, out, ctx, quant in rows:
        print(
            f"{alias:16} {pinned:18} {inp:10.4f} {out:10.4f} {str(ctx):>10} {str(quant):12}"
        )
        print(f"  slug={slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
