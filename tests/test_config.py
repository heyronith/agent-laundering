from __future__ import annotations

from memorytask.config import get_models, load_run_defaults, validate_models_pinned


def test_config_loads_and_providers_pinned():
    defaults = load_run_defaults()
    assert defaults["global_cap_usd"] == 95.00
    models = get_models()
    assert models, "expected at least one configured model"
    validate_models_pinned()
    for alias, entry in models.items():
        assert entry["allow_fallbacks"] is False
        assert isinstance(entry["provider_order"], list) and len(entry["provider_order"]) >= 1
        assert entry["slug"]
        assert entry["role"] in {"subject", "judge"}
