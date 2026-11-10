# memorytask

Do AI agents launder their own misconduct in memory?

## Setup

```bash
cp .env.example .env   # set OPENROUTER_API_KEY
make setup             # Python 3.11 venv + editable install
```

Requires Python 3.11+.

## Make targets

| Target | Purpose |
|--------|---------|
| `make setup` | Create `.venv`, install pinned deps, `pip install -e .` |
| `make test` | Offline pytest suite (no network) |
| `make prices` | Snapshot OpenRouter prices/endpoints into `data/prices/` |
| `make smoke` | T01 smoke test (`t01_smoke`, $1.00 cap) |
| `make verify` | Recompute spend from logs vs ledger |

## Standing rules

1. Never invent numbers. Every number in any report must come from a committed script reading committed logs.
2. Never commit secrets. `OPENROUTER_API_KEY` lives only in `.env`, which is git-ignored.
3. Do not silently substitute anything. If a model slug or provider is unavailable, STOP and report the alternatives in the task report. Do not pick one yourself.
4. Do only what this brief asks. No experiments beyond the smoke test.
5. Every API call is logged; prompt changes vs original frameworks must be listed as deviations.
