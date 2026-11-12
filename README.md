# agent-laundering

Do AI agents *launder* their own misconduct through memory notes — repeating a false claim about what they did even when the session evidence says otherwise?

This repository contains the **code, prompts, preregistration, coding rubric, and aggregate results** for a small open-model study of that question. Write-up: *[link forthcoming]*.

## What's included vs withheld

**Included:** Python package (`src/`), runners and analysis scripts, frozen prompts, model configs (no API keys), tests, `docs/preregistration_e2.md`, `docs/coding_rubric.md`, `RESULTS.md`, and `results/summary.json`.

**Withheld (gitignored locally; not in this public tree):** raw API logs, episode trajectories, benchmark task text, human coding sheets, and internal process briefs. Reasons: (1) the Upward Deceivers benchmark has no redistribution license beyond research use with citation; (2) Apollo insider-trading materials prohibit training use and are out of scope for the released arm; (3) logs and sheets can contain personal filesystem paths and are unnecessary once aggregates are published.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# set OPENROUTER_API_KEY in the environment (required only for live model calls)
make test
```

## Upward Deceivers (Framework B tasks)

Clone the upstream benchmark at the pinned commit and place it at `vendor/upward-deception/`:

```bash
git clone https://github.com/QingyuLiu/Agentic-Upward-Deception vendor/upward-deception
cd vendor/upward-deception
git checkout b26151c7195111271feba64fb15db6227f25f51b
```

Cite Guo et al. (arXiv:2512.04864). Do not redistribute that tree beyond what its authors allow. This repo's MIT license covers **our** code only, not third-party benchmarks.

## E2 pipeline (evidence format)

Preregistration: [`docs/preregistration_e2.md`](docs/preregistration_e2.md). Tag `prereg-e2` marks the commit before any E2 data collection:

```bash
git log -1 --format=%cI prereg-e2
```

With local episode data restored under `runs/t05b_killtest/` (not shipped here):

```bash
python scripts/run_e2_main.py          # stimuli → materials → successors → judges
python analysis/e2.py                  # → reports/E2_report.md (local)
python scripts/e2_validation.py        # after human sheets are filled
python scripts/build_results_summary.py
```

Aggregate tables live in [`RESULTS.md`](RESULTS.md) / [`results/summary.json`](results/summary.json). On a fresh clone without local data, `make test` still passes; the results recompute test skips.

## Citations

- Guo et al. *Agentic Upward Deception*. arXiv:2512.04864 (Upward Deceivers / Framework B).
- Scheurer, Balesni, & Hobbhahn. *Large Language Models can Strategically Deceive their Users when Put Under Pressure*. arXiv:2311.07592 (Apollo insider-trading eval; used in earlier arms of this project, not required to run E2).
- Related agent-memory / deception measurement work as cited in the forthcoming post.

## License

MIT © Ronith Sharmila — see [`LICENSE`](LICENSE). Vendored or separately downloaded benchmarks are **not** covered by this license.
