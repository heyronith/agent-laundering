# Results (aggregates only)

This file holds **aggregate** statistics from the preregistered E2 experiment,
its human validation, the K1/K1b feasibility runs, and the T05b pilot.
It contains no task text, model answers, raw logs, or human label sheets.

Machine-readable copy: [`results/summary.json`](results/summary.json).

## E2 — evidence format

n_stimuli=77, n_answers=1232, n_judgments=2464.

### Cell table (FALSE_AFFIRM k/n)

**judge_primary**

| cell | k/n | rate |
|---:|---:|---:|
| 1 | 151/154 | 0.981 |
| 2 | 150/154 | 0.974 |
| 3 | 11/154 | 0.071 |
| 4 | 4/154 | 0.026 |
| 5 | 72/154 | 0.468 |
| 6 | 64/154 | 0.416 |
| 7 | 0/154 | 0.000 |
| 8 | 8/154 | 0.052 |

**judge_oss**

| cell | k/n | rate |
|---:|---:|---:|
| 1 | 150/153 | 0.980 |
| 2 | 151/154 | 0.981 |
| 3 | 4/154 | 0.026 |
| 4 | 0/154 | 0.000 |
| 5 | 67/154 | 0.435 |
| 6 | 66/154 | 0.429 |
| 7 | 0/154 | 0.000 |
| 8 | 7/154 | 0.045 |

### Hypotheses

**judge_primary**
- H1: n_episodes=77, pos=42 neg=1 zero=34, p=5.00222e-12, mean RD=0.3929, 95% CI [0.3165, 0.4726]
- H2: n_episodes=77, pos=1 neg=51 zero=25, p=2.35367e-14, p_Holm=4.70735e-14, mean RD=-0.5357, 95% CI [-0.6149, -0.4533]
- H3: n_episodes=77, pos=10 neg=4 zero=63, p=0.179565, p_Holm=0.179565, mean RD=0.0519, 95% CI [0.0000, 0.1133]

**judge_oss**
- H1: n_episodes=77, pos=44 neg=0 zero=33, p=5.68434e-14, mean RD=0.4188, 95% CI [0.3368, 0.5032]
- H2: n_episodes=77, pos=2 neg=55 zero=20, p=2.29539e-14, p_Holm=4.59077e-14, mean RD=-0.5487, 95% CI [-0.6299, -0.4643]
- H3: n_episodes=77, pos=7 neg=6 zero=64, p=1, p_Holm=1, mean RD=0.0065, 95% CI [-0.0506, 0.0634]

### Per model (FALSE_AFFIRM rate by evidence)

**judge_primary**
- deepseek: RAWLOG 18/96=0.188, STATUS 0/96=0.000, NONE 96/96=1.000
- glm: RAWLOG 24/108=0.222, STATUS 0/108=0.000, NONE 103/108=0.954
- minimax: RAWLOG 94/104=0.904, STATUS 15/104=0.144, NONE 102/104=0.981

**judge_oss**
- deepseek: RAWLOG 17/96=0.177, STATUS 0/96=0.000, NONE 96/96=1.000
- glm: RAWLOG 24/108=0.222, STATUS 0/108=0.000, NONE 104/108=0.963
- minimax: RAWLOG 92/104=0.885, STATUS 4/104=0.038, NONE 101/103=0.981

### Per task type

**judge_primary**
- type 1: RAWLOG 33/104=0.317, STATUS 5/104=0.048, NONE 103/104=0.990
- type 3: RAWLOG 51/116=0.440, STATUS 8/116=0.069, NONE 115/116=0.991
- type 5: RAWLOG 52/88=0.591, STATUS 2/88=0.023, NONE 83/88=0.943

**judge_oss**
- type 1: RAWLOG 29/104=0.279, STATUS 2/104=0.019, NONE 101/103=0.981
- type 3: RAWLOG 58/116=0.500, STATUS 2/116=0.017, NONE 116/116=1.000
- type 5: RAWLOG 46/88=0.523, STATUS 0/88=0.000, NONE 84/88=0.955

### Faithful controls

**judge_primary**
- cell 7: 0/154=0.000
- cell 8: 8/154=0.052

**judge_oss**
- cell 7: 0/154=0.000
- cell 8: 7/154=0.045

### Log diagnosticity (descriptive)

Loose = existing `is_diagnostic` (error cue and filename **or** the word `file`).
Strict = error cue and the requested filename (full or basename); generic `file` does not count.

| | n | rate |
|---|---:|---:|
| loose | 72/77 | 0.935 |
| strict | 71/77 | 0.922 |

| model | n | loose | strict |
|---|---:|---:|---:|
| deepseek | 24 | 23/24=0.958 | 22/24=0.917 |
| glm | 27 | 24/27=0.889 | 24/27=0.889 |
| minimax | 26 | 25/26=0.962 | 25/26=0.962 |

RAWLOG FALSE_AFFIRM rate (judge_primary) by strict diagnostic:

| strict | k/n | rate |
|---|---:|---:|
| yes | 121/284 | 0.426 |
| no | 15/24 | 0.625 |

## E2 validation (judge vs human)

Coder 2 n=160; coder 1 n=50; coder1–coder2 κ=1.000 (n=50).
Judge-based H1–H3 stand: True.

| judge | κ | sensitivity | specificity | n |
|---|---:|---:|---:|---:|
| judge_primary | 0.922 | 0.925 | 0.989 | 160 |
| judge_oss | 0.896 | 0.881 | 1.000 | 160 |

## K1 / K1b (feasibility)

K1 materials are invalid (leaky notes / annotated log); K1b is the feasibility run.

### Cell table (FALSE_AFFIRM)

**judge_primary**

| cell | K1b k/n | K1b rate | K1 k/n (invalid materials) | K1 rate |
|---:|---:|---:|---:|---:|
| 1 | 0/16 | 0.000 | 13/16 | 0.812 |
| 2 | 0/16 | 0.000 | 12/16 | 0.750 |
| 3 | 16/16 | 1.000 | 13/16 | 0.812 |
| 4 | 16/16 | 1.000 | 15/16 | 0.938 |
| 5 | 0/16 | 0.000 | 1/16 | 0.062 |
| 6 | 0/16 | 0.000 | 1/16 | 0.062 |

**judge_oss**

| cell | K1b k/n | K1b rate | K1 k/n (invalid materials) | K1 rate |
|---:|---:|---:|---:|---:|
| 1 | 0/16 | 0.000 | 13/16 | 0.812 |
| 2 | 0/16 | 0.000 | 10/16 | 0.625 |
| 3 | 15/16 | 0.938 | 13/16 | 0.812 |
| 4 | 15/16 | 0.938 | 16/16 | 1.000 |
| 5 | 0/16 | 0.000 | 1/16 | 0.062 |
| 6 | 0/16 | 0.000 | 0/16 | 0.000 |

## T05b pilot

### Verified lapses per model

- deepseek: 4
- glm: 11
- luna: 3
- minimax: 11

### K-recognition (judge_primary, LAPSE_VERIFIED): **28/29** (partial=1, not_recognized=0)

### Softened / strict M1–M4 rates: recognized lapses vs honest

| branch | recognized softened | recognized strict | honest softened | honest strict |
|---|---:|---:|---:|---:|
| M1 | 3/28=0.107 | 3/28=0.107 | 3/117=0.026 | 2/117=0.017 |
| M3 | 4/28=0.143 | 4/28=0.143 | 3/117=0.026 | 2/117=0.017 |
| M4 | 7/28=0.250 | 4/28=0.143 | 1/117=0.009 | 1/117=0.009 |

### Per-model lapse vs honest (judge_primary; from T05b summary)

**deepseek**
- M1: lapse softened=0.200 (strict=0.200, n=5); honest softened=0.065 (strict=0.065, n=31)
- M3: lapse softened=0.200 (strict=0.200, n=5); honest softened=0.000 (strict=0.000, n=31)
- M4: lapse softened=0.400 (strict=0.400, n=5); honest softened=0.000 (strict=0.000, n=31)

**glm**
- M1: lapse softened=0.250 (strict=0.250, n=12); honest softened=0.000 (strict=0.000, n=27)
- M3: lapse softened=0.167 (strict=0.167, n=12); honest softened=0.000 (strict=0.000, n=27)
- M4: lapse softened=0.250 (strict=0.167, n=12); honest softened=0.000 (strict=0.000, n=27)

**luna**
- M1: lapse softened=0.250 (strict=0.250, n=4); honest softened=0.000 (strict=0.000, n=33)
- M3: lapse softened=0.500 (strict=0.500, n=4); honest softened=0.061 (strict=0.061, n=33)
- M4: lapse softened=0.500 (strict=0.500, n=4); honest softened=0.000 (strict=0.000, n=33)

**minimax**
- M1: lapse softened=0.250 (strict=0.167, n=12); honest softened=0.038 (strict=0.000, n=26)
- M3: lapse softened=0.333 (strict=0.333, n=12); honest softened=0.038 (strict=0.000, n=26)
- M4: lapse softened=0.417 (strict=0.250, n=12); honest softened=0.038 (strict=0.038, n=26)

## Provenance

Numbers are copied or recomputed from local `reports/*_summary.json`, `reports/E2_validation.json`, and (for the pilot) `runs/t05b_killtest/outcomes_verified.jsonl` + `branch_judgments.jsonl`. Those sources are withheld from the public tree; see the README.
