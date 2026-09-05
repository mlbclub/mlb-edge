# SPORTS LAB EPL V1

## Scope

- Competition: English Premier League first-team league matches only.
- Excludes cups, friendlies, reserves and continental competitions.
- Operating status: validation only; EPL remains disabled on the public site.
- Model version: `sports-lab-epl-v1`.

## Data

Historical league results are collected from the football-data.co.uk E0 season CSVs for 2020-21 through the current 2026-27 season. Current market odds come from the existing `soccer_epl` Odds API integration.

## Leakage controls

All features are past-only. Matches sharing the same local calendar date are featurized as one block before any same-day results update Elo or recent-form histories. Model candidate selection uses chronological development seasons only.

## 1X2 model

Multiclass logistic regression predicts home / draw / away directly. Candidate selection is based on chronological development log loss using:

- Elo home probability
- recent 5/10 points per game
- recent 5/10 goal difference
- recent scoring/conceding form
- home-only and away-only form
- rest-day differential

Selected candidate: `core`, `C=0.03`.

## Goal model

Separate Poisson regressions estimate home and away expected goals. Their score distributions produce totals probabilities and BTTS probability. V1 does not invent an Asian handicap probability model.

## Validation

Development selection folds:

- train 2020-21–2021-22 → validate 2022-23
- train 2020-21–2022-23 → validate 2023-24
- train 2020-21–2023-24 → validate 2024-25

Confirmation/holdout season: 2025-26, 380 matches.

- overall 1X2 accuracy: **47.89%**
- 45%+ confidence: **53.36%** (283)
- 50%+ confidence: **54.09%** (220)
- 55%+ confidence: **57.58%** (165)
- 60%+ confidence: **61.26%** (111)
- 1X2 log loss: **1.03783**
- total-goals MAE: **1.23982**

2026-27 monitoring currently contains only 20 completed matches, so its figures are not treated as reliable evidence of future performance.

## Current recommendation generation

For each future EPL event the pipeline evaluates all three 1X2 outcomes and the available totals sides. One best market per match is retained by model probability first and EV second. Market probability is no-vig where available. There is no minimum probability, edge or EV gate in the V1 ranking.

## Promotion rule

V1 is not automatically exposed on the website. It remains a research baseline until challenger work and prospective validation are complete.
