# Ensemble weight stabilization audit

This offline experiment does not write the production bundle or change the UI.
Run `python audit_stable_weights.py` after generating features. Results are in
`data/stable_weight_audit/`; component predictions permit independent metric checks.

All fitting, calibration, inner weight selection and outer evaluation use 2024–2025
only, with the existing three chronological outer folds and UTC-day-separated inner
splits. The V5 core vocabulary and all existing sample and promotion gates remain
unchanged. No 2026 outcomes were evaluated or used to choose these settings.

Before fitting, candidates were fixed to the historical default prior
(linear .44, tree .36, run .20), 50% shrinkage of the existing grid toward that prior,
and a grid constrained to ±.15 of each prior coordinate. Shrinkage is tested with
log-loss and hit-rate objectives; the capped grid uses the hit-rate objective.
Insufficient inner evidence falls back to the fixed prior. Secondary inner
guardrails compare against the unrestricted minimum-log-loss baseline.
These bounds constrain distance from the prior, not adjacent-fold changes: two
capped estimates can differ by up to .30. Actual maximum changes are reported.

## Results

There are 4,482 development games and 3,550 outer validation predictions per strategy.
The three folds are H2 2024, H1 2025 and H2 2025.

| Strategy | Fold 1 ≥60% count / accuracy | Fold 2 | Fold 3 | Pooled accuracy | Pooled Wilson lower bound | Maximum weight change |
|---|---:|---:|---:|---:|---:|---:|
| V9 | 290 / 53.79% | 10 / 80.00% | 86 / 62.79% | 56.48% | 51.49% | .40 |
| Fixed prior | 339 / 55.16% | 89 / 65.17% | 82 / 63.41% | 58.24% | 53.91% | .00 |
| Log-loss, 50% shrinkage | 267 / 54.31% | 30 / 70.00% | 86 / 62.79% | 57.44% | 52.44% | .15 |
| Hit-rate, 50% shrinkage | 339 / 55.16% | 66 / 68.18% | 311 / 55.31% | 56.42% | 52.77% | .30 |
| Hit-rate, ±.15 cap | 339 / 55.16% | 65 / 67.69% | 209 / 58.85% | 57.75% | 53.80% | .20 |

The fixed prior has the highest pooled hit rate and robust score in this experiment.
Its fold accuracy standard deviation falls from .1087 to .0436. Its overall accuracy
is 53.77%, AUC .5395 and log loss .690225, versus V9 53.69%, .5336 and .690724.
This is exploratory evidence, not an approved replacement.

**No candidate passed promotion; production remains V9 core.** Every candidate meets
the sample eligibility rule itself, but the champion has only 10 high-confidence
games in fold 2, below the required 30. The gate requires adequate evidence for
both models. Candidates also fail the required fold-level lower-bound wins and
fold accuracy regression checks; log-loss shrinkage additionally fails the pooled
lower-bound margin. The champion's 80% fold-2 accuracy is based on a small sample,
so rejection must not be interpreted as proof that the candidates are inferior.

Repeated experiments on the same development years introduce selection optimism.
Do not relax thresholds or pick additional settings based on 2026 results. A future
promotion requires a separately justified evaluation protocol and adequate evidence.
No production pipeline run is required for this analysis-only commit.
