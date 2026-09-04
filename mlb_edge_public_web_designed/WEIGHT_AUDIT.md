# Calibration and ensemble-weight audit

This offline experiment explains the V5/V9 difference and evaluates a proposed
hit-rate weight objective. It does not write the production model, alter the UI,
or relax the V9 selection thresholds.

## Reproduce

From this directory, with the project requirements installed:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python audit_moneyline_weights.py
python -m unittest discover -s tests -v
```

The audit writes only `data/weight_audit/` (or `--output-dir`). Input data is the
committed feature file. A development-data fingerprint, package versions and
exact feature lists are recorded in `metadata.json`.

## Design fixed before execution

The core feature list, models, hyperparameters and original three outer
chronological folds remain unchanged. Every experiment discards 2026 before
fitting, weight selection or evaluation. Historical V5-style controls learn
their own weights within each fold; the full-data V5 weights are not reused
inside development folds.

Four controls cross row/day calibration splits with row/day inner weight-learning
splits. This separates both effects and their interaction. Row-based controls
can split the same UTC day and are explanatory only, never promotion candidates.
The day/day/log-loss combination reproduces the existing V9 core baseline.

A fifth strategy uses the same day/day out-of-fold component predictions but
selects weights using the existing robust score on the three latest available
inner blocks. It retains the original 0.05 weight grid and its floating-point
boundary behavior. Candidates must satisfy the unchanged sample gates (three
blocks, 300 games per block, 30 >=60% predictions and 100 >=55% predictions per
block, 150 pooled >=60% predictions), plus the existing log-loss, AUC and overall
accuracy guardrails relative to the log-loss choice. If none qualify, it falls
back to the log-loss weights. The selected weights are then evaluated on the
untouched outer validation block.

The complete strategy still has to pass the original outer promotion criteria
against V9. Fallbacks are reported explicitly; identical output after fallback
does not demonstrate that an adequately supported hit-rate objective is useless.

## Outputs

- `folds.csv`: five strategies × three validation periods, fitted weights,
  fallback reasons, hit counts, accuracy, Wilson lower bounds and other metrics.
- `summary.csv`: pooled results, stability, eligibility and promotion reasons.
- `components.csv`: calibrated linear/tree and run probabilities, plus the
  uncalibrated mean of the same classifier members as an explanatory control;
  includes dispersion, quantiles, confidence counts and accuracy.
- `calibration_parameters.csv`: sigmoid parameters for each calibration member.
- `inner_splits.csv`: inner date boundaries and same-day overlap indicators.
- `inner_weight_grid.csv`: every examined hit-rate weight combination, sample
  eligibility, guardrails and score (inner selection results, not test results).

The V5 data artifact at `c76ae7f` and V9 input at `a5df85f` were compared by
`game_pk`: all 2024–2025 game IDs and all 488 distinct core classifier/run
input columns matched within 1e-10 (including missing values). This rules out
changed core input values as the explanation on this development cohort.

Regression tests cover sparse-data fallback, eligible weight selection, day
boundary isolation and invariance when all 2026 outcomes/feature values are
replaced. Existing training/live-compatibility tests remain in the suite.

## Completed results (2024–2025 only)

The audit used 4,482 development games and evaluated 3,550 nonoverlapping outer
validation games. All V9 fold counts and metrics reproduced the committed
`robust_ablation_report.csv` to within 2e-12. Fourteen regression tests passed.

### Why early 2025 had only ten high-confidence predictions

Holding the outer periods and feature inputs fixed, the >=60% counts for the
2025 first-half fold were:

| Calibration splits | Weight-learning splits | Games >=60% | Accuracy |
|---|---|---:|---:|
| Rows (historical control) | Rows | 35 | 77.14% |
| Rows (historical control) | Days | 20 | 70.00% |
| Days | Rows (historical control) | 11 | 72.73% |
| Days | Days (V9) | 10 | 80.00% |

Both split choices affect the learned system, with a larger count change for
the calibration-split factor in this period. These are end-to-end factor
contrasts including relearned weights, not fixed-weight causal estimates.
The row-based inner boundaries shared a UTC day in 22 of 24 recorded boundaries
(including two calibration variants); day-based boundaries shared none.
Historical controls should not be adopted merely for better observed accuracy.

Under day-based calibration, the tree component produced only two >=60%
predictions out of 1,245 games and had mean confidence 53.17%. V9 put 80% weight
on this component, yielding only ten >=60% ensemble predictions. The linear
component had 173 >=60% predictions at 60.69% accuracy. This identifies a
concentration of weight on compressed tree probabilities, not missing core
columns, as a proximate mechanism.

Removing calibration is not supported by this audit. For the same estimator
members, tree log loss was 0.7243 before calibration versus 0.6873 after it;
linear log loss was 1.0251 versus 0.6889. The raw probabilities were more
confident but less reliable. Raw-member means are diagnostics, not fitted
alternative final models.

### Hit-rate objective versus current log-loss weights

| Outer validation period | V9 >=60% count / accuracy | Experimental >=60% count / accuracy |
|---|---:|---:|
| 2024 second half | 290 / 53.79% | 290 / 53.79% |
| 2025 first half | 10 / 80.00% | 143 / 60.84% |
| 2025 second half | 86 / 62.79% | 266 / 57.52% |
| Pooled | 386 / 56.48% | 699 / 56.65% |

The first fold had no eligible inner weight combination and fell back to V9.
The other two folds had 62 and 79 eligible combinations, respectively.
Experimental linear/tree/run weights changed from 0.35/0.50/0.15 (fallback) to
0.80/0.05/0.15 and then 0.15/0.45/0.40, revealing substantial temporal variation.

| Pooled metric | V9 | Experimental |
|---|---:|---:|
| >=60% Wilson lower bound | 51.49% | 52.95% |
| >=55% count / accuracy | 1,500 / 56.07% | 2,041 / 55.66% |
| Overall accuracy | 53.69% | 53.41% |
| AUC | 0.5336 | 0.5340 |
| Log loss | 0.6907 | 0.6926 |
| Fold >=60% accuracy standard deviation | 10.87 pp | 2.88 pp |
| Robust score | 2.9873 | 3.2044 |

The improved robust score does not override the promotion gates. The experiment
did not improve all three fold lower bounds, regressed in fold accuracy, and
the champion itself failed the second-fold 30-game sample requirement. In
particular, 80% on ten games is not evidence of a stable baseline. The report's
`insufficient_samples` reason also covers champion insufficiency even when the
challenger has adequate samples. This is an inconclusive promotion comparison,
not proof that the challenger is inferior in the population.

**Decision: retain the production V9 bundle and all existing thresholds.** No
2026 experiment, production retraining, UI change or new workflow dispatch was
performed. The committed artifacts are audit outputs only. A subsequent bounded
experiment could test regularized/shrunk weights to limit the observed weight
swings, while retaining day-safe nested validation and the same promotion gates;
that experiment has not been run or adopted here.
