# NPB accuracy experiment — retain V2

Baseline main commit: `266e206f5482c722eccb54df7ad5b12a5f61e103`.

This experiment did **not** improve generalization. The selected challenger is saved for inspection but is **not connected to the operating workflow**. Existing V2 WDL/totals artifacts, prediction files and MLB/KBO behavior are unchanged.

## Experiment

Compared a fixed set of 21 candidates: multinomial logistic models, a separate draw/decisive-winner architecture, and a conservative 25% shallow-tree / 75% core blend. Feature groups were core, pitching/batting/bullpen matchup differences, and matchup plus recent form. Starter rates were shrunk toward **training-only medians** using observed start counts capped at five. No player statistics were invented and no probability inflation was applied.

Candidate selection used expanding-window predictions for 2024 H2 and 2025 H1 (872 validation games). Higher accuracy was eligible only when log loss and Brier score stayed within 0.005 of the core reference. The chosen candidate was then checked on 2025 H2 (450 games), with no reselection after that result. Only that chosen candidate was compared on the 727-game canonical 2026 cohort.

The selected model was matchup logistic regression, C=0.003, with starter shrinkage. Its selection-period accuracy was 54.24% versus the core reference's 51.83%, but the improvement did not persist:

| Period / metric | V2 reference | Selected challenger |
|---|---:|---:|
| 2025 H2 overall, 450 games | 55.33% | 52.44% |
| 2026 overall, 727 games | 57.50% | 54.75% |
| 2026 55%+ accuracy (games) | 62.26% (265) | 58.49% (265) |
| 2026 60%+ accuracy (games) | 59.02% (61) | 59.09% (66) |
| 2026 log loss | 0.761970 | 0.765735 |

The tiny 60%+ difference does not offset the overall and 55%+ regression. Both stage gates required at least a one-percentage-point overall gain plus probability-quality and confidence-coverage checks. The challenger failed. Totals were deliberately held fixed, with local re-evaluated MAE 3.167812 for both models; this experiment makes no totals-improvement claim.

2026 has already been examined in earlier work and is **monitoring data, not a new untouched holdout**. The 2025 H2 period was unused for selection in this experiment, but had been used during V2 development; it is not independent of all prior research. Historical actual starter identities remain retrospective proxies for announcements, as documented in V2. The 727-game figures must not be mixed with the old 605-game benchmark.

## Reproduce and inspect

Run `python run_npb_v3_audit.py` from the application directory. This uses cached pregame features, makes no network/odds calls and does not replace operating files. The entry point imports the model class under its stable module name so its saved joblib can be loaded in another process.

- `data/npb/v3_audit/experiment.json`: fixed experiment specification and input hash.
- `candidate_report.csv`, `folds.csv`, `selection.json`: development selection evidence.
- `report.json`, `monitoring_predictions.csv`: confirmation, monitoring and rejection reasons.
- `models/npb/npb_v3_challenger.joblib`: rejected challenger with unchanged V2 totals model.

Added tests verify target exclusion, training-only shrinkage priors, three-way probability normalization, serialization, overconfidence rejection and the promotion guard. No manual production rerun is needed for this audit.

Before another promotion attempt, the useful next evidence would be additional chronologically earlier official seasons and genuinely prospective, timestamped starter/prediction records. Repeatedly trying alternatives until 2026 improves would overfit the already observed results.
