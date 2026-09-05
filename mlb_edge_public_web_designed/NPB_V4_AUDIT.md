# Extended-history NPB audit: stronger confidence subset, no overall promotion

Added 1,750 completed club games from official 2022–2023 monthly schedules (1,818 schedule rows including special/postponed games). UTF-8 decoding and existing official parsers were used, with source URLs and collection counts recorded. No earlier player statistics were inferred or fabricated. Final eligible training increased to 3,315 games through 2025; 2026 outcomes are not used in fitting.

The experiment compared 21 long-history specifications: opponent-adjusted attack, defense and run-margin ratings at 30/60/120-game half-lives, three Elo speeds, and regularization strengths. Ratings update only after emitting every game on a local date; offseason strength regresses toward neutral. Fixed initial neutral scoring is a model initialization, not a claimed player/team statistic, and the first 30 team games are excluded from fitting.

Candidate selection used only expanding validation in 2024 H2 and 2025 H1. Accuracy was maximized with log loss/Brier non-regression guards, and the incumbent remained an explicit option. The winner was the 30-game opponent-adjusted rating model, C=1. Its 2025 H2 confirmation passed, but the already-observed 2026 monitoring comparison failed the overall-accuracy promotion gate.

| Metric | V2 | Extended-history candidate |
|---|---:|---:|
| 2024 H2 + 2025 H1 selection accuracy, 872 games | 51.83% | 54.13% |
| 2025 H2 accuracy, 450 games | 55.33% | 57.56% |
| 2025 H2 60%+ accuracy (games) | 44.44% (9) | 67.11% (76) |
| 2026 accuracy, same 727 games | **57.50%** | 55.71% |
| 2026 55%+ accuracy (games) | 62.26% (265) | 60.28% (287) |
| 2026 60%+ accuracy (games) | 59.02% (61) | **65.12% (86)** |
| 2026 log loss | 0.761970 | 0.763460 |

The candidate's 60%+ subset was 56/86 correct. It is a different selected subset from V2's 36/61, not a paired claim that the same predictions improved by six percentage points. Eighty-six games are insufficient to promise a future 65% hit rate. The overall decline prevents promotion; no threshold, displayed probability, TOP10 policy or operating model was changed to hide it.

Fixed 25/50/75% probability mixtures of the selected ratings model and V2 were also checked on **development predictions only**. None beat the standalone rating model on development accuracy, so no mixture was chosen or searched on 2026. See `development_blends.csv`.

## Status and reproduction

- **Operating model remains V2.** MLB/KBO behavior and production workflow are unchanged. No manual production rerun is needed.
- `python run_npb_v4_audit.py --collect` collects the two older seasons and reproduces the audit. Omit `--collect` to reuse the saved official history.
- `data/npb/games_2022_2023.csv` and `historical_collection_report.json` record added official data.
- `data/npb/v4_audit/` contains the experiment specification, date-ordered features, search/confirmation/monitoring results, and per-game probabilities.
- `models/npb/npb_v4_challenger.joblib` preserves the unpromoted WDL model and unchanged V2 totals model. To inspect it, join the rating feature columns in `v4_audit/features.csv` to V2 pregame features by game ID. It is not a drop-in replacement without those rating features.

2025 periods have been used in previous development, and 2026 has repeatedly been observed. These are historical research/monitoring results, not a new untouched evaluation. The next valid claim about the high-confidence specialist requires future timestamped predictions; repeated retrospective searches must not be presented as independent validation.
