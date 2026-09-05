# NPB V2: official detail challenger

Baseline: `600a6f2e301d51128a4d07689adfdb0dc3e6028e`. The V1 model, original games/features and V1 report remain frozen. MLB/KBO code, workflows and model artifacts are unchanged.

## Verified evaluation, September 5, 2026

Collected all **2,480 completed club games**, with zero remaining parser errors. Development has 1,630 eligible games. Selection chose **core / C=0.01 for WDL** and **core+starter for totals**. V2 passed both promotion comparisons and is selected for the next operating run.

| Same frozen 605 games | Original V1 | V1, corrected inputs | V2 |
|---|---:|---:|---:|
| Overall W/D/L accuracy | 56.5289% | 57.5207% | 57.5207% |
| 55%+ accuracy (games) | 61.1872% (219) | 61.4719% (231) | 61.4719% (231) |
| 60%+ accuracy (games) | 55.5556% (54) | 60.0000% (60) | 60.0000% (60) |
| Total MAE | 3.173580 | 3.160138 | 3.154391 |
| W/D/L log loss | 0.759675 | 0.758166 | 0.758166 |

The WDL improvement comes from restoring broken team histories, **not** from a winning expanded WDL feature group. The unchanged core remains the development winner. The totals starter candidate improves MAE modestly and passes the predeclared non-regression rule; this is not evidence of a large predictive gain.

On the full **727-game canonical 2026 holdout**, V2 records overall **57.4966%**, 55%+ **62.2642% (265)**, 60%+ **59.0164% (61)**, totals MAE **3.169290**. V1 with corrected inputs has the same WDL results and total MAE **3.183975**. Do not compare the 727-game figures directly with the original 605-game benchmark.

Twenty-nine frozen holdout game IDs required exact encoding repair. The local evaluation used scikit-learn 1.9.0, NumPy 2.3.5 and pandas 3.0.1. The saved original V1 report's totals MAE is 3.173337, versus 3.173580 when re-evaluating the saved binary here (a 0.000243 run runtime difference). Original V1 WDL probabilities reproduce its report. Both original artifacts are preserved.

Evaluation ran with `--cached --no-predict`: it did not fetch odds or overwrite today's original predictions. **Manually run Build NPB model once on main after this change** to publish fresh operating predictions and TOP10 with the GitHub odds secret. Subsequent runs remain manual. Full reports are `data/npb/v2_diagnostics.json`, `v2_candidate_report.csv`, `v2_folds.csv`, and `v2_holdout_predictions.csv`.

## Official collection

- Discover completed-game box links from the corresponding row of `https://npb.jp/games/{year}/schedule_{month}_detail.html`. Ignore live scores in page headers.
- Parse official `box.html` batting totals and pitching rows. Verify local date, home/away teams, final scores and final/called-game status. The first pitcher is the actual starter; subsequent pitchers form the bullpen.
- Convert official innings (`6.2`, `6 2/3`, `5+`) to integer outs. Rates use summed outs, never arithmetic on baseball inning decimals.
- Read `https://npb.jp/announcement/starter/` with an explicit page date and observation timestamp. Only announcements observed before the scheduled start enter live inputs. A page already showing tomorrow's starters cannot supply today's starters. Missing identities/stats remain missing.
- HTTPS `npb.jp` only, no guessed box URLs, browser challenge bypasses, private endpoints, or alternate data vendors. Requests are sequential, rate limited, retried for transient statuses, and cached. Parsed records include source and collection timestamp. Failed or ambiguous boxes are reported rather than invented.
- Preserve raw HTML in the Actions cache, not Git. Save parsed records in `data/npb/game_details_v2.json`. A failed schedule fetch cannot replace the previous successful canonical schedule.

### V1 encoding defect found during verification

The original May 2026 schedule was decoded as PTCP154 instead of UTF-8 through `apparent_encoding`. This created garbled team names and split team histories. V2 uses explicit UTF-8 and stores its canonical schedule separately in `games_v2.csv`. The original V1 files are retained for reproducibility.

The frozen holdout is matched to canonical official games by exact team-alias byte reversal, scheduled timestamp and scores; every match must be unique. No fuzzy team matching is used. The primary comparison is on the same frozen 605 games, with a separate full canonical holdout report. V2 must pass against both the frozen V1 predictions and the unchanged V1 model supplied with corrected inputs. All-star games are excluded from V2 team/park statistics.

## Pregame features

Every game on a local calendar date is emitted before any result from that date updates history. In-progress, cancelled and scheduled rows never update performance history, even if scores are present.

- Core V1 Elo, last-20 win/run form and rest; last-5/10/20 form.
- Team batting average from summed official hits / at-bats over the last 5/10/20 team games, with coverage diagnostics.
- Bullpen last-5/10 ERA and WHIP; pitches, outs and appearances over the previous 1/3/7 calendar days. Missing boxes do not become zero workload.
- Starter last-five-start ERA, WHIP, K9, BB9 and HR9; last pitch count, rest days and observed starts.
- Home/away splits, last-ten H2H, and past-only park scoring shrunk toward the historical league mean.

Historical starter identity comes from the actual first pitcher in a completed box. It is a retrospective proxy for the pregame starter, not evidence of when the announcement was published. Current-game pitching statistics never enter that game's predictors. Historical announcement revisions cannot be reconstructed; this limitation is recorded in diagnostics. Live missing starters are not inferred from rotations or fabricated names.

## Model selection and promotion

Preserve a three-class home/draw/away logistic model and a separate gradient-boosted totals model. Eight feature groups are compared: core, form, batting/bullpen, starter, context, form+starter, form+batting/bullpen, all.

Candidate and regularization selection uses only 2024–2025. Fixed validation blocks are July–December 2024, January–June 2025, and July–December 2025, each trained strictly on earlier dates. Imputers/scalers are fitted inside each fold. WDL selects pooled validation log loss over C = 0.01/0.03/0.1/0.3. Totals independently selects validation MAE. Totals residuals come from development out-of-fold predictions, not training fits or the holdout. The final models remain trained on 2024–2025; 2026 is not added to fitting.

Only the selected WDL and totals candidates are evaluated on 2026. Promotion tolerances, fixed before this evaluation:

- At least 200 paired games and 90% detail coverage.
- Overall accuracy drop no more than 0.5 percentage points.
- Log loss increase no more than 0.01; totals MAE increase no more than 0.05 runs.
- At 55% and 60%, accuracy drop no more than 2 percentage points when V1 has at least 30 samples. Challenger must retain at least 30 samples and half the V1 sample count.
- Both the original frozen V1 and V1 with corrected inputs must pass. Otherwise retain V1 and save V2 as a challenger.

Reports include exact input hashes, runtime versions, fold dates, all development candidates, paired predictions and confidence-subset counts. A later manual run extends the 2026 monitoring period; it is not a new untouched test set.

## Recommendations and operation

`npb.predict_today` reads WDL and totals feature lists from the selected bundle. It evaluates every actual bookmaker quote for W/D/L and all quoted total lines, chooses one market per game by raw model win probability, and ranks the best ten without a probability or EV gate. No artificial probability scaling is applied. Draw/push probability stays separate; EV accounts for refunds. Three-way quotes do not treat draws as pushes. Spreads are not assigned invented probabilities: neither existing NPB model models run margin.

Only scheduled games with start times still in the future are eligible. All quoted candidates are saved to `today_candidates.csv`. Today's operating predictions continue to use the existing `today_predictions.csv` / `today_top10.csv` paths and include the model version.

Run `python run_npb_v2_pipeline.py` or manually run **Build NPB model** on main. It collects/caches official details, tests parsers/leakage, selects/trains V2, applies promotion policy, fetches actual odds and saves operating predictions. Use `--cached --no-predict` for reproducible offline evaluation without consuming odds quota or replacing today's recommendation files. `run_npb_pipeline.py` remains the explicit legacy V1 entry point.

The workflow is manual-only. Diagnostics are uploaded even if later stages fail. Parsed official data, model and reports are committed by the workflow; raw pages are kept in the Actions cache.
