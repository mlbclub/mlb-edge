# Compact model improvement experiment

`python audit_compact_models.py` evaluates six predeclared candidates: three nested
feature sets, each with logistic-regression regularization C=.01 or .1. All use
training-only median imputation and standardization. No feature selection, fitting
or parameter tuning uses 2026. Every outer validation game receives a probability.
The baseline is replayed from the preceding stable-weight audit, aligned by game ID
and checked against actual outcomes. Regenerate that audit if input features change.

The smallest model uses just six inputs: Elo home-win probability and home-minus-away
20-game win rate, run differential, batting OPS, bullpen ERA and bullpen WHIP.
The other candidates add six starter inputs, then three rest/workload inputs.
This tests whether many correlated short-window inputs and a complex ensemble
are reducing generalization. It does not establish that any individual signal is causal.

## Development results (2024–2025 only)

| Model | ≥60% games | Accuracy in that group | Wilson 95% lower bound | Overall accuracy | AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Current V9 | 386 | 56.48% | 51.49% | 53.69% | .5336 | .690724 |
| Six inputs, C=.01 | 544 | **64.15%** | **60.04%** | 54.73% | .5645 | .684577 |
| Six inputs, C=.1 | 770 | 62.34% | 58.86% | **55.55%** | **.5695** | **.683854** |
| Add starters, C=.01 | 800 | 61.50% | 58.08% | 54.96% | .5627 | .685753 |
| Add starters, C=.1 | 1,050 | 60.57% | 57.58% | 54.54% | .5650 | .686610 |
| Add rest/workload, C=.01 | 831 | 60.65% | 57.29% | 54.70% | .5616 | .686450 |
| Add rest/workload, C=.1 | 1,080 | 59.91% | 56.95% | 54.45% | .5638 | .687481 |

Six-input C=.01 high-confidence results are 218 games / 61.93% in H2 2024,
178 / 66.29% in H1 2025 and 148 / 64.86% in H2 2025. Its pooled gain against
V9 is 7.68 percentage points, with 158 additional high-confidence games.
The ≥55% group contains 1,833 games at 57.66%, versus V9 1,500 at 56.07%.
Overall validation coverage is 3,550 games for every candidate.

## Decision and limitations

This is a meaningful development improvement, not a verified future hit rate.
The six-input C=.01 candidate has the highest pre-existing robust score (3.6143),
but **none passes the unchanged production gate**. Rejection reasons are
`insufficient_samples;fold_60_regression`: the champion has only 10 ≥60% games
in H1 2025 and 8 correct, whereas this candidate has 118 correct out of 178.
The candidate passes the pooled lower-bound, fold lower-bound wins, robust-score
margin and secondary metric checks. The gate's raw fold accuracy comparison
still rejects it. No threshold was relaxed and no production bundle was replaced.

Several experiments have already reused these development years. Wilson intervals
do not adjust for model selection or dependence between games. Treat the reported
winner as exploratory; an untouched prospective period is needed before claiming
the gain generalizes. Existing 2026 results were not consulted in this experiment.

Outputs include all candidate predictions, fold metrics, standardized coefficients,
policy, exact input columns and development-data hash. The regression test perturbs
2026 outcomes and features, verifies identical reports and checks that fitting
precedes validation. UI and production pipeline are unchanged; no manual workflow
execution is needed for this analysis-only change.
