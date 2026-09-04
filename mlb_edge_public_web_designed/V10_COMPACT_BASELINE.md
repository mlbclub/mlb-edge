# V10 designated compact baseline

The user explicitly chose the 64.15% development candidate as the next baseline.
V10 therefore uses `strength_C0.01`: six fixed features, median imputation,
standardization, and logistic regression with C=.01. Definitions are shared by
the audit and production training in `mlb_model/compact.py`.

This is a user-designated replacement, not an automatic promotion. The original
gate remains unchanged, its historical failure reasons remain in bundle diagnostics,
and the previous V9 training path remains available as `fit_v9_bundle`.
The earlier `COMPACT_MODEL_AUDIT.md` records the decision before this designation.

Training regenerates three chronological development-fold metrics and fits on
2024–2025 only. `robust_ablation_report.csv`, `ablation_report.csv` and the bundle
record the designated baseline explicitly. These reports now describe that fixed
baseline rather than rerunning the V9 context-family search. Previous comparison
results remain in `data/compact_model_audit/` and Git history.

The model artifact retains all legacy bundle keys. `moneyline_mode=compact_linear`
routes batch and live probabilities directly through the same six-input estimator.
Tree and run-model moneyline weights are zero. The downstream live market/similarity
consensus is retained as a diagnostic but no longer changes the V10 moneyline
probability, so the deployed prediction corresponds to the evaluated candidate.
Legacy bundles retain their previous behavior. Totals and run-line models retain
the V9 core inputs, estimator configuration and chronological residual calibration.
All games continue to receive raw predictions; no UI or recommendation filters changed.

The checked-in artifact was fitted locally on the available 2024–2025 development
data. Its evaluation status is development-only; no new 2026 holdout claim is made.
The existing scheduled/manual pipeline will retrain the same designated model,
write diagnostics and commit its artifact. No manual training run is required to
create the checked-in model. A running application must load the updated code and
artifact before its predictions use V10; external deployment was not verified here.

Validation: 19 tests, including production feature/regularization checks, future
target exclusion, full probability coverage, legacy compatibility, and live/batch
parity even when external consensus disagrees. A serialized real-model smoke test
also checks parity. The observed 544-game development hit rate is 64.15%; prospective
performance remains unverified and is the next evaluation target.
