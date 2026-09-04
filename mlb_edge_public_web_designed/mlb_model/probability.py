from __future__ import annotations
import numpy as np
from scipy.stats import poisson


def _clip_prob(p):
    return float(np.clip(float(p), 0.005, 0.995))


def _logit(p):
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-float(x))))


def joint_poisson(home_lambda: float, away_lambda: float, max_runs: int = 22):
    hl = max(0.15, float(home_lambda))
    al = max(0.15, float(away_lambda))
    hs = np.arange(max_runs + 1)
    as_ = np.arange(max_runs + 1)
    hp = poisson.pmf(hs, hl)
    ap = poisson.pmf(as_, al)
    hp[-1] += max(0.0, 1.0 - hp.sum())
    ap[-1] += max(0.0, 1.0 - ap.sum())
    return np.outer(hp, ap)


def market_probabilities(home_lambda: float, away_lambda: float, total_line: float | None = None):
    j = joint_poisson(home_lambda, away_lambda)
    h_idx, a_idx = np.indices(j.shape)
    p_hw_raw = float(j[h_idx > a_idx].sum())
    p_aw_raw = float(j[a_idx > h_idx].sum())
    non_tie = max(1e-12, p_hw_raw + p_aw_raw)
    out = {
        "home_win_run": p_hw_raw / non_tie,
        "away_win_run": p_aw_raw / non_tie,
        "home_minus_1_5": float(j[(h_idx - a_idx) >= 2].sum()),
        "away_minus_1_5": float(j[(a_idx - h_idx) >= 2].sum()),
        "expected_home_runs": float(home_lambda),
        "expected_away_runs": float(away_lambda),
        "expected_total": float(home_lambda + away_lambda),
    }
    if total_line is not None:
        total = h_idx + a_idx
        out["over_prob"] = float(j[total > float(total_line)].sum())
        out["under_prob"] = float(j[total < float(total_line)].sum())
        out["push_prob"] = float(j[total == int(total_line)].sum()) if float(total_line).is_integer() else 0.0
    return out


def blend_moneyline(classifier_home: float, run_home: float, classifier_weight: float = 0.62):
    """Backward-compatible two-model blend used by older saved bundles."""
    w = float(classifier_weight)
    return _clip_prob(w * float(classifier_home) + (1.0 - w) * float(run_home))


def fuse_with_market(model_prob: float, market_prob: float | None, books: int | float | None = None,
                     component_spread: float | None = None) -> tuple[float, float]:
    """Conservatively pool the calibrated model with no-vig bookmaker consensus.

    Market is treated as an independent information source, not as truth. Weight is
    intentionally bounded so SPORTS LAB's own signal remains dominant. Larger model/
    market disagreements receive a little more shrinkage to reduce overconfidence.
    Returns (fused_probability, market_weight).
    """
    pm = _clip_prob(model_prob)
    if market_prob is None or not np.isfinite(float(market_prob)):
        return pm, 0.0
    pk = _clip_prob(market_prob)
    try:
        n_books = max(0.0, float(books or 0.0))
    except Exception:
        n_books = 0.0
    w = 0.08 + min(0.10, 0.015 * n_books)
    disagreement = abs(pm - pk)
    if disagreement >= 0.16:
        w += 0.05
    elif disagreement >= 0.10:
        w += 0.025
    if component_spread is not None and np.isfinite(float(component_spread)) and float(component_spread) <= 0.07:
        w -= 0.02
    w = float(np.clip(w, 0.06, 0.24))
    fused = _sigmoid((1.0 - w) * _logit(pm) + w * _logit(pk))
    return _clip_prob(fused), w


def empirical_total_probabilities(expected_total: float, total_line: float, calibration: dict | None,
                                  k: int = 450) -> dict:
    """Estimate O/U from out-of-sample scoring errors instead of a pure Poisson tail.

    We use calibration games with a similar expected total and translate their observed
    residuals to today's expectation. This preserves MLB scoring over-dispersion and
    fat tails while remaining deterministic.
    """
    if not calibration:
        lam = max(0.4, float(expected_total))
        ts = np.arange(0, 31)
        p = poisson.pmf(ts, lam)
        p[-1] += max(0.0, 1.0 - p.sum())
        over = float(p[ts > float(total_line)].sum())
        under = float(p[ts < float(total_line)].sum())
        push = float(p[ts == int(total_line)].sum()) if float(total_line).is_integer() else 0.0
        return {"over_prob": over, "under_prob": under, "push_prob": push, "total_method": "poisson_fallback"}

    exp_hist = np.asarray(calibration.get("expected", []), dtype=float)
    actual_hist = np.asarray(calibration.get("actual", []), dtype=float)
    good = np.isfinite(exp_hist) & np.isfinite(actual_hist)
    exp_hist, actual_hist = exp_hist[good], actual_hist[good]
    if len(exp_hist) < 80:
        return empirical_total_probabilities(expected_total, total_line, None, k)

    kk = min(max(80, int(k)), len(exp_hist))
    d = np.abs(exp_hist - float(expected_total))
    idx = np.argpartition(d, kk - 1)[:kk]
    residual = actual_hist[idx] - exp_hist[idx]
    sim_total = np.rint(np.clip(float(expected_total) + residual, 0, 35))
    w = 1.0 / (0.65 + d[idx])
    sw = float(w.sum())
    over = float(w[sim_total > float(total_line)].sum() / sw)
    under = float(w[sim_total < float(total_line)].sum() / sw)
    push = float(w[sim_total == int(total_line)].sum() / sw) if float(total_line).is_integer() else 0.0
    return {
        "over_prob": over,
        "under_prob": under,
        "push_prob": push,
        "total_method": "empirical_residual",
        "total_calibration_n": int(kk),
    }
