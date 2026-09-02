from __future__ import annotations
import numpy as np
from scipy.stats import poisson


def joint_poisson(home_lambda: float, away_lambda: float, max_runs: int = 22):
    hl = max(0.15, float(home_lambda))
    al = max(0.15, float(away_lambda))
    hs = np.arange(max_runs + 1)
    as_ = np.arange(max_runs + 1)
    hp = poisson.pmf(hs, hl)
    ap = poisson.pmf(as_, al)
    # absorb tiny remaining tail into the final bucket so probabilities sum to ~1
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
        if float(total_line).is_integer():
            out["push_prob"] = float(j[total == int(total_line)].sum())
        else:
            out["push_prob"] = 0.0
    return out


def blend_moneyline(classifier_home: float, run_home: float, classifier_weight: float = 0.62):
    w = float(classifier_weight)
    p = w * float(classifier_home) + (1.0 - w) * float(run_home)
    return min(0.995, max(0.005, p))
