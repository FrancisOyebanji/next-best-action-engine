"""Measurement framework: uplift evaluation (Qini) + A/B incremental lift.

- Qini curve compares how much INCREMENTAL response a targeting rule captures as
  it contacts more members, vs. random targeting. The Qini coefficient (area
  between the curve and the random diagonal) summarizes targeting quality.
- We compare three targeting strategies on the same holdout: uplift score,
  propensity-to-respond (the naive baseline), and random.
- A/B analysis: treatment-vs-control response lift with a two-proportion z-test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def qini_curve(y, t, score, n_points: int = 50):
    """Qini points: cumulative incremental responses vs fraction targeted.

    At fraction f, target the top-f by score; incremental = (responders among
    treated) - (responders among control) * (n_treated/n_control), scaled.
    """
    order = np.argsort(-score)
    y, t = y[order], t[order]
    n = len(y)
    xs, ys = [0.0], [0.0]
    for f in np.linspace(1 / n_points, 1.0, n_points):
        k = int(f * n)
        yt, tt = y[:k], t[:k]
        n_t, n_c = max(tt.sum(), 1), max((1 - tt).sum(), 1)
        r_t = yt[tt == 1].sum()
        r_c = yt[tt == 0].sum()
        qini = r_t - r_c * (n_t / n_c)     # incremental responders
        xs.append(f); ys.append(float(qini))
    return np.array(xs), np.array(ys)


def qini_coefficient(y, t, score) -> float:
    xs, ys = qini_curve(y, t, score)
    # area between the model curve and the random straight line to the endpoint
    random_line = ys[-1] * xs
    return float(np.trapz(ys - random_line, xs))


def evaluate_targeting(y, t, uplift_score, propensity_score) -> dict:
    rng = np.random.default_rng(0)
    random_score = rng.random(len(y))
    return {
        "qini_uplift": round(qini_coefficient(y, t, uplift_score), 2),
        "qini_propensity": round(qini_coefficient(y, t, propensity_score), 2),
        "qini_random": round(qini_coefficient(y, t, random_score), 2),
    }


def ab_incremental_lift(y, t) -> dict:
    """Two-proportion z-test of treatment vs control response."""
    yt, yc = y[t == 1], y[t == 0]
    p_t, p_c = yt.mean(), yc.mean()
    n_t, n_c = len(yt), len(yc)
    pooled = (yt.sum() + yc.sum()) / (n_t + n_c)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "response_rate_treatment": round(float(p_t), 4),
        "response_rate_control": round(float(p_c), 4),
        "absolute_lift": round(float(p_t - p_c), 4),
        "relative_lift_pct": round(float(100 * (p_t - p_c) / p_c), 1),
        "p_value": float(f"{p_value:.2e}"),
        "significant": bool(p_value < 0.05),
    }


def uplift_at_k(true_uplift, score, k: float = 0.2) -> float:
    """Mean TRUE uplift among the top-k% by predicted score (grades the model
    against ground truth — only possible with synthetic data)."""
    n = max(1, int(len(score) * k))
    idx = np.argsort(-score)[:n]
    return float(true_uplift[idx].mean())
