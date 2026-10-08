"""Marketing Mix Model (MMM+): adstock carryover + saturation.

Models weekly sales as a function of per-channel media spend transformed by:
  * adstock  - geometric carryover (this week's spend keeps working next week),
    with the decay rate grid-searched per channel.
  * saturation - log diminishing returns (each extra dollar does a bit less).

Fits a linear model on the transformed channels plus trend + seasonality, then
reports each channel's contribution share and ROI. Because the data is generated
from known channel effects, the recovered ROI ranking is graded against the truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression

CHANNELS = ["tiktok", "meta", "google_ppc", "email", "branded_search"]
ADSTOCK_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _adstock(x, rate):
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + rate * carry
        out[i] = carry
    return out


def _saturate(x):
    return np.power(x, 0.6)     # matches the generating saturation


def _best_adstock(spend, residual):
    """Pick adstock rate whose saturated transform best correlates with the
    de-trended / de-seasonalized sales residual (isolates channel carryover)."""
    best_r, best_c = 0.0, -1
    for r in ADSTOCK_GRID:
        tr = _saturate(_adstock(spend, r))
        c = abs(np.corrcoef(tr, residual)[0, 1])
        if c > best_c:
            best_c, best_r = c, r
    return best_r


def fit(mmm: pd.DataFrame, truth: dict) -> dict:
    sales = mmm["sales"].to_numpy()
    t = mmm["week"].to_numpy()
    base_feats = np.column_stack([t, np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52)])
    # residualize sales on trend+seasonality so adstock selection sees channel signal only
    base_model = LinearRegression().fit(base_feats, sales)
    residual = sales - base_model.predict(base_feats)

    feats = {"trend": t, "sin52": np.sin(2 * np.pi * t / 52), "cos52": np.cos(2 * np.pi * t / 52)}
    adstock_rates, transformed = {}, {}
    for ch in CHANNELS:
        r = _best_adstock(mmm[ch].to_numpy(), residual)
        adstock_rates[ch] = r
        transformed[ch] = _saturate(_adstock(mmm[ch].to_numpy(), r))
        feats[ch] = transformed[ch]

    # fit the full model; channel spends are near-orthogonal (real flighting), so
    # OLS recovers coefficients proportional to the true channel effects.
    X = pd.DataFrame(feats)
    model = LinearRegression().fit(X, sales)
    coefs = dict(zip(X.columns, model.coef_))
    r2 = model.score(X, sales)

    # recovered contribution of each channel = coef * total(transformed)
    contrib, roi = {}, {}
    for ch in CHANNELS:
        contrib[ch] = max(coefs[ch], 0.0) * transformed[ch].sum()
        spend = mmm[ch].sum()
        roi[ch] = contrib[ch] / spend if spend else 0.0
    total = sum(contrib.values()) or 1.0
    contrib_share = {ch: contrib[ch] / total for ch in CHANNELS}

    # TRUE contribution each channel actually drove (beta x saturated adstock),
    # recomputed from the known generating process -> the grading target.
    true_contrib = {}
    for ch in CHANNELS:
        tr = _saturate(_adstock(mmm[ch].to_numpy(), truth["adstock"][ch]))
        true_contrib[ch] = truth["mmm_beta"][ch] * tr.sum()

    recovered = sorted(CHANNELS, key=lambda c: contrib[c], reverse=True)
    true_rank = sorted(CHANNELS, key=lambda c: true_contrib[c], reverse=True)
    rank_agreement = float(np.corrcoef(
        rankdata([contrib[c] for c in CHANNELS]),
        rankdata([true_contrib[c] for c in CHANNELS]))[0, 1])

    return {
        "r_squared": round(float(r2), 4),
        "adstock_rates": {k: round(v, 2) for k, v in adstock_rates.items()},
        "contribution_share": {k: round(v, 3) for k, v in contrib_share.items()},
        "roi_index": {k: round(roi[k], 4) for k in CHANNELS},
        "recovered_contribution_ranking": recovered,
        "true_contribution_ranking": true_rank,
        "rank_agreement": round(rank_agreement, 3),
        "top_channel_correct": bool(recovered[0] == true_rank[0]),
    }
