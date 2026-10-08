"""Multi-touch attribution (MTA) + a bias diagnostic against true incrementality.

Implements the standard rule-based MTA models over customer touch journeys:
last-touch, first-touch, linear, time-decay, and position-based (U-shaped). Each
distributes conversion credit across the channels in a journey.

The point of the module is the diagnostic: because the data is generated from KNOWN
per-channel incremental weights (upper-funnel channels are most incremental but
rarely the last touch), we can compare each model's credit share to the TRUE
incremental share and quantify how badly last-touch over-credits "closer" channels.
"""
from __future__ import annotations

import numpy as np

CHANNELS = ["tiktok", "meta", "google_ppc", "email", "branded_search"]


def _credit_journey(journey, model):
    n = len(journey)
    if n == 0:
        return {}
    if model == "last_touch":
        w = {journey[-1]: 1.0}
    elif model == "first_touch":
        w = {journey[0]: 1.0}
    elif model == "linear":
        w = {}
        for c in journey:
            w[c] = w.get(c, 0) + 1.0 / n
    elif model == "time_decay":
        half = 2.0
        raw = np.array([0.5 ** ((n - 1 - i) / half) for i in range(n)])
        raw = raw / raw.sum()
        w = {}
        for i, c in enumerate(journey):
            w[c] = w.get(c, 0) + raw[i]
    elif model == "position":   # U-shaped: 40% first, 40% last, 20% middle
        w = {}
        if n == 1:
            w[journey[0]] = 1.0
        else:
            mid = journey[1:-1]
            w[journey[0]] = w.get(journey[0], 0) + 0.4
            w[journey[-1]] = w.get(journey[-1], 0) + 0.4
            for c in mid:
                w[c] = w.get(c, 0) + (0.2 / len(mid) if mid else 0)
    else:
        raise ValueError(model)
    return w


def attribute(journeys, model: str) -> dict:
    credit = {c: 0.0 for c in CHANNELS}
    for j in journeys:
        if not j["converted"]:
            continue
        for ch, w in _credit_journey(j["journey"], model).items():
            credit[ch] += w
    total = sum(credit.values()) or 1.0
    return {c: credit[c] / total for c in CHANNELS}


def evaluate(journeys, truth: dict) -> dict:
    tw = truth["incr_weight"]
    tot = sum(tw.values())
    true_share = {c: tw[c] / tot for c in CHANNELS}

    models = ["last_touch", "first_touch", "linear", "time_decay", "position"]
    shares, mae = {}, {}
    for m in models:
        s = attribute(journeys, m)
        shares[m] = {c: round(s[c], 3) for c in CHANNELS}
        mae[m] = round(float(np.mean([abs(s[c] - true_share[c]) for c in CHANNELS])), 4)

    # the headline diagnostic: last-touch over-credit of the top "closer" channel
    lt = shares["last_touch"]
    closer = max(CHANNELS, key=lambda c: lt[c] - true_share[c])
    over_credit = round(lt[closer] - true_share[closer], 3)
    best_model = min(mae, key=mae.get)

    return {
        "true_incremental_share": {c: round(true_share[c], 3) for c in CHANNELS},
        "model_shares": shares,
        "mae_vs_truth": mae,
        "best_model": best_model,
        "last_touch_worst": bool(mae["last_touch"] == max(mae.values())),
        "most_over_credited_by_last_touch": closer,
        "last_touch_over_credit": over_credit,
    }
