"""Synthetic eCommerce marketing data with KNOWN ground truth for each pillar.

  * geo       - a geo holdout experiment: treated vs control DMAs across a pre and
                post period, with a KNOWN incremental lift applied to treated geos
                in the post period (so incrementality estimates can be graded).
  * mmm       - weekly national spend by channel with a KNOWN adstock + saturation
                process generating sales (so recovered channel ROI can be graded).
  * journeys  - customer touch sequences + conversions from KNOWN per-channel
                incremental weights, with upper-funnel channels appearing early and
                "closer" channels late (so multi-touch attribution bias can be shown).

No real customer, brand, or ad-platform data is used or represented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["tiktok", "meta", "google_ppc", "email", "branded_search"]

# ---- ground truth ----
# MMM: true response coefficient and adstock (carryover) per channel.
TRUE_MMM_BETA = {"tiktok": 3.2, "meta": 4.0, "google_ppc": 2.4, "email": 1.5, "branded_search": 0.8}
TRUE_ADSTOCK = {"tiktok": 0.6, "meta": 0.5, "google_ppc": 0.3, "email": 0.2, "branded_search": 0.1}
# MTA: true incremental weight per channel (upper-funnel channels are most incremental).
TRUE_INCR_WEIGHT = {"tiktok": 0.32, "meta": 0.26, "google_ppc": 0.20, "email": 0.13, "branded_search": 0.09}
# geo experiment: true incremental lift on treated geos in the post period.
TRUE_GEO_LIFT = 0.12          # +12% incremental sales
GEO_SPEND_PER_WEEK = 550.0    # incremental media spend per treated geo-week (for iROAS)


def _adstock(x, rate):
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + rate * carry
        out[i] = carry
    return out


def _saturate(x):
    # power/root saturation (diminishing returns) that keeps dynamic range so the
    # channel effects remain identifiable by the MMM (log1p compresses too hard).
    return np.power(x, 0.6)


def generate(seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "geo": _gen_geo(rng),
        "mmm": _gen_mmm(rng),
        "journeys": _gen_journeys(rng),
        "truth": {
            "mmm_beta": TRUE_MMM_BETA, "adstock": TRUE_ADSTOCK,
            "incr_weight": TRUE_INCR_WEIGHT, "geo_lift": TRUE_GEO_LIFT,
            "geo_spend_per_week": GEO_SPEND_PER_WEEK,
        },
        "channels": CHANNELS,
    }


def _gen_geo(rng, n_geo=60, pre=12, post=12):
    rows = []
    treated_geos = set(rng.choice(n_geo, n_geo // 2, replace=False).tolist())
    for g in range(n_geo):
        base = rng.uniform(8000, 20000)               # geo baseline weekly sales
        trend = rng.uniform(-40, 60)
        is_t = g in treated_geos
        for w in range(pre + post):
            season = 1.0 + 0.08 * np.sin(2 * np.pi * w / 26)
            sales = (base + trend * w) * season + rng.normal(0, base * 0.05)
            if is_t and w >= pre:
                sales *= (1 + TRUE_GEO_LIFT)          # known causal lift
            rows.append((f"DMA{g:02d}", w, "post" if w >= pre else "pre",
                         int(is_t), max(0.0, sales),
                         GEO_SPEND_PER_WEEK if (is_t and w >= pre) else 0.0))
    return pd.DataFrame(rows, columns=["geo", "week", "period", "is_treated", "sales", "spend"])


def _gen_mmm(rng, weeks=104):
    t = np.arange(weeks)
    # independent per-channel spend with real flighting variation (no shared
    # seasonal multiplier) so the channel effects are identifiable by the MMM.
    spend = {}
    for ch in CHANNELS:
        level = rng.uniform(4000, 12000)
        s = rng.normal(level, level * 0.45, weeks)
        flight = rng.random(weeks) < 0.15          # occasional dark weeks
        s[flight] *= 0.2
        spend[ch] = np.maximum(0, s)
    base = 50000.0
    trend = 120.0 * t
    season = 8000.0 * np.sin(2 * np.pi * t / 52)
    # per-channel contribution preserves the TRUE relative betas (no per-channel
    # normalization); one global scale makes media ~40% of sales.
    raw = {ch: TRUE_MMM_BETA[ch] * _saturate(_adstock(spend[ch], TRUE_ADSTOCK[ch]))
           for ch in CHANNELS}
    total_raw = sum(raw.values())
    scale = 22000.0 / total_raw.mean()
    sales = base + trend + season + scale * total_raw + rng.normal(0, 2000, weeks)
    df = pd.DataFrame({"week": t, "sales": np.maximum(0, sales)})
    for ch in CHANNELS:
        df[ch] = spend[ch].round(0)
    return df


def _gen_journeys(rng, n=9000):
    # position bias: how likely each channel is to appear early (0) vs late (1)
    position = {"tiktok": 0.15, "meta": 0.25, "google_ppc": 0.5, "email": 0.8, "branded_search": 0.9}
    rows = []
    for _ in range(n):
        k = rng.integers(1, 6)
        touches = list(rng.choice(CHANNELS, k, replace=True,
                                  p=[0.28, 0.24, 0.22, 0.14, 0.12]))
        # order by position bias (+noise) so closers land last, introducers first
        touches.sort(key=lambda c: position[c] + rng.normal(0, 0.15))
        uniq = set(touches)
        logit = -1.3 + sum(TRUE_INCR_WEIGHT[c] * 3.0 for c in uniq)
        converted = rng.random() < 1 / (1 + np.exp(-logit))
        rows.append({"journey": touches, "converted": int(converted)})
    return rows


if __name__ == "__main__":
    d = generate()
    print("geo rows:", len(d["geo"]), "| treated geos:",
          d["geo"][d["geo"].is_treated == 1].geo.nunique())
    print("mmm weeks:", len(d["mmm"]), "| channels:", d["channels"])
    jr = d["journeys"]
    print("journeys:", len(jr), "| conversion rate:",
          round(np.mean([j["converted"] for j in jr]), 3))
