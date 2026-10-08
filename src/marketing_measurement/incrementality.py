"""Incrementality via a geo holdout experiment.

Two estimators of the incremental lift from the treated geos' media push, each
graded against the known true lift:

  * difference-in-differences (DiD) - (treated post - treated pre) minus
    (control post - control pre), the standard geo-experiment estimator.
  * synthetic control - predict each treated geo's counterfactual post sales from
    the control geos' post sales using pre-period weights (regression), then the
    incremental = actual - counterfactual.

Also reports incremental revenue and iROAS (incremental revenue / incremental
spend), the number a customer actually acts on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def did_estimate(geo: pd.DataFrame) -> dict:
    g = geo.groupby(["is_treated", "period"])["sales"].mean().unstack()
    treated_delta = g.loc[1, "post"] - g.loc[1, "pre"]
    control_delta = g.loc[0, "post"] - g.loc[0, "pre"]
    incr_per_geo_week = treated_delta - control_delta
    # lift % relative to treated counterfactual (treated pre + control change)
    counterfactual = g.loc[1, "pre"] + control_delta
    lift_pct = incr_per_geo_week / counterfactual
    return {"lift_pct": float(lift_pct),
            "incremental_per_geo_week": float(incr_per_geo_week),
            "counterfactual_mean": float(counterfactual)}


def synthetic_control(geo: pd.DataFrame) -> dict:
    """Fit treated-total post sales from control geos using pre-period relationship."""
    pivot = geo.pivot_table(index="week", columns="geo", values="sales")
    treated_geos = geo[geo.is_treated == 1].geo.unique()
    control_geos = geo[geo.is_treated == 0].geo.unique()
    pre_weeks = geo[geo.period == "pre"].week.unique()
    post_weeks = geo[geo.period == "post"].week.unique()

    treated_total = pivot[treated_geos].sum(axis=1)
    X = pivot[control_geos]
    model = LinearRegression()
    model.fit(X.loc[pre_weeks], treated_total.loc[pre_weeks])
    pred_post = model.predict(X.loc[post_weeks])
    actual_post = treated_total.loc[post_weeks].to_numpy()
    incr_total = float((actual_post - pred_post).sum())
    lift_pct = incr_total / float(pred_post.sum())
    return {"lift_pct": lift_pct, "incremental_total": incr_total,
            "counterfactual_total": float(pred_post.sum())}


def evaluate(geo: pd.DataFrame, truth: dict) -> dict:
    did = did_estimate(geo)
    sc = synthetic_control(geo)
    true_lift = truth["geo_lift"]

    # incremental revenue + iROAS from the synthetic-control total
    incr_rev = sc["incremental_total"]
    total_spend = geo["spend"].sum()
    iroas = incr_rev / total_spend if total_spend else float("nan")

    return {
        "true_lift_pct": true_lift,
        "did_lift_pct": round(did["lift_pct"], 4),
        "synthetic_control_lift_pct": round(sc["lift_pct"], 4),
        "did_abs_error": round(abs(did["lift_pct"] - true_lift), 4),
        "synthetic_control_abs_error": round(abs(sc["lift_pct"] - true_lift), 4),
        "incremental_revenue": round(incr_rev, 0),
        "incremental_spend": round(float(total_spend), 0),
        "iroas": round(iroas, 2),
    }
