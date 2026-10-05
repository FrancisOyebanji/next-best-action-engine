"""Production scoring + model monitoring over time.

The JD asks to "deploy and maintain models in production, including feature
development, batch/real-time scoring, and monitoring model performance over time."
This module scores the churn model on successive monthly batches and watches for:

  * FEATURE DRIFT  - Population Stability Index (PSI) of each feature vs the
                     training-era baseline. PSI > 0.25 = significant shift.
  * PERFORMANCE DECAY - held-out ROC-AUC per month vs the baseline month; an
                     alert fires when AUC drops more than a tolerance.

A single `batch_score()` provides the scoring step; `monitor()` runs the batches
and returns an alert payload an orchestrator (Airflow) could act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_WARN = 0.10      # minor shift
PSI_ALERT = 0.25     # significant shift -> investigate / retrain
AUC_TOLERANCE = 0.03 # AUC drop beyond this vs baseline triggers an alert


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a baseline and a new sample."""
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=quantiles)[0] / len(expected)
    a = np.histogram(actual, bins=quantiles)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def batch_score(model, df: pd.DataFrame, features) -> np.ndarray:
    """Batch scoring step: churn probability for each partner in the batch."""
    return model.predict_proba(df[features].to_numpy(dtype=float))[:, 1]


def monitor(model, features, baseline: pd.DataFrame,
            scoring_months: dict, drift_features) -> dict:
    """Score each month and compute PSI + AUC decay vs the baseline."""
    base_scores = batch_score(model, baseline, features)
    base_auc = float(roc_auc_score(baseline["churned"], base_scores))

    months = {}
    alerts = []
    for name, df in scoring_months.items():
        scores = batch_score(model, df, features)
        auc = float(roc_auc_score(df["churned"], scores))

        feat_psi = {f: round(psi(baseline[f].to_numpy(dtype=float),
                                 df[f].to_numpy(dtype=float)), 3)
                    for f in drift_features}
        score_psi = round(psi(base_scores, scores), 3)
        drifted = [f for f, v in feat_psi.items() if v >= PSI_ALERT]
        auc_drop = round(base_auc - auc, 3)

        if drifted:
            alerts.append(f"{name}: feature drift (PSI>={PSI_ALERT}) on {', '.join(drifted)}")
        if auc_drop > AUC_TOLERANCE:
            alerts.append(f"{name}: AUC decay {base_auc:.3f} -> {auc:.3f} (drop {auc_drop})")

        months[name] = {
            "n_scored": int(len(df)),
            "mean_churn_score": round(float(scores.mean()), 3),
            "auc": round(auc, 3),
            "auc_drop_vs_baseline": auc_drop,
            "score_psi": score_psi,
            "feature_psi": feat_psi,
            "drifted_features": drifted,
        }

    return {
        "baseline_auc": round(base_auc, 3),
        "psi_alert_threshold": PSI_ALERT,
        "auc_tolerance": AUC_TOLERANCE,
        "months": months,
        "alerts": alerts,
        "healthy": len(alerts) == 0,
    }
