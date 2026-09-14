"""Fidelity / alignment evaluation — how well the Digital Twin matches reality.

Measures the alignment between the twin's SIMULATED decisions and the REAL
(ground-truth) outcomes — the "measure alignment between AI predictions and
real-world outcomes and improve through experimentation and validation"
requirement. Metrics:

  - decision accuracy         (simulated vs real at a 0.5 threshold)
  - ROC-AUC                    (ranking fidelity)
  - Brier score / calibration (are predicted probabilities well-calibrated?)
  - accept-rate alignment      (does the twin population reproduce the real rate?)
  - per-segment fidelity       (does each behavioral segment match?)
  - composite fidelity score   (0-100, blends the above)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def calibration_curve(y_true, p, bins=10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        rows.append({"bin_mid": round((edges[b] + edges[b + 1]) / 2, 3),
                     "predicted": round(float(p[m].mean()), 4),
                     "actual": round(float(y_true[m].mean()), 4), "n": int(m.sum())})
    return rows


def fidelity_report(y_true, p, segments=None) -> dict:
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    pred = (p >= 0.5).astype(int)
    accuracy = float((pred == y_true).mean())
    auc = float(roc_auc_score(y_true, p)) if len(set(y_true)) > 1 else float("nan")
    brier = float(brier_score_loss(y_true, p))
    real_rate, sim_rate = float(y_true.mean()), float(p.mean())
    rate_alignment = 1 - abs(real_rate - sim_rate)

    per_segment = {}
    if segments is not None:
        segs = np.asarray(segments)
        for s in sorted(set(segs.tolist())):
            m = segs == s
            per_segment[s] = {
                "accuracy": round(float(((p[m] >= 0.5).astype(int) == y_true[m]).mean()), 4),
                "real_accept_rate": round(float(y_true[m].mean()), 4),
                "sim_accept_rate": round(float(p[m].mean()), 4)}

    # composite fidelity: blend accuracy, ranking, calibration (1-brier), rate match
    composite = 100 * (0.35 * accuracy + 0.30 * (auc if not np.isnan(auc) else 0.5)
                       + 0.20 * (1 - brier) + 0.15 * rate_alignment)
    return {
        "decision_accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "real_accept_rate": round(real_rate, 4),
        "simulated_accept_rate": round(sim_rate, 4),
        "accept_rate_alignment": round(rate_alignment, 4),
        "per_segment_fidelity": per_segment,
        "calibration_curve": calibration_curve(y_true, p),
        "composite_fidelity_score": round(composite, 1),
    }
