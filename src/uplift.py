"""Uplift models for Next Best Action targeting.

Uplift (incremental treatment effect) is what NBA needs: not "who will respond?"
but "who responds BECAUSE of the action?". We implement the T-learner (two-model
approach):

    uplift(x) = P(response | treated, x) - P(response | control, x)

trained per action. A propensity model (P(response | treated)) is kept as the
naive baseline the measurement framework shows uplift beating.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

FEATURES = ["age", "tenure_years", "prior_engagements", "open_care_gaps",
            "chronic_conditions", "digital_active", "risk_score"]


class TLearner:
    """Two-model uplift learner: separate response models for treated & control."""

    def __init__(self, seed: int = 0):
        self.m_treat = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)
        self.m_ctrl = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)

    def fit(self, X, treatment, response):
        X, treatment, response = np.asarray(X), np.asarray(treatment), np.asarray(response)
        self.m_treat.fit(X[treatment == 1], response[treatment == 1])
        self.m_ctrl.fit(X[treatment == 0], response[treatment == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X)
        return self.m_treat.predict_proba(X)[:, 1] - self.m_ctrl.predict_proba(X)[:, 1]

    def predict_treated_response(self, X) -> np.ndarray:
        """Propensity-to-respond-if-treated — the naive targeting baseline."""
        return self.m_treat.predict_proba(np.asarray(X))[:, 1]


def frame_to_xy(rows: list[dict]):
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    t = np.array([int(r["treatment"]) for r in rows])
    y = np.array([int(r["response"]) for r in rows])
    tau = np.array([float(r["true_uplift"]) for r in rows])
    return X, t, y, tau
