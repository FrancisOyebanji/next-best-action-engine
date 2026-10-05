"""Partner performance model: predict next-quarter partner CHURN.

Maps to the JD's "industry performance ... models (regression, tree-based models),
robust, interpretable, actionable." We fit an interpretable logistic baseline and
a gradient-boosted tree, report ROC-AUC plus the business metrics that actually
drive a retention program (lift / capture at a targeting budget), and check that
the model recovers the KNOWN churn drivers from the data-generating process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_FEATURES = ["tenure_months", "tool_adoption", "listings_active",
                  "leads_received", "response_time_hrs", "tours_scheduled",
                  "transactions_90d", "ad_spend", "support_tickets", "nps"]


def _rank_agreement(pred_order, true_order) -> float:
    """Fraction of true top-k drivers that appear in the predicted top-k."""
    k = len(true_order)
    return len(set(pred_order[:k]) & set(true_order)) / k


def train(partners: pd.DataFrame, true_drivers, seed: int = 7) -> dict:
    X = partners[MODEL_FEATURES].to_numpy(dtype=float)
    y = partners["churned"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30,
                                          random_state=seed, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    logit = LogisticRegression(max_iter=1000, C=1.0)
    logit.fit(scaler.transform(Xtr), ytr)
    p_logit = logit.predict_proba(scaler.transform(Xte))[:, 1]

    gbm = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3)
    gbm.fit(Xtr, ytr)
    p_gbm = gbm.predict_proba(Xte)[:, 1]

    auc_logit = float(roc_auc_score(yte, p_logit))
    auc_gbm = float(roc_auc_score(yte, p_gbm))

    # --- interpretability: permutation importance on the GBM --------------
    perm = permutation_importance(gbm, Xte, yte, n_repeats=8,
                                  random_state=seed, scoring="roc_auc")
    imp = (pd.Series(perm.importances_mean, index=MODEL_FEATURES)
           .sort_values(ascending=False))
    driver_order = imp.index.tolist()
    rank_agreement = _rank_agreement(driver_order, true_drivers)

    # --- business metrics: lift & capture at a 20% outreach budget --------
    order = np.argsort(-p_gbm)
    k20 = int(0.20 * len(yte))
    top = order[:k20]
    base_rate = yte.mean()
    captured = yte[top].sum() / yte.sum()
    lift20 = (yte[top].mean() / base_rate) if base_rate else float("nan")
    k5 = int(0.05 * len(yte))
    lift5 = (yte[order[:k5]].mean() / base_rate) if base_rate else float("nan")

    return {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "base_churn_rate": round(float(base_rate), 3),
        "auc_logistic": round(auc_logit, 3),
        "auc_gbm": round(auc_gbm, 3),
        "top_drivers": driver_order[:5],
        "true_drivers": list(true_drivers),
        "driver_rank_agreement": round(rank_agreement, 3),
        "lift_at_5pct": round(float(lift5), 2),
        "lift_at_20pct": round(float(lift20), 2),
        "capture_at_20pct_targeted": round(float(captured), 3),
        "importances": {k: round(float(v), 4) for k, v in imp.items()},
        "_model": gbm, "_features": MODEL_FEATURES,
    }
