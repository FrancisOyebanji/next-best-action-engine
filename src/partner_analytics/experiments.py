"""Experimentation + causal inference for Partner Analytics.

Two settings a Partner Analytics DS handles (the JD's "design and analyze
experiments and observational studies (A/B testing, causal inference)"):

1. RANDOMIZED A/B test of a zPro UI change.
     - SRM (sample-ratio mismatch) guard: is the 50/50 split actually 50/50?
     - CUPED variance reduction using a pre-period covariate -> tighter CI,
       same unbiased effect, fewer samples needed to detect the lift.

2. OBSERVATIONAL study: does adopting the Follow Up Boss CRM raise transactions?
   Adoption is self-selected (confounded), so the naive treated-minus-control
   difference is biased. We estimate the effect with 1-nearest-neighbor
   propensity-score matching and compare naive vs matched vs the KNOWN truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ----------------------------- randomized A/B --------------------------------
def srm_check(variant: np.ndarray, expected=0.5) -> dict:
    """Chi-square sample-ratio-mismatch test on the allocation."""
    n = len(variant)
    n_t = int(variant.sum())
    n_c = n - n_t
    exp_t = n * expected
    chi2 = (n_t - exp_t) ** 2 / exp_t + (n_c - (n - exp_t)) ** 2 / (n - exp_t)
    p = float(stats.chi2.sf(chi2, df=1))
    return {"n_control": n_c, "n_treatment": n_t,
            "chi2": round(float(chi2), 3), "p_value": round(p, 4),
            "srm_flag": bool(p < 0.001)}   # standard SRM alarm threshold


def ab_test(ab: pd.DataFrame) -> dict:
    """Difference in post-period conversion, raw and CUPED-adjusted."""
    t = ab["variant"].to_numpy()
    y = ab["post_conversion"].to_numpy()
    x = ab["pre_conversion"].to_numpy()

    srm = srm_check(t)

    # raw effect + Welch t-test
    yt, yc = y[t == 1], y[t == 0]
    raw_effect = float(yt.mean() - yc.mean())
    raw_t = stats.ttest_ind(yt, yc, equal_var=False)
    raw_se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))

    # CUPED: y_adj = y - theta*(x - mean(x)), theta = cov(x,y)/var(x)
    theta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    y_adj = y - theta * (x - x.mean())
    at, ac = y_adj[t == 1], y_adj[t == 0]
    cuped_effect = float(at.mean() - ac.mean())
    cuped_se = float(np.sqrt(at.var(ddof=1) / len(at) + ac.var(ddof=1) / len(ac)))
    cuped_t = stats.ttest_ind(at, ac, equal_var=False)
    var_reduction = 1 - (y_adj.var(ddof=1) / y.var(ddof=1))

    return {
        "srm": srm,
        "raw_lift": round(raw_effect, 4),
        "raw_se": round(raw_se, 5),
        "raw_p_value": round(float(raw_t.pvalue), 5),
        "cuped_lift": round(cuped_effect, 4),
        "cuped_se": round(cuped_se, 5),
        "cuped_p_value": round(float(cuped_t.pvalue), 6),
        "theta": round(theta, 3),
        "variance_reduction_pct": round(float(var_reduction) * 100, 1),
        "se_reduction_pct": round((1 - cuped_se / raw_se) * 100, 1),
    }


# -------------------------- observational / PSM ------------------------------
def psm_effect(study: pd.DataFrame,
               confounders=("tool_adoption", "tenure_months",
                            "leads_received", "listings_active"),
               outcome="transactions_post", treat="fub_adopted",
               seed: int = 7) -> dict:
    """Estimate ATT of FUB adoption via 1-NN propensity-score matching."""
    X = study[list(confounders)].to_numpy(dtype=float)
    t = study[treat].to_numpy()
    y = study[outcome].to_numpy()

    Xs = StandardScaler().fit_transform(X)
    ps = LogisticRegression(max_iter=1000).fit(Xs, t).predict_proba(Xs)[:, 1]

    treated = np.where(t == 1)[0]
    control = np.where(t == 0)[0]

    # naive (confounded) difference
    naive = float(y[treated].mean() - y[control].mean())

    # match each treated unit to nearest control on propensity score (ATT)
    nn = NearestNeighbors(n_neighbors=1).fit(ps[control].reshape(-1, 1))
    _, idx = nn.kneighbors(ps[treated].reshape(-1, 1))
    matched_control = control[idx.ravel()]
    att = float(np.mean(y[treated] - y[matched_control]))

    # covariate balance: standardized mean difference before vs after matching
    def smd(col_idx):
        a, b = X[treated, col_idx], X[control, col_idx]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
        return abs(a.mean() - b.mean()) / pooled
    def smd_matched(col_idx):
        a, b = X[treated, col_idx], X[matched_control, col_idx]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
        return abs(a.mean() - b.mean()) / pooled

    balance = {c: {"smd_before": round(smd(i), 3), "smd_after": round(smd_matched(i), 3)}
               for i, c in enumerate(confounders)}

    return {
        "naive_diff": round(naive, 3),
        "psm_att": round(att, 3),
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
        "covariate_balance": balance,
    }


def run(ab: pd.DataFrame, fub_study: pd.DataFrame, truth: dict) -> dict:
    ab_res = ab_test(ab)
    ab_res["true_lift"] = truth["ab_lift"]

    psm = psm_effect(fub_study)
    psm["true_effect"] = truth["fub_effect"]
    psm["naive_bias"] = round(psm["naive_diff"] - truth["fub_effect"], 3)
    psm["psm_bias"] = round(psm["psm_att"] - truth["fub_effect"], 3)
    psm["psm_beats_naive"] = bool(abs(psm["psm_bias"]) < abs(psm["naive_bias"]))

    return {"ab_test": ab_res, "causal_psm": psm}
