"""Synthetic B2B real-estate partner (agent) data with KNOWN ground truth.

Everything a Partner Analytics data scientist works with is simulated from a
*known* data-generating process so every downstream model can be graded against
the truth it is supposed to recover:

  * partners    - cross-section of agent partners with engagement/performance
                  features and a next-quarter CHURN label. Churn is a logistic
                  function of a KNOWN, ordered set of drivers (TRUE_CHURN_DRIVERS),
                  so the performance model can be checked on recovering them.
  * fub_study   - OBSERVATIONAL data on adopting the Follow Up Boss CRM. Adoption
                  is confounded (more-engaged partners self-select in), but the
                  true causal effect on transactions is KNOWN (TRUE_FUB_EFFECT),
                  so propensity-score matching can be graded.
  * ab          - a RANDOMIZED experiment on a zPro UI change with a pre-period
                  covariate (for CUPED). True lift is KNOWN (TRUE_AB_LIFT).
  * scoring_months - monthly scoring frames; later months inject feature drift so
                  PSI / AUC-decay monitoring has something real to catch.

No real Zillow data, PII, or partner records are used or represented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = ["West", "Mountain", "Midwest", "Southeast", "Northeast"]

# ---- ground truth the downstream models are graded against -------------------
# Churn drivers in descending order of true importance (|effect| on churn logit).
TRUE_CHURN_DRIVERS = [
    "tool_adoption",       # strongest: partners who adopt the product stack stay
    "transactions_90d",    # closing deals -> value realized -> retention
    "response_time_hrs",   # slow lead response -> churn (positive effect)
    "tenure_months",       # longer-tenured partners churn less
    "support_tickets",     # friction -> churn (positive effect)
]
TRUE_FUB_EFFECT = 2.5      # true causal lift in transactions from adopting FUB CRM
TRUE_AB_LIFT = 0.040       # true lift in lead->tour conversion from the zPro variant


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def generate(seed: int = 7, n_partners: int = 6000) -> dict:
    """Return a dict of DataFrames plus the known-truth constants."""
    rng = np.random.default_rng(seed)

    # ---------------- partner cross-section ----------------------------------
    tenure_months = rng.gamma(2.2, 9.0, n_partners).clip(1, 120).round()
    tool_adoption = rng.beta(2.5, 2.5, n_partners)                  # 0..1 share of stack adopted
    listings_active = rng.poisson(6 + 10 * tool_adoption)           # engaged partners list more
    leads_received = rng.poisson(18 + 0.9 * listings_active)
    response_time_hrs = (rng.gamma(2.0, 2.2, n_partners) * (1.4 - tool_adoption)).clip(0.2, 48)
    tours_scheduled = rng.poisson(np.maximum(0.1, 0.35 * leads_received * (1.2 - response_time_hrs / 48)))
    transactions_90d = rng.poisson(np.maximum(0.05, 0.22 * tours_scheduled)).astype(float)
    ad_spend = (rng.gamma(2.0, 180.0, n_partners) * (0.5 + tool_adoption)).round(2)
    support_tickets = rng.poisson(1.5 + 3.0 * (1 - tool_adoption))
    nps = (rng.normal(30 + 40 * tool_adoption - 0.4 * support_tickets, 15)).clip(-100, 100).round()
    region = rng.choice(REGIONS, n_partners, p=[0.24, 0.12, 0.2, 0.26, 0.18])

    # churn from a KNOWN logit of standardized drivers (signs matter)
    logit = (
        -1.55
        - 1.40 * _standardize(tool_adoption)
        - 0.95 * _standardize(transactions_90d)
        + 0.70 * _standardize(response_time_hrs)
        - 0.55 * _standardize(tenure_months)
        + 0.35 * _standardize(support_tickets)
    )
    churn_prob = _sigmoid(logit + rng.normal(0, 0.35, n_partners))
    churned = (rng.random(n_partners) < churn_prob).astype(int)

    partners = pd.DataFrame({
        "partner_id": np.arange(1, n_partners + 1),
        "region": region,
        "tenure_months": tenure_months,
        "tool_adoption": tool_adoption.round(3),
        "listings_active": listings_active,
        "leads_received": leads_received,
        "response_time_hrs": response_time_hrs.round(2),
        "tours_scheduled": tours_scheduled,
        "transactions_90d": transactions_90d,
        "ad_spend": ad_spend,
        "support_tickets": support_tickets,
        "nps": nps,
        "churned": churned,
    })

    # ---------------- observational Follow Up Boss (FUB) study ---------------
    # Adoption is CONFOUNDED: engaged, higher-tenure partners self-select in.
    adopt_logit = (-0.2 + 1.3 * _standardize(tool_adoption)
                   + 0.6 * _standardize(tenure_months)
                   + 0.5 * _standardize(leads_received.astype(float)))
    p_adopt = _sigmoid(adopt_logit)
    fub_adopted = (rng.random(n_partners) < p_adopt).astype(int)
    # transactions outcome = baseline(confounders) + TRUE causal effect*treatment + noise
    base_tx = (3.0 + 2.2 * tool_adoption + 0.03 * tenure_months
               + 0.04 * leads_received)
    transactions_post = (base_tx + TRUE_FUB_EFFECT * fub_adopted
                         + rng.normal(0, 1.2, n_partners)).clip(0)
    fub_study = pd.DataFrame({
        "partner_id": partners["partner_id"],
        "tool_adoption": tool_adoption.round(3),
        "tenure_months": tenure_months,
        "leads_received": leads_received,
        "listings_active": listings_active,
        "fub_adopted": fub_adopted,
        "transactions_post": transactions_post.round(2),
    })

    # ---------------- randomized zPro A/B experiment -------------------------
    n_ab = n_partners
    variant = rng.integers(0, 2, n_ab)                      # 0=control, 1=treatment (balanced)
    pre_conv = rng.beta(6, 24, n_ab)                        # pre-period conversion (CUPED covariate)
    # post conversion correlated with pre (same partners) + true treatment lift
    post_conv = (0.75 * pre_conv + 0.25 * rng.beta(6, 24, n_ab)
                 + TRUE_AB_LIFT * variant)
    post_conv = post_conv.clip(0, 1)
    ab = pd.DataFrame({
        "partner_id": partners["partner_id"],
        "variant": variant,
        "pre_conversion": pre_conv.round(4),
        "post_conversion": post_conv.round(4),
    })

    # ---------------- monthly scoring frames (for monitoring) ----------------
    # month 0 = training-era distribution; later months drift leads/response up.
    feat_cols = ["tenure_months", "tool_adoption", "listings_active",
                 "leads_received", "response_time_hrs", "tours_scheduled",
                 "transactions_90d", "ad_spend", "support_tickets"]
    scoring_months = {}
    for m, drift in zip([0, 1, 2], [0.0, 0.0, 1.0]):
        df = partners.sample(n=2500, random_state=seed + m).reset_index(drop=True).copy()
        if drift:
            # realistic covariate shift: lead volume jumps, response times worsen
            df["leads_received"] = (df["leads_received"] * 1.6 + 8).round()
            df["response_time_hrs"] = (df["response_time_hrs"] * 1.5).clip(0.2, 48).round(2)
            df["tool_adoption"] = (df["tool_adoption"] * 0.85).round(3)
        scoring_months[f"month_{m}"] = df

    return {
        "partners": partners,
        "fub_study": fub_study,
        "ab": ab,
        "scoring_months": scoring_months,
        "feature_cols": feat_cols,
        "truth": {
            "churn_drivers": TRUE_CHURN_DRIVERS,
            "fub_effect": TRUE_FUB_EFFECT,
            "ab_lift": TRUE_AB_LIFT,
        },
    }


if __name__ == "__main__":
    d = generate()
    p = d["partners"]
    print(f"partners={len(p)}  churn rate={p['churned'].mean():.3f}")
    print(f"FUB adopters={d['fub_study']['fub_adopted'].mean():.3f}  "
          f"A/B treated={d['ab']['variant'].mean():.3f}")
    print("scoring months:", {k: len(v) for k, v in d["scoring_months"].items()})
