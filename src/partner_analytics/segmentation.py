"""Partner segmentation: unsupervised KMeans clustering into actionable tiers.

Business framing (the JD's "customer segmentation models ... robust, interpretable,
and actionable"): group real-estate partners by value + engagement so Partner
Analytics can tailor support, advertising, and product pushes. Clusters are named
by their value/engagement profile and each gets a recommended play.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["transactions_90d", "tool_adoption", "leads_received",
                "response_time_hrs", "ad_spend", "tenure_months"]


def segment_partners(partners: pd.DataFrame, k: int = 4, seed: int = 7) -> dict:
    X = partners[SEG_FEATURES].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)
    df = partners.copy()
    df["cluster"] = labels

    # silhouette on a sample (quality of separation)
    sample = min(2000, len(df))
    idx = np.random.default_rng(seed).choice(len(df), sample, replace=False)
    sil = float(silhouette_score(Xs[idx], labels[idx]))

    # score each cluster by a value index (transactions + adoption) to name tiers
    prof = df.groupby("cluster").agg(
        n=("partner_id", "size"),
        avg_transactions=("transactions_90d", "mean"),
        avg_tool_adoption=("tool_adoption", "mean"),
        avg_leads=("leads_received", "mean"),
        avg_response_hrs=("response_time_hrs", "mean"),
        avg_ad_spend=("ad_spend", "mean"),
        churn_rate=("churned", "mean"),
    )
    value_index = (prof["avg_transactions"].rank() + prof["avg_tool_adoption"].rank()).rank(method="first")
    names = {}
    ordered = value_index.sort_values(ascending=False).index.tolist()
    tier_names = ["Champions", "Growers", "Developing", "Low-Engagement"][:len(ordered)]
    for cl, nm in zip(ordered, tier_names):
        names[cl] = nm

    # Value/engagement tiers from clustering; churn risk comes from the separate
    # supervised model, so plays are keyed to each tier's observed churn too.
    plays = {
        "Champions":      "Protect & upsell: premium advertising inventory, early-access betas.",
        "Growers":        "Accelerate: targeted tool-adoption nudges, lead-gen package upsell.",
        "Developing":     "Educate: onboarding to Follow Up Boss / zPro, response-time coaching.",
        "Low-Engagement": "Re-engage: proactive success outreach; churn-save offers where risk is high.",
    }

    segments = {}
    for cl in prof.index:
        nm = names[cl]
        r = prof.loc[cl]
        segments[nm] = {
            "cluster_id": int(cl),
            "n_partners": int(r["n"]),
            "avg_transactions_90d": round(float(r["avg_transactions"]), 2),
            "avg_tool_adoption": round(float(r["avg_tool_adoption"]), 3),
            "avg_leads": round(float(r["avg_leads"]), 1),
            "avg_response_hrs": round(float(r["avg_response_hrs"]), 2),
            "avg_ad_spend": round(float(r["avg_ad_spend"]), 0),
            "churn_rate": round(float(r["churn_rate"]), 3),
            "recommended_play": plays.get(nm, ""),
        }

    return {
        "k": k,
        "silhouette": round(sil, 3),
        "segments": segments,
        "labeled": df[["partner_id", "cluster"]].assign(
            segment=df["cluster"].map(names)),
    }
