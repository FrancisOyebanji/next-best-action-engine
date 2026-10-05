"""End-to-end Partner Analytics pipeline.

Problem framing -> model development -> validation -> (simulated) productionization,
mirroring how a Partner Analytics DS would own a project end to end:

  1. segment the partner portfolio (clustering)
  2. model partner churn (tree-based + interpretable baseline)
  3. measure product changes (randomized A/B with CUPED) and a non-randomized
     feature rollout (propensity-score matching)
  4. put the churn model into batch scoring with drift / performance monitoring
"""
from __future__ import annotations

import json
from pathlib import Path

from partner_analytics import (experiments, generate_partner_data, monitoring,
                               performance_model, segmentation)

DRIFT_WATCH = ["leads_received", "response_time_hrs", "tool_adoption",
               "transactions_90d", "tenure_months"]


def main() -> dict:
    print("=== 1/5 Generating partner data (RWD-shaped, known ground truth) ===")
    d = generate_partner_data.generate()
    partners, truth = d["partners"], d["truth"]
    print(f"   partners={len(partners)} | churn rate {partners['churned'].mean():.3f}")

    print("=== 2/5 Partner segmentation (KMeans) ===")
    seg = segmentation.segment_partners(partners)
    print(f"   silhouette {seg['silhouette']} | tiers: "
          f"{ {k: v['n_partners'] for k, v in seg['segments'].items()} }")

    print("=== 3/5 Partner performance model (churn) ===")
    perf = performance_model.train(partners, truth["churn_drivers"])
    print(f"   AUC logit {perf['auc_logistic']} | GBM {perf['auc_gbm']} | "
          f"capture@20% {perf['capture_at_20pct_targeted']}")
    print(f"   recovered drivers {perf['top_drivers']}")
    print(f"   true drivers      {perf['true_drivers']} "
          f"(rank agreement {perf['driver_rank_agreement']})")

    print("=== 4/5 Experimentation + causal inference ===")
    exp = experiments.run(d["ab"], d["fub_study"], truth)
    ab, ps = exp["ab_test"], exp["causal_psm"]
    print(f"   A/B raw lift {ab['raw_lift']} (SE {ab['raw_se']}) -> "
          f"CUPED {ab['cuped_lift']} (SE {ab['cuped_se']}, -{ab['se_reduction_pct']}% SE) | true {ab['true_lift']}")
    print(f"   PSM: naive {ps['naive_diff']} -> ATT {ps['psm_att']} | true {ps['true_effect']} "
          f"(PSM beats naive: {ps['psm_beats_naive']})")

    print("=== 5/5 Production scoring + monitoring ===")
    mon = monitoring.monitor(perf["_model"], perf["_features"],
                             baseline=d["scoring_months"]["month_0"],
                             scoring_months={k: v for k, v in d["scoring_months"].items()
                                             if k != "month_0"},
                             drift_features=DRIFT_WATCH)
    print(f"   baseline AUC {mon['baseline_auc']} | alerts: {len(mon['alerts'])}")
    for a in mon["alerts"]:
        print(f"     ! {a}")

    result = {
        "segmentation": {"silhouette": seg["silhouette"], "segments": seg["segments"]},
        "performance_model": {k: v for k, v in perf.items() if not k.startswith("_")},
        "experiments": exp,
        "monitoring": mon,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/partner_analytics.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nResults -> reports/partner_analytics.json")
    return result


if __name__ == "__main__":
    main()
