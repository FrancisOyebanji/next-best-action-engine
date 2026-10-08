"""End-to-end marketing measurement run: incrementality + MMM + MTA + reconciliation."""
from __future__ import annotations

import json
from pathlib import Path

from marketing_measurement import (attribution, generate_marketing_data,
                                   incrementality, mmm, reconciliation)


def main() -> dict:
    print("=== Generating eCommerce marketing data (known ground truth) ===")
    d = generate_marketing_data.generate()
    truth = d["truth"]

    print("=== 1/4 Incrementality (geo holdout) ===")
    inc = incrementality.evaluate(d["geo"], truth)
    print(f"   true lift {inc['true_lift_pct']} | DiD {inc['did_lift_pct']} | "
          f"synthetic control {inc['synthetic_control_lift_pct']} | iROAS {inc['iroas']}")

    print("=== 2/4 Marketing Mix Model (adstock + saturation) ===")
    mm = mmm.fit(d["mmm"], truth)
    print(f"   R2 {mm['r_squared']} | recovered {mm['recovered_contribution_ranking']}")
    print(f"   true ranking     {mm['true_contribution_ranking']} (rank agreement {mm['rank_agreement']})")

    print("=== 3/4 Multi-touch attribution (bias vs truth) ===")
    at = attribution.evaluate(d["journeys"], truth)
    print(f"   best model {at['best_model']} | last-touch worst: {at['last_touch_worst']}")
    print(f"   last-touch over-credits {at['most_over_credited_by_last_touch']} "
          f"by +{at['last_touch_over_credit']}")

    print("=== 4/4 Reconciliation (MTA vs MMM) ===")
    rec = reconciliation.reconcile(d["mmm"], d["journeys"], truth)
    print(f"   {rec['n_divergences']} channel divergences flagged")
    for r in rec["recommendations"]:
        print(f"     - {r}")

    result = {"incrementality": inc, "mmm": mm, "attribution": at, "reconciliation": rec}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/marketing_measurement.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nResults -> reports/marketing_measurement.json")
    return result


if __name__ == "__main__":
    main()
