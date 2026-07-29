"""End-to-end NBA pipeline: experiment data -> uplift models -> measurement -> NBA decisions."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import generate_members
from measurement import ab_incremental_lift, evaluate_targeting, uplift_at_k
from nba_engine import decide_batch, summarize
from uplift import FEATURES, TLearner, frame_to_xy


def load_experiment(path="data/experiment.csv"):
    rows = list(csv.DictReader(open(path)))
    by_action = defaultdict(list)
    for r in rows:
        by_action[r["action"]].append(r)
    return by_action


def main() -> None:
    print("=== 1/4 Generating members + randomized experiment ===")
    generate_members.generate()
    by_action = load_experiment()
    actions = json.loads(Path("data/actions.json").read_text())

    print("=== 2/4 Training uplift models (T-learner) per action ===")
    uplift_by_action, measurement = {}, {}
    rng = np.random.default_rng(0)
    for a, rows in by_action.items():
        X, t, y, tau = frame_to_xy(rows)
        # train/holdout split (holdout used for honest Qini + targeting eval)
        idx = rng.permutation(len(rows))
        cut = int(0.7 * len(rows))
        tr, te = idx[:cut], idx[cut:]
        model = TLearner(seed=0).fit(X[tr], t[tr], y[tr])
        up_te = model.predict_uplift(X[te])
        prop_te = model.predict_treated_response(X[te])

        meas = evaluate_targeting(y[te], t[te], up_te, prop_te)
        meas["ab_test"] = ab_incremental_lift(y[te], t[te])
        meas["true_uplift_top20pct_uplift_targeting"] = round(uplift_at_k(tau[te], up_te, 0.2), 4)
        meas["true_uplift_top20pct_propensity_targeting"] = round(uplift_at_k(tau[te], prop_te, 0.2), 4)
        measurement[a] = meas

        # Score ALL members for the NBA engine
        up_all = model.predict_uplift(X)
        mids = [r["member_id"] for r in rows]
        uplift_by_action[a] = dict(zip(mids, up_all.tolist()))
        print(f"   {a:<20} Qini uplift {meas['qini_uplift']} vs propensity {meas['qini_propensity']} "
              f"vs random {meas['qini_random']}")

    print("=== 3/4 Measurement summary ===")
    for a, m in measurement.items():
        ab = m["ab_test"]
        print(f"   {a:<20} A/B lift +{ab['absolute_lift']} ({ab['relative_lift_pct']}%, "
              f"{'sig' if ab['significant'] else 'ns'})")

    print("=== 4/4 NBA decisioning (eligibility + ranking + constraints) ===")
    members = list(csv.DictReader(open("data/members.csv")))
    action_values = {a: v["value"] for a, v in actions.items()}
    capacity = {"care_gap_outreach": 8000, "wellness_program": 6000, "telehealth_nudge": 6000}
    decisions = decide_batch(members, uplift_by_action, action_values, capacity=capacity)
    summ = summarize(decisions)
    print(f"   contacted {summ['contacted']:,}/{summ['members']:,}, "
          f"suppressed {summ['suppressed']:,}, by action: {summ['by_action']}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/nba_results.json").write_text(json.dumps(
        {"measurement": measurement, "nba_summary": summ}, indent=2))
    # sample decisions for inspection
    with open("reports/sample_decisions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["member_id", "action", "expected_value", "reason", "runner_up"])
        for d in decisions[:500]:
            w.writerow([d.member_id, d.action, d.expected_value, d.reason, d.runner_up])
    print("\nResults written to reports/nba_results.json")


if __name__ == "__main__":
    main()
