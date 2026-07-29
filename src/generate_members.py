"""Synthetic health-plan member population for Next Best Action modeling.

The critical design property: each member has a KNOWN true incremental uplift for
each candidate action, and the uplift is driven by DIFFERENT features than the
baseline response level. That gap is what makes uplift modeling beat propensity
targeting — propensity chases members who respond anyway ("sure things"), while
uplift finds the "persuadables". Members fall into the four uplift archetypes:
persuadable (+), sure-thing (~0, high base), lost-cause (~0, low base), and
sleeping-dog (negative — the action backfires).

Actions modeled (member engagement NBAs):
  care_gap_outreach  : close an open care gap (e.g., overdue screening)
  wellness_program   : enroll in a wellness/rewards program
  telehealth_nudge   : encourage a telehealth visit

A randomized holdout experiment is simulated per action (treatment vs control),
producing the observed binary response used to TRAIN the uplift models. The true
uplift is retained only for grading. All data is synthetic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEED = 21
N = 40000

ACTIONS = {
    # value = business value of an incremental success (relative units)
    "care_gap_outreach": {"value": 5.0},
    "wellness_program":  {"value": 3.0},
    "telehealth_nudge":  {"value": 2.0},
}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate(out_dir: str = "data") -> None:
    rng = np.random.default_rng(SEED)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Member covariates
    age = rng.integers(18, 85, N)
    tenure_years = rng.uniform(0, 15, N)
    prior_engagements = rng.poisson(2.0, N)
    open_care_gaps = rng.poisson(1.1, N)
    chronic_conditions = rng.poisson(0.9, N).clip(0, 6)
    digital_active = rng.integers(0, 2, N)
    risk_score = rng.beta(2, 3, N)

    Xcols = ["age", "tenure_years", "prior_engagements", "open_care_gaps",
             "chronic_conditions", "digital_active", "risk_score"]

    # Baseline response propensity P0 (responds even WITHOUT the action) — driven
    # mostly by prior_engagements + digital_active (the "sure things").
    z0 = (-1.4 + 0.30 * prior_engagements + 0.9 * digital_active
          + 0.02 * (age - 50) / 10 + rng.normal(0, 0.3, N))
    p0 = _sigmoid(z0)

    # True uplift per action — driven by DIFFERENT features (need/eligibility),
    # deliberately weakly correlated with p0 so propensity is a poor proxy.
    tau = {}
    tau["care_gap_outreach"] = np.clip(
        0.02 + 0.06 * open_care_gaps + 0.03 * chronic_conditions
        - 0.10 * digital_active + rng.normal(0, 0.02, N), -0.15, 0.5)   # sleeping-dogs when already digital-active
    tau["wellness_program"] = np.clip(
        0.05 + 0.06 * (risk_score > 0.5) + 0.03 * (age < 45)
        - 0.01 * prior_engagements + rng.normal(0, 0.02, N), -0.12, 0.5)
    tau["telehealth_nudge"] = np.clip(
        0.03 + 0.05 * digital_active + 0.03 * (chronic_conditions >= 2)
        - 0.005 * tenure_years + rng.normal(0, 0.02, N), -0.08, 0.5)

    X = np.column_stack([age, tenure_years, prior_engagements, open_care_gaps,
                         chronic_conditions, digital_active, risk_score])

    rows = []
    for a in ACTIONS:
        # Randomized experiment: 50/50 treat/control for this action's campaign
        treat = rng.integers(0, 2, N)
        p_resp = np.clip(p0 + treat * tau[a], 0.001, 0.999)
        response = (rng.random(N) < p_resp).astype(int)
        for i in range(N):
            rows.append({
                "member_id": f"M{i:06d}", "action": a,
                "treatment": int(treat[i]), "response": int(response[i]),
                "true_uplift": round(float(tau[a][i]), 4),
                **{c: round(float(X[i, j]), 4) for j, c in enumerate(Xcols)},
            })

    with open(out / "experiment.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Member master (one row per member, for the NBA engine / eligibility)
    with open(out / "members.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["member_id"] + Xcols + ["baseline_p0"])
        for i in range(N):
            w.writerow([f"M{i:06d}"] + [round(float(X[i, j]), 4) for j in range(len(Xcols))]
                       + [round(float(p0[i]), 4)])

    (out / "actions.json").write_text(json.dumps(ACTIONS, indent=2))
    print(f"Wrote {N:,} members x {len(ACTIONS)} actions = {len(rows):,} experiment rows to {out}/")


if __name__ == "__main__":
    generate()
