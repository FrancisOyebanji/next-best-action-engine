"""Tests: uplift modeling, Qini measurement, and NBA decision rules."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------- measurement ----------
def test_qini_rewards_good_targeting():
    from measurement import qini_coefficient
    rng = np.random.default_rng(0)
    n = 4000
    t = rng.integers(0, 2, n)
    score = rng.random(n)
    # Response driven by score among treated only (score = true uplift proxy)
    y = ((rng.random(n) < 0.1 + 0.6 * score * t)).astype(int)
    good = qini_coefficient(y, t, score)          # targets by true driver
    bad = qini_coefficient(y, t, rng.random(n))   # random targeting
    assert good > bad


def test_ab_incremental_lift_detects_effect():
    from measurement import ab_incremental_lift
    rng = np.random.default_rng(1)
    n = 8000
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2 + 0.1 * t).astype(int)   # +10pp true lift
    r = ab_incremental_lift(y, t)
    assert r["absolute_lift"] > 0.05 and r["significant"]


# ---------- uplift model ----------
def test_uplift_model_recovers_true_uplift_ordering(tmp_path):
    os.chdir(tmp_path)
    import generate_members
    generate_members.generate()
    import csv
    from collections import defaultdict
    from uplift import TLearner, frame_to_xy
    rows = defaultdict(list)
    for r in csv.DictReader(open("data/experiment.csv")):
        rows[r["action"]].append(r)
    X, t, y, tau = frame_to_xy(rows["care_gap_outreach"])
    cut = int(0.7 * len(X))
    m = TLearner(seed=0).fit(X[:cut], t[:cut], y[:cut])
    up = m.predict_uplift(X[cut:])
    # Predicted uplift should correlate positively with the true uplift.
    corr = np.corrcoef(up, tau[cut:])[0, 1]
    assert corr > 0.3


# ---------- NBA engine ----------
def _members():
    return [
        {"member_id": "M1", "open_care_gaps": 2, "age": 40, "digital_active": 1},
        {"member_id": "M2", "open_care_gaps": 0, "age": 75, "digital_active": 0},
    ]


def test_eligibility_rules_applied():
    from nba_engine import eligible
    assert eligible(_members()[0], "care_gap_outreach")          # has a gap
    assert not eligible(_members()[1], "care_gap_outreach")      # no gap
    assert not eligible(_members()[1], "wellness_program")       # age >= 70 excluded


def test_engine_picks_highest_expected_value_and_one_per_member():
    from nba_engine import decide_batch
    members = _members()
    uplift = {
        "care_gap_outreach": {"M1": 0.20, "M2": 0.20},
        "wellness_program": {"M1": 0.30, "M2": 0.30},
        "telehealth_nudge": {"M1": 0.10, "M2": 0.10},
    }
    values = {"care_gap_outreach": 5.0, "wellness_program": 3.0, "telehealth_nudge": 2.0}
    decisions = {d.member_id: d for d in decide_batch(members, uplift, values)}
    # M1: care_gap EV=1.0 vs wellness EV=0.9 vs telehealth EV=0.2 -> care_gap
    assert decisions["M1"].action == "care_gap_outreach"
    # M2 ineligible for care_gap (no gap) and wellness (age>=70); only telehealth,
    # but M2 isn't digital_active -> ineligible there too -> no action
    assert decisions["M2"].action is None


def test_engine_suppresses_negative_uplift():
    from nba_engine import decide_batch
    members = [{"member_id": "M1", "open_care_gaps": 3, "age": 40, "digital_active": 1}]
    uplift = {"care_gap_outreach": {"M1": -0.2}, "wellness_program": {"M1": -0.1},
              "telehealth_nudge": {"M1": -0.05}}
    values = {"care_gap_outreach": 5.0, "wellness_program": 3.0, "telehealth_nudge": 2.0}
    d = decide_batch(members, uplift, values)[0]
    assert d.action is None      # all sleeping-dogs -> suppressed


def test_engine_respects_capacity():
    from nba_engine import decide_batch, summarize
    members = [{"member_id": f"M{i}", "open_care_gaps": 2, "age": 40, "digital_active": 1}
               for i in range(10)]
    uplift = {"care_gap_outreach": {f"M{i}": 0.3 for i in range(10)},
              "wellness_program": {f"M{i}": 0.01 for i in range(10)},
              "telehealth_nudge": {f"M{i}": 0.01 for i in range(10)}}
    values = {"care_gap_outreach": 5.0, "wellness_program": 3.0, "telehealth_nudge": 2.0}
    decisions = decide_batch(members, uplift, values, capacity={"care_gap_outreach": 3,
                                                                "wellness_program": 100,
                                                                "telehealth_nudge": 100})
    by_action = summarize(decisions)["by_action"]
    assert by_action.get("care_gap_outreach", 0) == 3   # capped
