"""Tests for the Partner Analytics module.

The pipeline is run once on known-ground-truth synthetic data; each test checks a
model recovers the truth it should or that a guardrail behaves correctly.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp("partner"))
    from partner_analytics import run_partner
    return run_partner.main()


# ---------------- segmentation ----------------
def test_segmentation_four_named_tiers(report):
    segs = report["segmentation"]["segments"]
    assert len(segs) == 4
    assert set(segs) == {"Champions", "Growers", "Developing", "Low-Engagement"}
    # Champions out-transact Low-Engagement (value ordering holds)
    assert segs["Champions"]["avg_transactions_90d"] > segs["Low-Engagement"]["avg_transactions_90d"]


# ---------------- performance model ----------------
def test_performance_model_is_skillful(report):
    p = report["performance_model"]
    assert p["auc_gbm"] > 0.75
    assert p["auc_logistic"] > 0.75
    assert p["capture_at_20pct_targeted"] > 0.35   # beats random (0.20)


def test_performance_model_recovers_true_drivers(report):
    p = report["performance_model"]
    assert p["top_drivers"][0] == "tool_adoption"   # true strongest driver
    assert p["driver_rank_agreement"] >= 0.8        # finds the known driver set


# ---------------- experimentation / causal ----------------
def test_ab_cuped_reduces_variance_without_bias(report):
    ab = report["experiments"]["ab_test"]
    assert ab["variance_reduction_pct"] > 0          # CUPED tightens the estimate
    assert ab["cuped_se"] < ab["raw_se"]
    # both estimates stay near the true lift (CUPED is unbiased)
    assert abs(ab["cuped_lift"] - ab["true_lift"]) < 0.01
    assert ab["srm"]["srm_flag"] is False            # randomization is clean


def test_psm_recovers_causal_effect_better_than_naive(report):
    ps = report["experiments"]["causal_psm"]
    assert ps["psm_beats_naive"] is True
    assert abs(ps["psm_att"] - ps["true_effect"]) < 0.5   # within tolerance of truth
    # matching improves covariate balance (tool_adoption SMD shrinks)
    bal = ps["covariate_balance"]["tool_adoption"]
    assert bal["smd_after"] < bal["smd_before"]


# ---------------- production monitoring ----------------
def test_monitoring_flags_drift_and_decay(report):
    mon = report["monitoring"]
    m2 = mon["months"]["month_2"]
    m1 = mon["months"]["month_1"]
    # the drifted month shows PSI alert + AUC drop; the stable month does not
    assert len(m2["drifted_features"]) >= 1
    assert m2["auc_drop_vs_baseline"] > m1["auc_drop_vs_baseline"]
    assert len(mon["alerts"]) >= 1
