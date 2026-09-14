"""Tests for the Digital Twin customer-behavior simulation module."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp("twin"))
    from digital_twin import run_twin
    return run_twin.main()


# ---------- behavioral ground-truth process ----------
def test_true_process_responds_to_scenario():
    from digital_twin.personas import true_accept_prob
    price_sensitive = {"price_sensitivity": 0.9, "loyalty": 0.3, "risk_aversion": 0.5,
                       "digital_pref": 0.5, "novelty": 0.4, "base_propensity": 0.4}
    no_discount = true_accept_prob(price_sensitive, {"discount": 0.0, "is_incumbent_brand": 0,
                                                     "is_digital_channel": 0, "is_new_product": 0})
    big_discount = true_accept_prob(price_sensitive, {"discount": 0.3, "is_incumbent_brand": 0,
                                                      "is_digital_channel": 0, "is_new_product": 0})
    assert big_discount > no_discount + 0.1     # discount lifts a price-sensitive twin


# ---------- fidelity improves through experimentation ----------
def test_calibration_improves_fidelity(report):
    base = report["baseline_fidelity"]["composite_fidelity_score"]
    cal = report["calibrated_fidelity"]["composite_fidelity_score"]
    assert cal > base                            # experimentation raised fidelity
    assert report["fidelity_improvement"] > 3


def test_calibrated_twin_ranks_and_is_accurate(report):
    cal = report["calibrated_fidelity"]
    assert cal["roc_auc"] > 0.65
    assert cal["decision_accuracy"] > 0.6


# ---------- alignment to real outcomes per segment ----------
def test_segment_accept_rates_align_with_reality(report):
    for s, m in report["calibrated_fidelity"]["per_segment_fidelity"].items():
        # simulated accept rate within 5pp of the real accept rate for every segment
        assert abs(m["sim_accept_rate"] - m["real_accept_rate"]) < 0.05


def test_population_accept_rate_alignment(report):
    cal = report["calibrated_fidelity"]
    assert abs(cal["simulated_accept_rate"] - cal["real_accept_rate"]) < 0.03


# ---------- what-if simulation ----------
def test_whatif_returns_population_rate(report):
    w = report["whatif_scenario"]
    assert 0 <= w["predicted_population_accept_rate"] <= 1


# ---------- fidelity metric math ----------
def test_fidelity_report_bounds():
    from digital_twin.fidelity import fidelity_report
    y = np.array([0, 1, 0, 1, 1, 0])
    perfect = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    r = fidelity_report(y, perfect)
    assert r["decision_accuracy"] == 1.0
    assert 0 <= r["composite_fidelity_score"] <= 100
