"""Tests for the marketing measurement module (graded vs known ground truth)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp("mm"))
    from marketing_measurement import run_measurement
    return run_measurement.main()


# ---------------- incrementality ----------------
def test_incrementality_recovers_true_lift(result):
    inc = result["incrementality"]
    # at least one estimator lands within 5pp of the true 12% lift
    assert min(inc["did_abs_error"], inc["synthetic_control_abs_error"]) < 0.05
    assert inc["iroas"] > 1.0          # campaign was incremental


# ---------------- MMM ----------------
def test_mmm_recovers_channel_ranking(result):
    mm = result["mmm"]
    assert mm["r_squared"] > 0.6
    assert mm["top_channel_correct"] is True
    assert mm["rank_agreement"] >= 0.5


# ---------------- attribution ----------------
def test_last_touch_is_most_biased(result):
    at = result["attribution"]
    # last-touch should be the worst MTA model vs true incrementality
    assert at["last_touch_worst"] is True
    # and it over-credits a closer channel materially
    assert at["last_touch_over_credit"] > 0.1


def test_better_models_beat_last_touch(result):
    mae = result["attribution"]["mae_vs_truth"]
    assert mae["linear"] < mae["last_touch"]
    assert mae[result["attribution"]["best_model"]] <= min(mae.values()) + 1e-9


# ---------------- reconciliation ----------------
def test_reconciliation_flags_divergence(result):
    rec = result["reconciliation"]
    assert rec["n_divergences"] >= 1
    assert rec["recommendations"]
