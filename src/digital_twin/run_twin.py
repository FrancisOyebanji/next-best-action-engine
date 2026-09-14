"""Digital Twin pipeline: generate twins -> simulate -> measure fidelity ->
improve via calibration experiment -> re-measure -> what-if scenario.

Demonstrates the core loop the role describes: improve simulation fidelity
through experimentation, validation, and real-world outcome measurement.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from digital_twin import fidelity, personas
from digital_twin.twin_simulator import BehavioralSimulator, build_matrix, scenario_features


def _load():
    twins = {r["twin_id"]: {k: float(v) if k not in ("twin_id", "segment") else v
                            for k, v in r.items()}
             for r in csv.DictReader(open("data/twins/twins.csv"))}
    decisions = list(csv.DictReader(open("data/twins/real_decisions.csv")))
    return twins, decisions


def main() -> dict:
    print("=== 1/5 Generating digital twins + real (ground-truth) decisions ===")
    personas.generate()
    twins, decisions = _load()
    X, y, segs = build_matrix(twins, decisions)

    # split: calibrate on a "training" outcome window, validate on held-out real outcomes
    n = len(y); rng = np.random.default_rng(0); idx = rng.permutation(n)
    cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]

    print("=== 2/5 Baseline twin fidelity (uncalibrated heuristic) ===")
    base_sim = BehavioralSimulator()                       # not calibrated
    base_p = base_sim.accept_prob(X[te])
    base_fid = fidelity.fidelity_report(y[te], base_p, [segs[i] for i in te])
    print(f"   composite fidelity {base_fid['composite_fidelity_score']} "
          f"(accuracy {base_fid['decision_accuracy']}, AUC {base_fid['roc_auc']})")

    print("=== 3/5 Improve via calibration experiment (learn from real outcomes) ===")
    cal_sim = BehavioralSimulator().calibrate(X[tr], y[tr])
    cal_p = cal_sim.accept_prob(X[te])
    cal_fid = fidelity.fidelity_report(y[te], cal_p, [segs[i] for i in te])
    gain = round(cal_fid["composite_fidelity_score"] - base_fid["composite_fidelity_score"], 1)
    print(f"   composite fidelity {cal_fid['composite_fidelity_score']} "
          f"(accuracy {cal_fid['decision_accuracy']}, AUC {cal_fid['roc_auc']}) "
          f"-> +{gain} improvement")

    print("=== 4/5 Per-segment fidelity ===")
    for s, m in cal_fid["per_segment_fidelity"].items():
        print(f"   {s:<15} acc {m['accuracy']}  real {m['real_accept_rate']}  sim {m['sim_accept_rate']}")

    print("=== 5/5 What-if: simulate a 20% discount, digital channel, new product ===")
    scen = {"discount": 0.2, "is_incumbent_brand": 0, "is_digital_channel": 1, "is_new_product": 1}
    Xw = np.array([scenario_features(t, scen) for t in twins.values()])
    pop_accept = float(cal_sim.accept_prob(Xw).mean())
    print(f"   predicted population accept rate: {pop_accept:.1%} "
          f"(synthetic A/B before spending on a real one)")

    result = {
        "n_twins": len(twins), "n_decisions": n,
        "baseline_fidelity": base_fid, "calibrated_fidelity": cal_fid,
        "fidelity_improvement": gain,
        "whatif_scenario": {**scen, "predicted_population_accept_rate": round(pop_accept, 4)},
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/digital_twin.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nResults -> reports/digital_twin.json")
    return result


if __name__ == "__main__":
    main()
