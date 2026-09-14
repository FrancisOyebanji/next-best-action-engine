"""Synthetic customer personas (Digital Twins) with a latent behavioral process.

Each twin is a customer with behavioral-science traits (price sensitivity,
loyalty, risk aversion, channel preference, novelty seeking). A KNOWN latent
utility model turns a marketing scenario (offer) into a real decision
(accept / reject) — this is the ground-truth "real-world outcome" the twin's
simulator is later validated against.

Grounding the traits in real customer-research/survey data (a RAG step over a
research corpus) is the production path; here traits are drawn from documented
behavioral segments so the twins are reproducible and gradeable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEED = 42
N_TWINS = 8000

# Behavioral segments (from marketing/behavioral-science archetypes) with the
# mean trait vector for each. Traits are on a 0-1 scale.
SEGMENTS = {
    "Value Seeker":   {"price_sensitivity": 0.85, "loyalty": 0.35, "risk_aversion": 0.55,
                       "digital_pref": 0.55, "novelty": 0.40, "base_propensity": 0.30},
    "Loyalist":       {"price_sensitivity": 0.35, "loyalty": 0.85, "risk_aversion": 0.60,
                       "digital_pref": 0.45, "novelty": 0.30, "base_propensity": 0.55},
    "Digital Native": {"price_sensitivity": 0.55, "loyalty": 0.45, "risk_aversion": 0.30,
                       "digital_pref": 0.90, "novelty": 0.80, "base_propensity": 0.50},
    "Skeptic":        {"price_sensitivity": 0.60, "loyalty": 0.40, "risk_aversion": 0.85,
                       "digital_pref": 0.40, "novelty": 0.25, "base_propensity": 0.20},
}
TRAITS = ["price_sensitivity", "loyalty", "risk_aversion", "digital_pref", "novelty"]

# The TRUE utility weights that govern how traits + a scenario produce a decision.
# The twin simulator does NOT get these; it must approximate them, and fidelity
# is how well it does. (This is the "real-world" data-generating process.)
TRUE_WEIGHTS = {
    "intercept": -0.6,
    "discount_x_price_sens": 3.2,     # discount appeals most to price-sensitive
    "loyalty_x_incumbent": 1.4,       # loyal customers accept incumbent-brand offers
    "digital_x_digital_pref": 1.1,    # digital channel works on digital-preferrers
    "risk_x_newproduct": -1.6,        # risk-averse reject new/unproven products
    "novelty_x_newproduct": 1.3,
    "base": 1.0,
}


def _logit(x): return 1 / (1 + np.exp(-x))


def true_accept_prob(traits: dict, scenario: dict) -> float:
    """The ground-truth decision process (unknown to the simulator)."""
    w = TRUE_WEIGHTS
    z = (w["intercept"]
         + w["base"] * traits["base_propensity"]
         + w["discount_x_price_sens"] * scenario["discount"] * traits["price_sensitivity"]
         + w["loyalty_x_incumbent"] * scenario["is_incumbent_brand"] * traits["loyalty"]
         + w["digital_x_digital_pref"] * scenario["is_digital_channel"] * traits["digital_pref"]
         + w["risk_x_newproduct"] * scenario["is_new_product"] * traits["risk_aversion"]
         + w["novelty_x_newproduct"] * scenario["is_new_product"] * traits["novelty"])
    return float(_logit(z))


def make_scenarios(rng, n) -> list[dict]:
    return [{
        "discount": round(float(rng.choice([0.0, 0.1, 0.2, 0.3])), 2),
        "is_incumbent_brand": int(rng.random() < 0.5),
        "is_digital_channel": int(rng.random() < 0.5),
        "is_new_product": int(rng.random() < 0.4),
    } for _ in range(n)]


def generate(out_dir: str = "data/twins") -> None:
    rng = np.random.default_rng(SEED)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    twin_rows, decision_rows = [], []
    seg_names = list(SEGMENTS)
    for i in range(N_TWINS):
        seg = rng.choice(seg_names, p=[0.3, 0.3, 0.22, 0.18])
        base = SEGMENTS[seg]
        traits = {t: float(np.clip(base[t] + rng.normal(0, 0.08), 0, 1)) for t in TRAITS}
        traits["base_propensity"] = float(np.clip(base["base_propensity"] + rng.normal(0, 0.05), 0, 1))
        tid = f"T{i:06d}"
        twin_rows.append({"twin_id": tid, "segment": seg,
                          **{t: round(traits[t], 4) for t in TRAITS},
                          "base_propensity": round(traits["base_propensity"], 4)})
        # each twin faces 3 scenarios -> real (ground-truth) decisions
        for sc in make_scenarios(rng, 3):
            p = true_accept_prob(traits, sc)
            decision = int(rng.random() < p)
            decision_rows.append({"twin_id": tid, "segment": seg, **sc,
                                  "real_decision": decision})

    with open(out / "twins.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(twin_rows[0].keys())); w.writeheader(); w.writerows(twin_rows)
    with open(out / "real_decisions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(decision_rows[0].keys())); w.writeheader(); w.writerows(decision_rows)
    (out / "segments.json").write_text(json.dumps(SEGMENTS, indent=2))

    accept_rate = np.mean([d["real_decision"] for d in decision_rows])
    print(f"Wrote {N_TWINS:,} digital twins and {len(decision_rows):,} real decisions "
          f"(overall accept rate {accept_rate:.1%}) to {out}/")


if __name__ == "__main__":
    generate()
