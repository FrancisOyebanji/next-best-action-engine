"""Digital Twin decision simulator.

Given a customer twin (traits) and a marketing scenario (offer), the simulator
predicts the twin's decision (accept probability). The reasoning step is a
SWAPPABLE adapter:

  - BehavioralSimulator : an interpretable utility model whose weights are LEARNED
    by calibrating against observed real decisions (the offline, runnable default).
  - LLMPersonaSimulator : the production path — prompts an LLM to role-play the
    persona and decide, grounded in customer-research context (RAG). Kept behind
    the same .accept_prob() interface; requires ANTHROPIC_API_KEY.

Keeping reasoning behind one interface is what lets us compare a heuristic twin,
a calibrated twin, and an LLM twin on the same fidelity metrics.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from digital_twin.personas import TRAITS


def scenario_features(traits: dict, sc: dict) -> list[float]:
    """Interaction features mirroring the behavioral hypotheses (trait x scenario)."""
    return [
        traits["base_propensity"],
        sc["discount"] * traits["price_sensitivity"],
        sc["is_incumbent_brand"] * traits["loyalty"],
        sc["is_digital_channel"] * traits["digital_pref"],
        sc["is_new_product"] * traits["risk_aversion"],
        sc["is_new_product"] * traits["novelty"],
    ]


class BehavioralSimulator:
    """Utility-model twin: weights calibrated to observed decisions (no peeking at
    the true generating weights — it must learn them from real outcomes)."""
    name = "behavioral_calibrated"

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def calibrate(self, X, y):
        self.model.fit(X, y)
        self.fitted = True
        return self

    def accept_prob(self, X) -> np.ndarray:
        if not self.fitted:
            # uncalibrated prior: a naive constant heuristic (base propensity only)
            return np.clip(np.asarray(X)[:, 0], 0.01, 0.99)
        return self.model.predict_proba(X)[:, 1]


class LLMPersonaSimulator:
    """Production adapter: LLM role-plays the persona and decides. Same interface."""
    name = "llm_persona"

    SYSTEM = ("You are simulating a customer with the given behavioral traits. "
              "Decide whether they accept the offer. Respond with a probability 0-1.")

    def __init__(self, model="claude-sonnet-5"):
        import anthropic  # deferred; offline users never need it
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def accept_prob(self, prompts: list[str]) -> np.ndarray:  # pragma: no cover
        out = []
        for pr in prompts:
            msg = self._client.messages.create(model=self.model, max_tokens=8,
                system=self.SYSTEM, messages=[{"role": "user", "content": pr}])
            try:
                out.append(float(msg.content[0].text.strip()))
            except ValueError:
                out.append(0.5)
        return np.clip(np.array(out), 0, 1)


def build_matrix(twins: dict, decisions: list[dict]):
    """Assemble (features, real_decision) from twins + their scenario decisions."""
    X, y, meta = [], [], []
    for d in decisions:
        t = twins[d["twin_id"]]
        X.append(scenario_features(t, {
            "discount": float(d["discount"]),
            "is_incumbent_brand": int(d["is_incumbent_brand"]),
            "is_digital_channel": int(d["is_digital_channel"]),
            "is_new_product": int(d["is_new_product"])}))
        y.append(int(d["real_decision"]))
        meta.append(d["segment"])
    return np.array(X), np.array(y), meta
