# Customer Behavior Intelligence — Digital Twins, Uplift Modeling & Decisioning

**An AI platform that models and simulates customer behavior end to end: AI-powered Digital Twins (synthetic personas that simulate customer decisions, validated against real outcomes with a fidelity framework), uplift models that find the "persuadables," a next-best-action decision engine, and an experimentation/measurement layer (Qini, A/B lift). Python + SQL.**

> **In one breath (Digital Twin / customer-simulation focus):** Built AI-powered Digital Twins — synthetic customer personas with behavioral-science traits and a latent decision process — that simulate customer preferences and decisions, then measured the alignment between simulated and real-world outcomes with a fidelity framework (accuracy, ROC-AUC, calibration, per-segment accept-rate match) and improved simulation fidelity +14 points through a calibration experiment. Paired with uplift modeling, an eligibility-constrained next-best-action engine, and a Qini/A-B measurement layer, it spans modeling, simulating, and acting on customer behavior.

---

## Digital Twins — customer behavior simulation (headline for the Digital Twin role)

Synthetic customer personas whose simulated decisions are validated against real outcomes, and improved through experimentation:

```bash
PYTHONPATH=src python -m digital_twin.run_twin          # generate twins -> simulate -> measure fidelity -> improve -> what-if
PYTHONPATH=src python -m digital_twin.build_dashboard   # fidelity dashboard
PYTHONPATH=src python -m pytest tests/test_digital_twin.py -q   # 7 tests incl. per-segment alignment
```

Verified run (8,000 twins, 24,000 decisions):

| Stage | Composite fidelity | Accuracy | ROC-AUC |
|---|---|---|---|
| Uncalibrated heuristic twin | 60.7 | 0.51 | 0.57 |
| After calibration experiment | **75.0** | 0.67 | 0.69 |
| **Fidelity improvement** | **+14.3** | | |

Per-segment alignment (simulated vs real accept rate): Digital Native 74.5% / 72.7%, Loyalist 66.1% / 66.3%, Skeptic 48.6% / 49.1%, Value Seeker 61.5% / 63.3% — the twins reproduce each behavioral segment's real behavior within ~2pp.

- **Behavioral personas** ([personas.py](src/digital_twin/personas.py)) — traits (price sensitivity, loyalty, risk aversion, channel preference, novelty) from marketing/behavioral-science archetypes, driving a known latent decision process (the "real-world outcome").
- **Decision simulator** ([twin_simulator.py](src/digital_twin/twin_simulator.py)) — an interpretable utility twin calibrated to real outcomes, with a **swappable LLM-persona adapter** (role-play + RAG grounding) behind the same interface as the production path.
- **Fidelity / alignment framework** ([fidelity.py](src/digital_twin/fidelity.py)) — measures simulated-vs-real alignment (accuracy, AUC, Brier/calibration, accept-rate match, per-segment fidelity, composite score) and drives improvement through experimentation.
- **What-if simulation** — run the twin population through a proposed offer to predict aggregate response (a *synthetic A/B* before spending on a real one).

This maps directly to the role: build/enhance **Digital Twins that simulate customer behavior and decision-making**, apply **behavioral science**, use **LLM/RAG/agentic + evaluation frameworks**, and **measure alignment between AI predictions and real-world outcomes**, improving fidelity through validation.

---

## Next Best Action — uplift modeling & decisioning (the acting layer)

**An end-to-end Next Best Action system: uplift models that find the "persuadables," a decision engine that applies eligibility rules and constraints to pick each customer's next best action, and a measurement framework (Qini, A/B lift) that proves uplift targeting beats propensity — plus BigQuery targeting pipelines.**

> **In one breath:** Built a Next Best Action decisioning platform spanning uplift modeling (T-learner incremental-effect models per action), an eligibility- and capacity-constrained decision engine that selects each member's single best action by expected value, and an uplift measurement framework (Qini curves, A/B incremental-lift testing) demonstrating that uplift-based targeting captures up to 7x the incremental response of propensity targeting — with BigQuery SQL pipelines for audience sizing, campaign sizing, and performance reporting.

## Why uplift, not propensity

Propensity targeting asks "who will respond?" — and wastes contacts on **sure things** (members who'd respond anyway) while missing **persuadables** (members who respond *because* of the action). Uplift modeling estimates the **incremental** effect per member. This repo grades both against a synthetic ground truth where the answer is known:

| Action | Qini — uplift targeting | Qini — propensity | A/B incremental lift |
|---|---|---|---|
| care_gap_outreach | **119.4** | 17.4 | +9.8 pp (24.9%, sig) |
| wellness_program | **26.5** | 11.8 | +5.6 pp (13.5%, sig) |
| telehealth_nudge | 14.2 | **36.0** | +1.7 pp (ns) |

**The insight in that last row:** for `telehealth_nudge`, propensity *beats* uplift — because its uplift and its baseline response are both driven by the same feature (digital engagement), so propensity is an adequate proxy. For `care_gap_outreach`, uplift is driven by care-gap need while baseline response is driven by prior engagement — **decorrelated**, so uplift modeling is essential and wins 7x. Knowing *when* uplift modeling matters is the senior-level judgment this demonstrates.

## Run it

```bash
pip install -r requirements.txt
python src/run_pipeline.py       # data -> uplift models -> measurement -> NBA decisions
python src/build_dashboard.py    # measurement + allocation dashboard
python -m pytest tests/ -q       # 7 tests: uplift recovery, Qini, eligibility, constraints
```

## The NBA decision engine ([src/nba_engine.py](src/nba_engine.py))

For each member it produces one auditable decision, enforcing the rules a real decisioning platform (e.g. Pega) applies:

1. **Eligibility** — inclusion/exclusion per action (care-gap outreach only if an open gap exists; wellness excludes members 70+; telehealth for digital members).
2. **Expected value ranking** — `uplift x action_value` across eligible actions.
3. **Suppression** — drop negative-uplift actions (**sleeping dogs** the action would annoy) and any below a minimum expected value.
4. **Constraints** — one action per member (contact fatigue) and per-action capacity caps (campaign sizing), solved greedily by descending expected value.

Output: chosen action, expected value, reason, and runner-up per member (`reports/sample_decisions.csv`).

## Measurement framework ([src/measurement.py](src/measurement.py))

- **Qini curve & coefficient** — incremental response captured vs. fraction of members targeted, benchmarked against random.
- **A/B incremental lift** — treatment-vs-control response with a two-proportion z-test.
- **Uplift@k against ground truth** — mean *true* uplift among the top-k% the model selects (only possible with synthetic data; this is how the model is graded).

## Structure

```
src/generate_members.py   40k members, 3 actions, KNOWN per-member uplift, randomized experiment
src/uplift.py             T-learner uplift models + propensity baseline
src/nba_engine.py         eligibility, expected-value ranking, suppression, capacity constraints
src/measurement.py        Qini, A/B lift, uplift@k
src/run_pipeline.py       orchestration
sql/nba_targeting_pipeline.sql   BigQuery: eligibility funnel, campaign sizing (QUALIFY),
                                 contact-fatigue de-dup, post-campaign A/B reporting
tests/test_nba.py         7 tests incl. uplift recovery + all decision rules
```

## Design choices worth noting

- **Graded against known uplift.** Because the true per-member uplift is constructed, `test_uplift_model_recovers_true_uplift_ordering` proves the T-learner recovers it — the model is validated, not just fit.
- **Randomized experiment, honest holdout.** Uplift models train on a randomized treat/control assignment; Qini is measured on a held-out slice.
- **The engine encodes real decisioning rules.** Eligibility, suppression of sleeping-dogs, contact fatigue, and capacity are all enforced and tested — the operational half of the role, not just the modeling half.
- **SQL is the campaign-ops layer.** BigQuery queries size audiences and report campaign performance in-warehouse, mirroring how NBA delivery actually runs.

## Data & compliance

All members, actions, uplifts, and responses are synthetic and seeded. No real member, campaign, or PHI data is used or represented.

---

*Francis Oluwatobi · oluwatobi.ou@gmail.com*
