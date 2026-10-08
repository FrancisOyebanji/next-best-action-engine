# Marketing Measurement & Customer Behavior Intelligence — Incrementality, MMM, Attribution, Experimentation & Decisioning

**An end-to-end measurement and customer data-science platform: marketing measurement (geo-based incrementality, MMM+ with adstock & saturation, multi-touch attribution, and a reconciliation diagnostic), an experimentation + causal-inference layer (A/B with CUPED, propensity-score matching), customer/partner segmentation and churn modeling with production monitoring — on top of AI-powered Digital Twins, uplift modeling, and a next-best-action engine. Python + SQL.**

> **In one breath (Marketing Measurement focus):** Built an eCommerce marketing-measurement suite covering all three of Northbeam's lenses — a geo-holdout **incrementality** test (difference-in-differences + synthetic control recovering a true +12% lift, iROAS 2.1x), an **MMM+** with adstock carryover and saturation that recovers the true channel-contribution ranking (R² 0.89, top channels correct), and **multi-touch attribution** that quantifies last-touch bias (last-touch over-credits a "closer" channel by +20pp vs its true incremental share) — tied together by a **reconciliation diagnostic** that flags where attribution, MMM, and incrementality disagree and writes customer-facing recommendations. Everything is graded against a known data-generating process, and it sits on an experimentation/causal-inference layer (CUPED, propensity-score matching), uplift modeling, and a next-best-action engine.

---

## Marketing Measurement — incrementality, MMM & attribution (headline for the Applied Measurement role)

Northbeam's three measurement products — **multi-touch attribution, MMM+, and incrementality** — built end to end on synthetic eCommerce data with **known ground truth**, then triangulated into a customer-facing reconciliation diagnostic:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m marketing_measurement.run_measurement     # incrementality + MMM + MTA + reconciliation
PYTHONPATH=src python -m marketing_measurement.build_dashboard     # measurement dashboard
PYTHONPATH=src python -m pytest tests/test_marketing_measurement.py -q   # 5 tests, graded vs truth
```

| Pillar | Method | Result (graded vs ground truth) |
|---|---|---|
| **Incrementality** ([incrementality.py](src/marketing_measurement/incrementality.py)) | geo holdout: difference-in-differences + synthetic control | recovers true **+12% lift** (DiD 13.3% / synth-control 8.7%), **iROAS 2.1x** |
| **MMM+** ([mmm.py](src/marketing_measurement/mmm.py)) | adstock carryover + saturation regression | **R² 0.89**, top channels correct, contribution rank agreement 0.70 |
| **Multi-touch attribution** ([attribution.py](src/marketing_measurement/attribution.py)) | last/first/linear/time-decay/position models vs true incrementality | last-touch is **most biased**, over-crediting a closer by **+20pp**; linear closest to truth |
| **Reconciliation** ([reconciliation.py](src/marketing_measurement/reconciliation.py)) | triangulate MTA vs MMM vs incrementality | flags over/under-credited channels + **customer-facing recommendations** |

The headline insight is the one that justifies Northbeam's whole product: **last-click attribution systematically over-credits "closer" channels (branded search, email) and starves the upper-funnel channels (TikTok, Meta) that actually drive incremental revenue** — which only incrementality and MMM reveal. The reconciliation tool turns that into per-channel recommendations ("validate with a geo holdout before scaling"), exactly the applied, customer-facing measurement work the role centers on. Because the data has a known causal lift and known channel effects, every estimate is graded against truth.

---

## Partner Analytics — segmentation, performance, experiments & monitoring

The end-to-end work a Partner Analytics data scientist owns — problem framing → model development → validation → productionization — on a synthetic B2B real-estate-partner portfolio with **known ground truth** so every estimate is graded against the truth it should recover:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m partner_analytics.run_partner       # segmentation + churn model + experiments + monitoring
PYTHONPATH=src python -m partner_analytics.build_dashboard   # partner analytics dashboard
PYTHONPATH=src python -m pytest tests/test_partner_analytics.py -q   # 6 tests incl. causal recovery & drift alerts
```

| Capability | Method | Result / insight |
|---|---|---|
| **Segmentation** ([segmentation.py](src/partner_analytics/segmentation.py)) | KMeans + value/engagement tiers | Champions / Growers / Developing / Low-Engagement, each with a recommended play |
| **Performance model** ([performance_model.py](src/partner_analytics/performance_model.py)) | GBM + logistic baseline, permutation importance | churn **AUC 0.88**, captures **54% of churn in the top 20%**, recovers all true drivers (rank agreement 1.0) |
| **A/B experiment** ([experiments.py](src/partner_analytics/experiments.py)) | CUPED variance reduction + SRM guard | lift 0.040 (= true), **~68% lower standard error** vs raw, SRM clean |
| **Causal inference** (observational) | 1-NN propensity-score matching | naive 3.24 → **PSM ATT 2.6 vs true 2.5**; covariate balance SMD 1.22 → 0.01 |
| **Production monitoring** ([monitoring.py](src/partner_analytics/monitoring.py)) | batch scoring + PSI + AUC decay | correctly flags month-2 **feature drift** and **AUC decay 0.919 → 0.885** |

The two centerpieces are **causal inference** and **monitoring**. Because the synthetic data has a *known* causal effect and injected drift, the propensity-score-matching estimate is **graded against ground truth** (it removes the confounding bias the naive comparison suffers), and the drift monitor is shown catching a real covariate shift and the performance decay it causes. That maps directly to the JD's "customer segmentation models (clustering, regression, tree-based)," "experiments and observational studies (A/B testing, causal inference)," and "monitoring model performance over time."

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
