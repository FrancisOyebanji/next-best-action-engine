"""Next Best Action decision engine.

Given per-member, per-action uplift scores, decide the single next best action
for each member, subject to the decisioning rules a real NBA platform (e.g. Pega)
enforces:

  1. Eligibility  — inclusion/exclusion rules per action (e.g. care_gap_outreach
     only for members with an open care gap).
  2. Expected value — rank eligible actions by  uplift x action_value.
  3. Suppression  — drop actions whose expected value is below a threshold
     (don't contact where the incremental value doesn't justify it), and drop
     negative-uplift actions ("sleeping dogs").
  4. Constraints  — one action per member (contact fatigue) and per-action
     capacity caps (campaign sizing / operational limits).

The output is an auditable decision per member: the chosen action, why, and the
runner-up — the artifact campaign ops and stakeholders consume.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---- eligibility rules (inclusion/exclusion) ----
def eligible(member: dict, action: str) -> bool:
    if action == "care_gap_outreach":
        return float(member["open_care_gaps"]) >= 1            # inclusion
    if action == "wellness_program":
        return float(member["age"]) < 70                       # exclusion of >=70
    if action == "telehealth_nudge":
        return float(member["digital_active"]) == 1            # inclusion: digital members
    return True


@dataclass
class Decision:
    member_id: str
    action: str | None
    expected_value: float
    reason: str
    runner_up: str | None


def decide_batch(members: list[dict], uplift_by_action: dict[str, dict],
                 action_values: dict[str, float],
                 min_expected_value: float = 0.05,
                 capacity: dict[str, int] | None = None) -> list[Decision]:
    """members: list of member dicts. uplift_by_action[action][member_id] = uplift."""
    # Score every eligible (member, action) pair by expected value
    scored = []  # (member_id, action, expected_value, uplift)
    for m in members:
        mid = m["member_id"]
        for a, val in action_values.items():
            up = uplift_by_action[a].get(mid)
            if up is None or not eligible(m, a):
                continue
            ev = up * val
            if up > 0 and ev >= min_expected_value:      # suppress sleeping-dogs & low EV
                scored.append((mid, a, ev, up))

    # Greedy assignment: highest expected value first, one action per member,
    # respecting per-action capacity caps.
    scored.sort(key=lambda s: -s[2])
    used_member, used_cap = set(), {a: 0 for a in action_values}
    chosen: dict[str, tuple] = {}
    runner: dict[str, str] = {}
    for mid, a, ev, up in scored:
        if mid in used_member:
            if mid not in runner:
                runner[mid] = a
            continue
        if capacity and used_cap[a] >= capacity.get(a, 10 ** 12):
            continue
        chosen[mid] = (a, ev, up)
        used_member.add(mid)
        used_cap[a] += 1

    decisions = []
    for m in members:
        mid = m["member_id"]
        if mid in chosen:
            a, ev, up = chosen[mid]
            decisions.append(Decision(mid, a, round(ev, 4),
                                      f"highest expected value (uplift {up:.3f} x value)",
                                      runner.get(mid)))
        else:
            decisions.append(Decision(mid, None, 0.0,
                                      "no eligible action above value/uplift threshold", None))
    return decisions


def summarize(decisions: list[Decision]) -> dict:
    from collections import Counter
    counts = Counter(d.action for d in decisions)
    contacted = sum(1 for d in decisions if d.action)
    return {
        "members": len(decisions),
        "contacted": contacted,
        "suppressed": len(decisions) - contacted,
        "by_action": {a: c for a, c in counts.items() if a},
        "total_expected_value": round(sum(d.expected_value for d in decisions), 1),
    }
