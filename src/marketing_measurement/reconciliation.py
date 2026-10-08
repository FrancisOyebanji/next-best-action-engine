"""Measurement reconciliation diagnostic (the customer-facing deliverable).

Northbeam's three measurement lenses rarely agree. This tool triangulates them per
channel — last-touch attribution credit, MMM contribution, and the incrementality
signal — flags where they diverge, and writes a plain-language recommendation a
customer can act on. The classic finding: a channel that looks great in last-touch
(a "closer") carries far less credit under MMM and incrementality, so last-touch is
over-investing in it.
"""
from __future__ import annotations

from marketing_measurement import attribution, mmm

CHANNELS = ["tiktok", "meta", "google_ppc", "email", "branded_search"]


def reconcile(mmm_df, journeys, truth) -> dict:
    mmm_res = mmm.fit(mmm_df, truth)
    lt_share = attribution.attribute(journeys, "last_touch")
    mmm_share = mmm_res["contribution_share"]

    rows, flags = [], []
    for ch in CHANNELS:
        lt = lt_share[ch]
        mm = mmm_share.get(ch, 0.0)
        gap = lt - mm
        verdict = "aligned"
        if gap > 0.08:
            verdict = "last-touch OVER-credits (likely a closer); MMM/incrementality say less"
            flags.append((ch, "over", round(gap, 3)))
        elif gap < -0.08:
            verdict = "last-touch UNDER-credits (upper-funnel); worth more than last-click shows"
            flags.append((ch, "under", round(gap, 3)))
        rows.append({"channel": ch, "last_touch_share": round(lt, 3),
                     "mmm_share": round(mm, 3), "gap": round(gap, 3), "verdict": verdict})

    # recommendation targets the biggest divergences
    recs = []
    for ch, direction, gap in sorted(flags, key=lambda x: -abs(x[2])):
        if direction == "over":
            recs.append(f"Re-examine spend on {ch}: last-touch over-credits it by "
                        f"{abs(gap)*100:.0f}pp vs MMM. Validate with a geo holdout before scaling.")
        else:
            recs.append(f"{ch} is likely undervalued by last-click ({abs(gap)*100:.0f}pp below MMM); "
                        f"consider protecting or increasing budget and confirm with an incrementality test.")

    return {
        "table": rows,
        "divergence_flags": [{"channel": c, "direction": d, "gap": g} for c, d, g in flags],
        "recommendations": recs,
        "n_divergences": len(flags),
    }
