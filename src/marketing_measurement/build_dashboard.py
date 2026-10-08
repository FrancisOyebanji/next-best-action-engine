"""Marketing measurement dashboard from reports/marketing_measurement.json."""
from __future__ import annotations

import json
from pathlib import Path

R = json.loads(Path("reports/marketing_measurement.json").read_text())
inc, mm, at, rec = R["incrementality"], R["mmm"], R["attribution"], R["reconciliation"]

CH = ["tiktok", "meta", "google_ppc", "email", "branded_search"]
true_share = [at["true_incremental_share"][c] for c in CH]
lt_share = [at["model_shares"]["last_touch"][c] for c in CH]
mmm_share = [mm["contribution_share"].get(c, 0) for c in CH]

kpis = [
    ("Incrementality lift (true)", f"{inc['true_lift_pct']*100:.0f}%"),
    ("Measured lift (DiD)", f"{inc['did_lift_pct']*100:.1f}%"),
    ("iROAS", f"{inc['iroas']:.1f}x"),
    ("MMM R²", f"{mm['r_squared']:.2f}"),
    ("MMM rank agreement", f"{mm['rank_agreement']:.2f}"),
    ("Best MTA model", at["best_model"].replace("_", "-")),
]
kpi_html = "".join(
    f"<div class='kpi'><div class='l'>{l}</div><div class='v'>{v}</div></div>" for l, v in kpis)

rec_table = "".join(
    "<tr><td>{channel}</td><td>{last_touch_share}</td><td>{mmm_share}</td>"
    "<td>{gap:+}</td><td style='text-align:left'>{verdict}</td></tr>".format(**r)
    for r in rec["table"])
rec_list = "".join(f"<li>{x}</li>" for x in rec["recommendations"]) or "<li>No material divergences.</li>"

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Marketing Measurement</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f6fa;color:#191c33;margin:0}}
 header{{background:#1b1f4b;color:#fff;padding:16px 30px}} h1{{font-size:20px;margin:0}}
 .sub{{color:#c3c7ea;font-size:13px}}
 main{{max-width:1120px;margin:22px auto;padding:0 18px}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px;margin-bottom:20px}}
 .kpi{{background:#fff;border:1px solid #e4e6f0;border-radius:12px;padding:16px}}
 .kpi .l{{color:#5f6388;font-size:13px}}.kpi .v{{font-size:22px;font-weight:700;margin-top:4px;color:#1b1f4b}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 .panel{{background:#fff;border:1px solid #e4e6f0;border-radius:12px;padding:18px;margin-bottom:18px}}
 .panel h2{{font-size:15px;margin:0 0 12px}} canvas{{max-height:290px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:7px 9px;border-bottom:1px solid #eef0f6;text-align:center}}
 th{{color:#5f6388}}
 .note{{background:#eceef9;border-left:4px solid #1b1f4b;padding:10px 14px;border-radius:6px;font-size:14px}}
 ul{{margin:8px 0;padding-left:18px;font-size:14px}} ul li{{margin:5px 0}}
 @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
 footer{{color:#5f6388;font-size:12px;text-align:center;margin:24px 0}}
</style></head><body>
<header><h1>Marketing Measurement &mdash; incrementality, MMM & attribution</h1>
<div class="sub">Geo incrementality &middot; MMM+ (adstock + saturation) &middot; multi-touch attribution &middot; reconciliation</div></header>
<main>
<div class="kpis">{kpi_html}</div>
<div class="grid">
 <div class="panel"><h2>Attribution vs truth &mdash; credit share by channel</h2><canvas id="c1"></canvas>
   <p style="font-size:13px;color:#5f6388;margin-top:10px">Last-touch over-credits
   <b>{at['most_over_credited_by_last_touch']}</b> (a closer) by +{at['last_touch_over_credit']*100:.0f}pp
   vs its true incremental share &mdash; the case for incrementality over last-click.</p></div>
 <div class="panel"><h2>Incrementality &mdash; measured vs true lift</h2><canvas id="c2"></canvas>
   <p style="font-size:13px;color:#5f6388;margin-top:10px">Geo holdout: DiD {inc['did_lift_pct']*100:.1f}% and synthetic
   control {inc['synthetic_control_lift_pct']*100:.1f}% vs true {inc['true_lift_pct']*100:.0f}%; iROAS {inc['iroas']:.1f}x.</p></div>
</div>
<div class="panel"><h2>Reconciliation &mdash; last-touch vs MMM by channel</h2>
 <table><tr><th>channel</th><th>last-touch share</th><th>MMM share</th><th>gap</th><th style="text-align:left">verdict</th></tr>{rec_table}</table>
 <h3 style="font-size:14px;margin:14px 0 4px">Recommendations</h3><ul>{rec_list}</ul></div>
<div class="panel"><h2>Why it maps to the role</h2><div class="note">
 Northbeam's three measurement lenses &mdash; multi-touch attribution, MMM+, and incrementality &mdash; built end to end,
 triangulated into a customer-facing diagnostic, with every estimate graded against known ground truth.
</div></div>
</main>
<footer>Synthetic data &middot; src/marketing_measurement/build_dashboard.py &middot; estimates graded vs known truth</footer>
<script>
new Chart(c1,{{type:'bar',data:{{labels:{json.dumps(CH)},datasets:[
  {{label:'true incremental',data:{json.dumps(true_share)},backgroundColor:'#14804a'}},
  {{label:'last-touch',data:{json.dumps(lt_share)},backgroundColor:'#b42318'}},
  {{label:'MMM',data:{json.dumps(mmm_share)},backgroundColor:'#2749c9'}}
 ]}},options:{{scales:{{y:{{min:0}}}}}}}});
new Chart(c2,{{type:'bar',data:{{labels:['True','DiD','Synthetic control'],
  datasets:[{{data:[{inc['true_lift_pct']},{inc['did_lift_pct']},{inc['synthetic_control_lift_pct']}],
  backgroundColor:['#14804a','#1b1f4b','#7a8cc9']}}]}},
  options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{min:0}}}}}}}});
</script></body></html>"""
Path("reports").mkdir(exist_ok=True)
Path("reports/marketing_dashboard.html").write_text(html, encoding="utf-8")
print("Dashboard written to reports/marketing_dashboard.html")
