"""Partner Analytics dashboard from reports/partner_analytics.json (self-service readout)."""
from __future__ import annotations

import json
from pathlib import Path

R = json.loads(Path("reports/partner_analytics.json").read_text())
seg = R["segmentation"]["segments"]
perf = R["performance_model"]
ab = R["experiments"]["ab_test"]
ps = R["experiments"]["causal_psm"]
mon = R["monitoring"]

seg_labels = list(seg.keys())
seg_counts = [seg[s]["n_partners"] for s in seg_labels]
seg_churn = [round(seg[s]["churn_rate"] * 100, 1) for s in seg_labels]

imp = perf["importances"]
imp_items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:8]
imp_labels = [k for k, _ in imp_items]
imp_vals = [round(v, 4) for _, v in imp_items]

months = list(mon["months"].keys())
auc_series = [mon["months"][m]["auc"] for m in months]
psi_leads = [mon["months"][m]["feature_psi"].get("leads_received", 0) for m in months]

data = {
    "seg_labels": seg_labels, "seg_counts": seg_counts, "seg_churn": seg_churn,
    "imp_labels": imp_labels, "imp_vals": imp_vals,
    "months": ["baseline"] + months, "auc_series": [mon["baseline_auc"]] + auc_series,
    "psi_months": months, "psi_leads": psi_leads,
}

alerts_html = "".join(f"<li>{a}</li>" for a in mon["alerts"]) or "<li>All batches healthy</li>"

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Partner Analytics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f4f6fb;color:#13203a;margin:0}}
 header{{background:#0b2a6b;color:#fff;padding:16px 30px}} h1{{font-size:20px;margin:0}}
 .sub{{color:#b9c8ea;font-size:13px}}
 main{{max-width:1120px;margin:22px auto;padding:0 18px}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px}}
 .kpi{{background:#fff;border:1px solid #e3e8f2;border-radius:12px;padding:16px}}
 .kpi .l{{color:#5a6a86;font-size:13px}}.kpi .v{{font-size:22px;font-weight:700;margin-top:4px;color:#0b2a6b}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 .panel{{background:#fff;border:1px solid #e3e8f2;border-radius:12px;padding:18px;margin-bottom:18px}}
 .panel h2{{font-size:15px;margin:0 0 12px}} canvas{{max-height:270px}}
 .reco{{background:#eef3ff;border-left:4px solid #0b2a6b;padding:10px 14px;border-radius:6px;font-size:14px}}
 ul.alerts{{margin:0;padding-left:18px;font-size:14px}} ul.alerts li{{margin:4px 0;color:#b42318}}
 table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eef1f7}}
 th{{color:#5a6a86;font-weight:600}}
 @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
 footer{{color:#5a6a86;font-size:12px;text-align:center;margin:24px 0}}
</style></head><body>
<header><h1>Partner Analytics &mdash; portfolio segmentation, churn, experiments & monitoring</h1>
<div class="sub">Segmentation (clustering) &middot; churn performance model &middot; A/B + causal inference &middot; production monitoring</div></header>
<main>
<div class="kpis">
 <div class="kpi"><div class="l">Churn model AUC (GBM)</div><div class="v">{perf['auc_gbm']}</div></div>
 <div class="kpi"><div class="l">Churn captured @20% targeted</div><div class="v">{round(perf['capture_at_20pct_targeted']*100)}%</div></div>
 <div class="kpi"><div class="l">A/B CUPED lift (SE &minus;{ab['se_reduction_pct']}%)</div><div class="v">{round(ab['cuped_lift']*100,2)} pp</div></div>
 <div class="kpi"><div class="l">Causal ATT (FUB adoption)</div><div class="v">+{ps['psm_att']} tx</div></div>
</div>
<div class="grid">
 <div class="panel"><h2>Partner segments &mdash; size & churn rate</h2><canvas id="c1"></canvas></div>
 <div class="panel"><h2>Churn drivers &mdash; permutation importance</h2><canvas id="c2"></canvas></div>
</div>
<div class="grid">
 <div class="panel"><h2>Model monitoring &mdash; AUC over monthly batches</h2><canvas id="c3"></canvas></div>
 <div class="panel"><h2>Causal estimate vs ground truth</h2>
   <table>
     <tr><th>Estimator</th><th>Effect (transactions)</th><th>Bias vs truth</th></tr>
     <tr><td>Naive (confounded)</td><td>{ps['naive_diff']}</td><td>{ps['naive_bias']:+}</td></tr>
     <tr><td>Propensity-score matching</td><td><b>{ps['psm_att']}</b></td><td>{ps['psm_bias']:+}</td></tr>
     <tr><td>Known true effect</td><td>{ps['true_effect']}</td><td>&mdash;</td></tr>
   </table>
   <p style="font-size:13px;color:#5a6a86;margin-top:10px">A/B raw lift {ab['raw_lift']} &rarr; CUPED {ab['cuped_lift']}
   (variance &minus;{ab['variance_reduction_pct']}%); SRM p={ab['srm']['p_value']}.</p>
 </div>
</div>
<div class="panel"><h2>Monitoring alerts (what an orchestrator would page on)</h2>
 <ul class="alerts">{alerts_html}</ul></div>
<div class="panel"><h2>Recommendation</h2><div class="reco">
 Target the {round(perf['capture_at_20pct_targeted']*100)}% of churn concentrated in the top-scored 20% of partners with retention plays;
 roll out Follow Up Boss (causal +{ps['psm_att']} transactions, confounding-adjusted); ship the zPro variant
 (+{round(ab['cuped_lift']*100,2)}pp conversion, confirmed under CUPED); and retrain after the month-2 drift alert.
</div></div>
</main>
<footer>Synthetic partner RWD &middot; src/partner_analytics/build_dashboard.py &middot; causal & driver estimates graded vs known ground truth</footer>
<script>
const D={json.dumps(data)};
new Chart(c1,{{data:{{labels:D.seg_labels,datasets:[
  {{type:'bar',label:'partners',data:D.seg_counts,backgroundColor:'#0b2a6b',yAxisID:'y'}},
  {{type:'line',label:'churn %',data:D.seg_churn,borderColor:'#e8a33d',backgroundColor:'#e8a33d',yAxisID:'y1',tension:.2}}
 ]}},options:{{scales:{{y:{{position:'left',title:{{display:true,text:'partners'}}}},
   y1:{{position:'right',title:{{display:true,text:'churn %'}},grid:{{drawOnChartArea:false}}}}}}}}}});
new Chart(c2,{{type:'bar',data:{{labels:D.imp_labels,datasets:[{{data:D.imp_vals,backgroundColor:'#2749c9'}}]}},
  options:{{indexAxis:'y',plugins:{{legend:{{display:false}}}}}}}});
new Chart(c3,{{type:'line',data:{{labels:D.months,datasets:[{{label:'AUC',data:D.auc_series,
  borderColor:'#14804a',backgroundColor:'#14804a',tension:.2,pointRadius:4}}]}},
  options:{{scales:{{y:{{min:0.8,max:0.95}}}}}}}});
</script></body></html>"""
Path("reports").mkdir(exist_ok=True)
Path("reports/partner_dashboard.html").write_text(html, encoding="utf-8")
print("Dashboard written to reports/partner_dashboard.html")
