"""Flask API + control-room dashboard.   Run:  python -m src.api   ->  http://127.0.0.1:5000"""
import json
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response
from src.config import *
from src.alerts import level
from src.predict import predict_steady

app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive Maintenance Control Room</title><style>
:root{--bg:%(bg)s;--panel:%(panel)s;--border:%(border)s;--text:%(text)s;--muted:%(muted)s;--amber:%(amber)s;--red:%(red)s;--green:%(green)s;--blue:%(blue)s}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Arial,sans-serif}
header{padding:18px 24px;border-bottom:2px solid var(--amber);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
header h1{font-size:20px;margin:0}header span{color:var(--amber)}.pill{margin-left:auto;font-size:12px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%%;background:var(--green);margin-right:6px;box-shadow:0 0 8px var(--green)}
main{padding:20px;display:grid;gap:16px;max-width:1200px;margin:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.kpi b{display:block;font-size:28px;margin-top:4px}.kpi small,h2{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
h2{margin:0 0 10px}table{width:100%%;border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid var(--border);text-align:left}
.bar{height:8px;background:var(--border);border-radius:4px;overflow:hidden;min-width:80px}.bar i{display:block;height:100%%}
.tag{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;color:#111}
.OK{background:var(--green)}.WARNING{background:var(--amber)}.CRITICAL{background:var(--red)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
label{display:block;font-size:12px;color:var(--muted);margin-top:8px}input{width:100%%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px;border-radius:6px}
button{margin-top:12px;background:var(--amber);border:0;color:#111;font-weight:700;padding:10px 16px;border-radius:6px;cursor:pointer}
#res{font-size:34px;font-weight:800;margin-top:10px}img{width:100%%;border-radius:8px;border:1px solid var(--border)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
</style></head><body>
<header><h1>⚙️ Predictive <span>Maintenance</span> Control Room</h1><div class="pill"><i class="dot"></i>Model: <b id="mname">…</b></div></header>
<main>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>Machine fleet – latest risk</h2><div style="overflow-x:auto"><table id="tbl"></table></div></div>
<div class="two">
<div class="card"><h2>What-if simulator</h2>
<label>Temperature (°C)</label><input id="temperature" type="number" value="70">
<label>Vibration (mm/s)</label><input id="vibration" type="number" step="0.1" value="2">
<label>Pressure (psi)</label><input id="pressure" type="number" value="100">
<label>RPM</label><input id="rpm" type="number" value="1500">
<label>Current (A)</label><input id="current" type="number" step="0.1" value="12">
<button onclick="pred()">Predict failure risk</button><div id="res">–</div><div id="lvl"></div>
<small style="color:var(--muted)">Assumes these readings held steady for the last 24 h.</small></div>
<div class="card"><h2>Latest alerts</h2><div id="alerts" style="font-size:14px"></div></div></div>
<div class="card"><h2>Model evidence</h2><div class="grid">
<img src="/images/06_failure_prediction_timeline.png"><img src="/images/03_confusion_matrix.png"><img src="/images/05_feature_importance.png"></div></div>
</main><script>
const col=r=>r>=%(crit)s?'var(--red)':r>=%(warn)s?'var(--amber)':'var(--green)';
async function load(){const d=await (await fetch('/api/machines')).json();
document.getElementById('mname').textContent=d.model;
const m=d.machines,c=m.filter(x=>x.level=='CRITICAL').length,w=m.filter(x=>x.level=='WARNING').length;
document.getElementById('kpis').innerHTML=[['Machines',m.length,'var(--blue)'],['Critical',c,'var(--red)'],['Warning',w,'var(--amber)'],['Healthy',m.length-c-w,'var(--green)'],['Best F1',d.f1.toFixed(2),'var(--amber)']]
.map(k=>`<div class="card kpi"><small>${k[0]}</small><b style="color:${k[2]}">${k[1]}</b></div>`).join('');
document.getElementById('tbl').innerHTML='<tr><th>Machine</th><th>Risk</th><th></th><th>Status</th></tr>'+m.map(x=>`<tr><td>${x.machine_id}</td><td>${(x.risk*100).toFixed(0)}%%</td><td><div class="bar"><i style="width:${x.risk*100}%%;background:${col(x.risk)}"></i></div></td><td><span class="tag ${x.level}">${x.level}</span></td></tr>`).join('');
const a=await (await fetch('/api/alerts')).json();
document.getElementById('alerts').innerHTML=a.alerts.map(x=>`<p><span class="tag ${x.level}">${x.level}</span> ${x.message}</p>`).join('')||'No alerts';}
async function pred(){const b={};for(const k of ['temperature','vibration','pressure','rpm','current'])b[k]=+document.getElementById(k).value;
const r=await (await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json();
const e=document.getElementById('res');e.textContent=(r.risk*100).toFixed(1)+'%%';e.style.color=col(r.risk);
document.getElementById('lvl').innerHTML=`<span class="tag ${r.level}">${r.level}</span>`;}
load();</script></body></html>"""

@app.route("/")
def home():
    return Response(PAGE % {**THEME, "crit": CRIT_THRESHOLD, "warn": WARN_THRESHOLD}, mimetype="text/html")

@app.route("/api/health")
def health():
    return jsonify(status="ok")

@app.route("/api/machines")
def machines():
    p = pd.read_csv(PRED_CSV).sort_values("timestamp").groupby("machine_id").tail(1)
    p["level"] = p["risk"].apply(level)
    met = json.loads(METRICS_JSON.read_text())
    return jsonify(model=met["best"], f1=met["results"][met["best"]]["f1"],
                   machines=p[["machine_id", "risk", "level"]].sort_values("risk", ascending=False).to_dict("records"))

@app.route("/api/alerts")
def alerts():
    a = pd.read_csv(ALERT_CSV).sort_values("timestamp").tail(8)[::-1]
    return jsonify(alerts=a[["machine_id", "level", "message"]].to_dict("records"))

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        r = predict_steady({s: float(data[s]) for s in SENSORS})
    except (KeyError, ValueError):
        return jsonify(error=f"send numbers for: {SENSORS}"), 400
    return jsonify(risk=r, level=level(r))

@app.route("/images/<path:f>")
def images(f):
    return send_from_directory(IMG_DIR, f)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
