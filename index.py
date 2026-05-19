import os
import math
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
from modelo.predict import init, predict, top5, team_stats, get_available_teams
from modelo.montecarlo import print_json_prediction
#from modelo.hibrido.main import prediccion_personalizada


def _sanitize(obj):
    """Reemplaza NaN/Inf con None recursivamente para producir JSON válido."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


app = Flask(__name__)
CORS(app)

init()

PREDICT_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Predicción de Partido</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 16px; }
    h1 { font-size: 1.8rem; margin-bottom: 8px; color: #58a6ff; }
    p.subtitle { color: #8b949e; margin-bottom: 32px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 32px; width: 100%; max-width: 520px; }
    .field { margin-bottom: 20px; }
    label { display: block; font-size: 0.85rem; color: #8b949e; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
    select { width: 100%; padding: 10px 14px; background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px; font-size: 1rem; appearance: none; cursor: pointer; }
    select:focus { outline: none; border-color: #58a6ff; }
    select option:disabled { color: #484f58; }
    button { width: 100%; padding: 12px; background: #238636; color: #fff; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover:not(:disabled) { background: #2ea043; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #result { margin-top: 28px; display: none; }
    .section-title { font-size: 0.9rem; color: #58a6ff; font-weight: 600; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.06em; }
    .prob-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 16px; }
    .prob-box { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }
    .prob-box .label { font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; }
    .prob-box .value { font-size: 1.3rem; font-weight: 700; color: #e6edf3; }
    .prob-box.highlight .value { color: #58a6ff; }
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .stat-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .stat-card h4 { font-size: 0.8rem; color: #8b949e; margin-bottom: 8px; }
    .stat-row { display: flex; justify-content: space-between; font-size: 0.85rem; padding: 3px 0; border-bottom: 1px solid #21262d; }
    .stat-row:last-child { border-bottom: none; }
    .stat-row .key { color: #8b949e; }
    .stat-row .val { color: #e6edf3; font-weight: 500; }
    .error-msg { background: #2d1117; border: 1px solid #f85149; border-radius: 8px; padding: 12px; color: #f85149; margin-top: 16px; }
    .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #fff; border-top-color: transparent; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <h1>Predicción de Partido</h1>
  <p class="subtitle">Premier League — Selecciona los equipos</p>
  <div class="card">
    <div class="field">
      <label for="home">Equipo Local</label>
      <select id="home"><option value="">-- Selecciona equipo local --</option></select>
    </div>
    <div class="field">
      <label for="away">Equipo Visitante</label>
      <select id="away"><option value="">-- Selecciona equipo visitante --</option></select>
    </div>
    <button id="btn" disabled>Predecir</button>
    <div id="result"></div>
  </div>

  <script>
    const homeEl = document.getElementById('home');
    const awayEl = document.getElementById('away');
    const btn = document.getElementById('btn');
    const resultEl = document.getElementById('result');
    let teams = [];

    async function loadTeams() {
      const res = await fetch('/teams');
      teams = await res.json();
      teams.forEach(t => {
        homeEl.add(new Option(t, t));
        awayEl.add(new Option(t, t));
      });
    }

    function syncSelects() {
      const hVal = homeEl.value;
      const aVal = awayEl.value;
      [...awayEl.options].forEach(o => { o.disabled = o.value !== '' && o.value === hVal; });
      [...homeEl.options].forEach(o => { o.disabled = o.value !== '' && o.value === aVal; });
      btn.disabled = !hVal || !aVal;
    }

    homeEl.addEventListener('change', syncSelects);
    awayEl.addEventListener('change', syncSelects);

    btn.addEventListener('click', async () => {
      const home = encodeURIComponent(homeEl.value);
      const away = encodeURIComponent(awayEl.value);
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Calculando...';
      resultEl.style.display = 'none';
      try {
        const res = await fetch(`/team/${home}/${away}`);
        const data = await res.json();
        if (!res.ok) { throw new Error(data.error || 'Error desconocido'); }
        renderResult(data);
      } catch (e) {
        resultEl.style.display = 'block';
        resultEl.innerHTML = `<div class="error-msg">Error: ${e.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Predecir';
      }
    });

    function renderResult(data) {
      const p = data.predict || {};
      const probs = p['Probabilidades'] || {};
      const homeStats = p['Estadísticas recientes del equipo local'] || {};
      const awayStats = p['Estadísticas recientes del equipo visitante'] || {};
      const mc = data.montecarlo || {};

      const maxKey = Object.entries(probs).sort((a,b) => parseFloat(b[1]) - parseFloat(a[1]))[0]?.[0];

      const probBoxes = Object.entries(probs).map(([k,v]) =>
        `<div class="prob-box ${k === maxKey ? 'highlight' : ''}">
          <div class="label">${k}</div>
          <div class="value">${v}</div>
        </div>`
      ).join('');

      const statRows = (obj) => Object.entries(obj)
        .filter(([k]) => k !== 'n_played')
        .map(([k,v]) => `<div class="stat-row"><span class="key">${k.replace(/_/g,' ')}</span><span class="val">${v !== null ? v : '—'}</span></div>`)
        .join('');

      let mcSection = '';
      if (mc && Object.keys(mc).length) {
        const mcRows = Object.entries(mc)
          .filter(([k]) => !['home','away'].includes(k))
          .map(([k,v]) => `<div class="stat-row"><span class="key">${k}</span><span class="val">${typeof v === 'number' ? (v >= 0 && v <= 1 ? (v*100).toFixed(1)+'%' : v.toFixed(2)) : v}</span></div>`)
          .join('');
        if (mcRows) mcSection = `<div class="section-title">Montecarlo</div><div class="stat-card" style="grid-column:1/-1">${mcRows}</div>`;
      }

      resultEl.style.display = 'block';
      resultEl.innerHTML = `
        <div class="section-title">Probabilidades</div>
        <div class="prob-grid">${probBoxes}</div>
        <div class="section-title">Estadísticas recientes</div>
        <div class="stats-grid">
          <div class="stat-card"><h4>${p['Equipo_local'] || homeEl.value} (Local)</h4>${statRows(homeStats)}</div>
          <div class="stat-card"><h4>${p['Equipo_visitante'] || awayEl.value} (Visitante)</h4>${statRows(awayStats)}</div>
          ${mcSection}
        </div>`;
    }

    loadTeams();
  </script>
</body>
</html>
"""


@app.route('/', methods=['GET'])
def home():
    result = top5()
    return jsonify({'top5': result})


@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template_string(PREDICT_PAGE)


@app.route('/teams', methods=['GET'])
def teams_endpoint():
    return jsonify(get_available_teams())


@app.route('/team/<string:teamhome>/<string:teamaway>', methods=['GET'])
def disp(teamhome, teamaway):
    try:
        result = print_json_prediction(teamhome, teamaway)
        result2 = predict(teamhome, teamaway)
        #resultLogit = prediccion_personalizada(teamhome, teamaway)
        return jsonify(_sanitize({'montecarlo': result, 'predict': result2}))
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/team/<string:team>', methods=['GET'])
def team_stats_endpoint(team):
    try:
        result = team_stats(team)
        return jsonify(_sanitize(result))
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/status', methods=['GET'])
def status():
    try:
         return {'status': 'ok'}
    except ValueError as e:
        return jsonify({'error': str(e)}), 404