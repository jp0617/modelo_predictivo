"""
Modelo de Simulación Monte Carlo - Premier League

Simula partidos usando distribuciones de Poisson calibradas con datos históricos.
Devuelve probabilidades de Victoria/Empate/Derrota, goles esperados y
distribución de marcadores posibles.

Uso standalone: lee directamente de los CSV del directorio data/.
"""

import os
from unittest import result
import numpy as np
import pandas as pd
from scipy.stats import poisson

# ─── Normalización de nombres ────────────────────────────────────────────────

TEAM_NAME_MAP = {
    # Nombres cortos del histórico → nombre oficial completo
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Birmingham": "Birmingham City FC",
    "Blackburn": "Blackburn Rovers FC",
    "Blackpool": "Blackpool FC",
    "Bolton": "Bolton Wanderers FC",
    "Bournemouth": "AFC Bournemouth",
    "Bradford": "Bradford City FC",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Burnley": "Burnley FC",
    "Cardiff": "Cardiff City FC",
    "Charlton": "Charlton Athletic FC",
    "Chelsea": "Chelsea FC",
    "Coventry": "Coventry City FC",
    "Crystal Palace": "Crystal Palace FC",
    "Derby": "Derby County FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Huddersfield": "Huddersfield Town FC",
    "Hull": "Hull City FC",
    "Ipswich": "Ipswich Town FC",
    "Leeds": "Leeds United FC",
    "Leicester": "Leicester City FC",
    "Liverpool": "Liverpool FC",
    "Luton": "Luton Town FC",
    "Man City": "Manchester City FC",
    "Man United": "Manchester United FC",
    "Middlesbrough": "Middlesbrough FC",
    "Newcastle": "Newcastle United FC",
    "Norwich": "Norwich City FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Portsmouth": "Portsmouth FC",
    "QPR": "Queens Park Rangers FC",
    "Reading": "Reading FC",
    "Sheffield United": "Sheffield United FC",
    "Southampton": "Southampton FC",
    "Stoke": "Stoke City FC",
    "Sunderland": "Sunderland AFC",
    "Swansea": "Swansea City FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Watford": "Watford FC",
    "West Brom": "West Bromwich Albion FC",
    "West Ham": "West Ham United FC",
    "Wigan": "Wigan Athletic FC",
    "Wolves": "Wolverhampton Wanderers FC",
    # Variantes que pueden aparecer con espacios o tildes distintas
    "AFC Bournemouth": "AFC Bournemouth",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion FC",
    "Sunderland AFC": "Sunderland AFC",
}


def normalize_team_name(name: str) -> str:
    """Devuelve el nombre oficial completo del equipo."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)


# ─── Carga y preparación de datos ────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


def load_all_data() -> pd.DataFrame:
    """
    Carga y une datos históricos + temporada actual,
    normalizando los nombres de equipos.
    """
    hist = _load_csv("epl_final.csv")
    current = _load_csv("PL_2025_actual.csv")

    # Normalizar nombres en histórico
    hist["HomeTeam"] = hist["HomeTeam"].apply(normalize_team_name)
    hist["AwayTeam"] = hist["AwayTeam"].apply(normalize_team_name)

    # Normalizar nombres en temporada actual (ya suelen estar bien, pero por consistencia)
    current["HomeTeam"] = current["HomeTeam"].apply(normalize_team_name)
    current["AwayTeam"] = current["AwayTeam"].apply(normalize_team_name)

    # Columnas mínimas necesarias
    cols = ["Season", "MatchDate", "HomeTeam", "AwayTeam",
            "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult"]
    df = pd.concat([hist[cols], current[cols]], ignore_index=True)

    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df["FullTimeHomeGoals"] = pd.to_numeric(df["FullTimeHomeGoals"], errors="coerce")
    df["FullTimeAwayGoals"] = pd.to_numeric(df["FullTimeAwayGoals"], errors="coerce")

    # Solo partidos jugados (con goles registrados)
    df = df.dropna(subset=["FullTimeHomeGoals", "FullTimeAwayGoals"])
    return df.sort_values("MatchDate").reset_index(drop=True)


# ─── Cálculo de fortalezas ────────────────────────────────────────────────────

def calculate_team_strengths(df: pd.DataFrame) -> dict:
    """
    Calcula la fortaleza ofensiva y defensiva de cada equipo
    relativa al promedio de la liga.

    Retorna dict con claves de equipo y valores:
        home_attack, home_defense, away_attack, away_defense
    """
    avg_home_goals = df["FullTimeHomeGoals"].mean()
    avg_away_goals = df["FullTimeAwayGoals"].mean()

    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    strengths = {}

    for team in teams:
        home_matches = df[df["HomeTeam"] == team]
        away_matches = df[df["AwayTeam"] == team]

        # Goles marcados y concedidos como local
        if len(home_matches) > 0:
            home_scored = home_matches["FullTimeHomeGoals"].mean()
            home_conceded = home_matches["FullTimeAwayGoals"].mean()
            home_attack = home_scored / avg_home_goals if avg_home_goals > 0 else 1.0
            home_defense = home_conceded / avg_away_goals if avg_away_goals > 0 else 1.0
        else:
            home_attack = 1.0
            home_defense = 1.0

        # Goles marcados y concedidos como visitante
        if len(away_matches) > 0:
            away_scored = away_matches["FullTimeAwayGoals"].mean()
            away_conceded = away_matches["FullTimeHomeGoals"].mean()
            away_attack = away_scored / avg_away_goals if avg_away_goals > 0 else 1.0
            away_defense = away_conceded / avg_home_goals if avg_home_goals > 0 else 1.0
        else:
            away_attack = 1.0
            away_defense = 1.0

        strengths[team] = {
            "home_attack": home_attack,
            "home_defense": home_defense,
            "away_attack": away_attack,
            "away_defense": away_defense,
        }

    return strengths, avg_home_goals, avg_away_goals


# ─── Simulación Monte Carlo ───────────────────────────────────────────────────

def simulate_match(
    home_team: str,
    away_team: str,
    n_simulations: int = 50_000,
    df: pd.DataFrame = None,
    strengths: dict = None,
    avg_home_goals: float = None,
    avg_away_goals: float = None,
    max_goals: int = 10,
) -> dict:
    """
    Simula un partido N veces mediante Monte Carlo con distribuciones de Poisson.

    Args:
        home_team: Nombre del equipo local (nombre oficial completo).
        away_team: Nombre del equipo visitante (nombre oficial completo).
        n_simulations: Número de simulaciones.
        df: DataFrame pre-cargado (opcional, evita recargar datos).
        strengths: Diccionario de fortalezas pre-calculado (opcional).
        avg_home_goals / avg_away_goals: Promedios globales pre-calculados (opcional).
        max_goals: Máximo de goles por equipo en la matriz de marcadores.

    Returns:
        dict con probabilidades, goles esperados, marcador más probable, etc.
    """
    # Normalizar nombres por si se pasa forma corta
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    if df is None:
        df = load_all_data()
    if strengths is None:
        strengths, avg_home_goals, avg_away_goals = calculate_team_strengths(df)

    s_home = strengths.get(home_team, {"home_attack": 1.0, "home_defense": 1.0})
    s_away = strengths.get(away_team, {"away_attack": 1.0, "away_defense": 1.0})

    # Lambdas de Poisson: ataque_local × defensa_visitante × promedio_liga
    lambda_home = s_home["home_attack"] * s_away["away_defense"] * avg_home_goals
    lambda_away = s_away["away_attack"] * s_home["home_defense"] * avg_away_goals

    # ── Distribución analítica de marcadores ──────────────────────────────
    home_probs = np.array([poisson.pmf(i, lambda_home) for i in range(max_goals + 1)])
    away_probs = np.array([poisson.pmf(i, lambda_away) for i in range(max_goals + 1)])
    score_matrix = np.outer(home_probs, away_probs)

    prob_home_win_poisson = float(np.sum(np.tril(score_matrix, -1)))
    prob_draw_poisson = float(np.sum(np.diag(score_matrix)))
    prob_away_win_poisson = float(np.sum(np.triu(score_matrix, 1)))

    # ── Simulación Monte Carlo ─────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    home_goals_sim = rng.poisson(lambda_home, n_simulations)
    away_goals_sim = rng.poisson(lambda_away, n_simulations)

    home_wins = int(np.sum(home_goals_sim > away_goals_sim))
    draws = int(np.sum(home_goals_sim == away_goals_sim))
    away_wins = int(np.sum(home_goals_sim < away_goals_sim))

    prob_home_win_mc = home_wins / n_simulations
    prob_draw_mc = draws / n_simulations
    prob_away_win_mc = away_wins / n_simulations

    # ── Over/Under 2.5 goles ──────────────────────────────────────────────
    total_goals_sim = home_goals_sim + away_goals_sim
    prob_over_2_5 = float(np.mean(total_goals_sim > 2.5))
    prob_under_2_5 = float(np.mean(total_goals_sim <= 2.5))

    # ── Ambos equipos marcan (BTTS) ───────────────────────────────────────
    prob_btts = float(np.mean((home_goals_sim > 0) & (away_goals_sim > 0)))

    # ── Marcador más probable (distribución analítica) ────────────────────
    most_likely_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    most_likely_score = f"{most_likely_idx[0]}-{most_likely_idx[1]}"
    most_likely_prob = float(score_matrix[most_likely_idx])

    # ── Top 5 marcadores más probables ───────────────────────────────────
    flat_indices = np.argsort(score_matrix.ravel())[::-1][:5]
    top_scores = []
    for idx in flat_indices:
        h, a = divmod(int(idx), max_goals + 1)
        top_scores.append({
            "score": f"{h}-{a}",
            "probability": round(float(score_matrix[h, a]), 4),
        })

    # ── Distribución de goles totales (MC) ────────────────────────────────
    unique_totals, counts = np.unique(total_goals_sim, return_counts=True)
    total_goals_dist = {int(t): round(c / n_simulations, 4)
                        for t, c in zip(unique_totals, counts)}

    # ── Resultado combinado (promedio Poisson analítico + MC) ─────────────
    prob_home_win = round(0.5 * prob_home_win_poisson + 0.5 * prob_home_win_mc, 4)
    prob_draw = round(0.5 * prob_draw_poisson + 0.5 * prob_draw_mc, 4)
    prob_away_win = round(0.5 * prob_away_win_poisson + 0.5 * prob_away_win_mc, 4)

    probs = {"H": prob_home_win, "D": prob_draw, "A": prob_away_win}
    predicted_result = max(probs, key=probs.get)
    result_labels = {"H": f"Victoria {home_team}", "D": "Empate", "A": f"Victoria {away_team}"}

    return {
        "home_team": home_team,
        "away_team": away_team,
        "n_simulations": n_simulations,
        "lambdas": {
            "home": round(lambda_home, 4),
            "away": round(lambda_away, 4),
        },
        "expected_goals": {
            "home": round(lambda_home, 2),
            "away": round(lambda_away, 2),
            "total": round(lambda_home + lambda_away, 2),
        },
        "probabilities": {
            "home_win": prob_home_win,
            "draw": prob_draw,
            "away_win": prob_away_win,
        },
        "monte_carlo_probabilities": {
            "home_win": round(prob_home_win_mc, 4),
            "draw": round(prob_draw_mc, 4),
            "away_win": round(prob_away_win_mc, 4),
        },
        "poisson_probabilities": {
            "home_win": round(prob_home_win_poisson, 4),
            "draw": round(prob_draw_poisson, 4),
            "away_win": round(prob_away_win_poisson, 4),
        },
        "most_likely_score": most_likely_score,
        "most_likely_score_probability": round(most_likely_prob, 4),
        "top_scores": top_scores,
        "over_under": {
            "over_2_5": round(prob_over_2_5, 4),
            "under_2_5": round(prob_under_2_5, 4),
        },
        "btts": round(prob_btts, 4),
        "total_goals_distribution": total_goals_dist,
        "predicted_result": predicted_result,
        "predicted_result_label": result_labels[predicted_result],
        "confidence": max(prob_home_win, prob_draw, prob_away_win),
    }


def simulate_season(
    remaining_fixtures: list[tuple],
    n_simulations: int = 10_000,
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Simula los partidos restantes de la temporada N veces y devuelve
    la tabla de clasificación promedio esperada.

    Args:
        remaining_fixtures: Lista de tuplas (home_team, away_team).
        n_simulations: Número de simulaciones de temporada completa.
        df: DataFrame pre-cargado (opcional).

    Returns:
        DataFrame con columnas: Team, Pts_avg, W_avg, D_avg, L_avg, GF_avg, GA_avg
        ordenado por Pts_avg descendente.
    """
    if df is None:
        df = load_all_data()

    strengths, avg_home_goals, avg_away_goals = calculate_team_strengths(df)
    rng = np.random.default_rng(seed=42)

    # Pre-calcular lambdas para cada fixture
    fixtures_lambdas = []
    for home_raw, away_raw in remaining_fixtures:
        home = normalize_team_name(home_raw)
        away = normalize_team_name(away_raw)
        s_h = strengths.get(home, {"home_attack": 1.0, "home_defense": 1.0})
        s_a = strengths.get(away, {"away_attack": 1.0, "away_defense": 1.0})
        lh = s_h["home_attack"] * s_a["away_defense"] * avg_home_goals
        la = s_a["away_attack"] * s_h["home_defense"] * avg_away_goals
        fixtures_lambdas.append((home, away, lh, la))

    # Acumuladores por equipo
    teams = sorted(set(t for f in fixtures_lambdas for t in (f[0], f[1])))
    pts_acc = {t: 0.0 for t in teams}
    w_acc = {t: 0.0 for t in teams}
    d_acc = {t: 0.0 for t in teams}
    l_acc = {t: 0.0 for t in teams}
    gf_acc = {t: 0.0 for t in teams}
    ga_acc = {t: 0.0 for t in teams}

    for _ in range(n_simulations):
        for home, away, lh, la in fixtures_lambdas:
            hg = rng.poisson(lh)
            ag = rng.poisson(la)
            gf_acc[home] += hg
            ga_acc[home] += ag
            gf_acc[away] += ag
            ga_acc[away] += hg
            if hg > ag:
                pts_acc[home] += 3
                w_acc[home] += 1
                l_acc[away] += 1
            elif hg == ag:
                pts_acc[home] += 1
                pts_acc[away] += 1
                d_acc[home] += 1
                d_acc[away] += 1
            else:
                pts_acc[away] += 3
                w_acc[away] += 1
                l_acc[home] += 1

    rows = []
    for t in teams:
        rows.append({
            "Team": t,
            "Pts_avg": round(pts_acc[t] / n_simulations, 2),
            "W_avg": round(w_acc[t] / n_simulations, 2),
            "D_avg": round(d_acc[t] / n_simulations, 2),
            "L_avg": round(l_acc[t] / n_simulations, 2),
            "GF_avg": round(gf_acc[t] / n_simulations, 2),
            "GA_avg": round(ga_acc[t] / n_simulations, 2),
        })

    return pd.DataFrame(rows).sort_values("Pts_avg", ascending=False).reset_index(drop=True)


# ─── Interfaz de línea de comandos ───────────────────────────────────────────

def print_prediction(result: dict) -> None:
    """Imprime el resultado de una predicción de forma legible."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SIMULACION MONTE CARLO: {result['home_team']} vs {result['away_team']}")
    print(f"  ({result['n_simulations']:,} simulaciones)")
    print(sep)

    p = result["probabilities"]
    print(f"\n  Probabilidades combinadas (Poisson + MC):")
    print(f"    Victoria Local : {p['home_win']*100:5.1f}%")
    print(f"    Empate         : {p['draw']*100:5.1f}%")
    print(f"    Victoria Visit.: {p['away_win']*100:5.1f}%")

    eg = result["expected_goals"]
    print(f"\n  Goles esperados:")
    print(f"    {result['home_team']}: {eg['home']:.2f}")
    print(f"    {result['away_team']}: {eg['away']:.2f}")
    print(f"    Total          : {eg['total']:.2f}")

    ou = result["over_under"]
    print(f"\n  Over/Under 2.5:")
    print(f"    Más de 2.5 goles  : {ou['over_2_5']*100:.1f}%")
    print(f"    Menos de 2.5 goles: {ou['under_2_5']*100:.1f}%")
    print(f"  Ambos marcan (BTTS) : {result['btts']*100:.1f}%")

    print(f"\n  Top marcadores más probables:")
    for s in result["top_scores"]:
        print(f"    {s['score']}  →  {s['probability']*100:.2f}%")

    print(f"\n  Prediccion final : {result['predicted_result_label']} "
          f"({result['confidence']*100:.1f}%)")
    print(f"{sep}\n")

def print_json_prediction(home_team: str,
    away_team: str) -> None:
    """Imprime el resultado de la predicción en formato JSON."""
    result = simulate_match(home_team, away_team)
    
    return {
            "Equipo_local": result['home_team'],
            "Equipo_visitante": result['away_team'],
            "Victoria_local": f"{result['probabilities']['home_win']*100:.1f}%",
            "Empate": f"{result['probabilities']['draw']*100:.1f}%",
            "Victoria_visitante": f"{result['probabilities']['away_win']*100:.1f}%",
            "Goles_esperados_local": result['expected_goals']['home'],
            "Goles_esperados_visitante": result['expected_goals']['away'],
            "Over_2.5": f"{result['over_under']['over_2_5']*100:.1f}%",
            "Under_2.5": f"{result['over_under']['under_2_5']*100:.1f}%",
            "BTTS": f"{result['btts']*100:.1f}%",
            "Prediccion_final": result['predicted_result_label'],
            "Confianza": f"{result['confidence']*100:.1f}%",
    }

if __name__ == "__main__":
    print("Cargando datos...")
    data = load_all_data()
    print(f"  {len(data)} partidos cargados.\n")

    print_prediction(simulate_match("Manchester City", "Cristal palace", df=data))
