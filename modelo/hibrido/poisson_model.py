"""
Modelo de Distribución de Poisson

Predice la probabilidad de número de goles y córners usando distribución de Poisson,
basándose en las tasas promedio de ataque y defensa de cada equipo.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import joblib
import os
import sys

# Agregar el directorio padre al path
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # __file__ no está definido en notebooks de Databricks, usar ruta absoluta
    sys.path.insert(0, "/Users/mago-dios666@hotmail.com/modelo_predictivo/modelo/hibrido")

# Ruta del volumen de Unity Catalog para modelos entrenados
MODELS_VOLUME_PATH = "/Volumes/workspace/gold/premier_modelos_entrenados"

try:
    from feature_engineering import load_data_from_catalog, calculate_rolling_stats
except ImportError:
    print("⚠ No se pudo importar feature_engineering. Asegúrate de ejecutar desde el directorio correcto.")


def calculate_team_strengths():
    """
    Calcula las fortalezas ofensivas y defensivas de cada equipo
    basándose en datos históricos.
    
    Returns:
        dict con estadísticas de ataque y defensa por equipo
    """
    df = load_data_from_catalog()
    df_pd = df.toPandas()
    
    # Calcular promedios globales
    avg_home_goals = df_pd[df_pd["Venue"] == "Home"]["Goals"].mean()
    avg_away_goals = df_pd[df_pd["Venue"] == "Away"]["Goals"].mean()
    
    # Calcular fortaleza de ataque (goals scored / promedio)
    team_stats = {}
    
    for team in df_pd["Team"].unique():
        team_data = df_pd[df_pd["Team"] == team]
        
        # Stats como local
        home_data = team_data[team_data["Venue"] == "Home"]
        home_attack = home_data["Goals"].mean() / avg_home_goals if len(home_data) > 0 else 1.0
        
        # Stats como visitante
        away_data = team_data[team_data["Venue"] == "Away"]
        away_attack = away_data["Goals"].mean() / avg_away_goals if len(away_data) > 0 else 1.0
        
        team_stats[team] = {
            "home_attack": home_attack,
            "away_attack": away_attack,
            "home_goals_avg": home_data["Goals"].mean() if len(home_data) > 0 else avg_home_goals,
            "away_goals_avg": away_data["Goals"].mean() if len(away_data) > 0 else avg_away_goals,
        }
    
    return team_stats, avg_home_goals, avg_away_goals


def save_poisson_parameters():
    """
    Calcula y guarda los parámetros necesarios para el modelo de Poisson.
    """
    team_stats, avg_home_goals, avg_away_goals = calculate_team_strengths()
    
    params = {
        "team_stats": team_stats,
        "avg_home_goals": avg_home_goals,
        "avg_away_goals": avg_away_goals
    }
    
    # Guardar en Unity Catalog Volume
    print(f"Guardando parámetros en Unity Catalog Volume: {MODELS_VOLUME_PATH}")
    joblib.dump(params, os.path.join(MODELS_VOLUME_PATH, "poisson_params.pkl"))
    
    print("✓ Parámetros de Poisson guardados")
    print(f"  - Promedio goles local: {avg_home_goals:.2f}")
    print(f"  - Promedio goles visitante: {avg_away_goals:.2f}")
    print(f"  - Equipos procesados: {len(team_stats)}")
    print(f"  - Ubicación: {MODELS_VOLUME_PATH}/poisson_params.pkl")
    
    return params


def load_poisson_parameters():
    """Carga los parámetros guardados del modelo de Poisson desde Unity Catalog Volume."""
    return joblib.load(os.path.join(MODELS_VOLUME_PATH, "poisson_params.pkl"))


def predict_goals_distribution(home_team, away_team, max_goals=7):
    """
    Predice la distribución de probabilidad de goles para un partido.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
        max_goals: Número máximo de goles a calcular
        
    Returns:
        dict con distribuciones de probabilidad y estadísticas
    """
    params = load_poisson_parameters()
    team_stats = params["team_stats"]
    avg_home = params["avg_home_goals"]
    avg_away = params["avg_away_goals"]
    
    # Obtener fortalezas de los equipos
    home_attack = team_stats.get(home_team, {}).get("home_attack", 1.0)
    away_attack = team_stats.get(away_team, {}).get("away_attack", 1.0)
    
    # Calcular lambda (tasa esperada de goles) usando modelo de Dixon-Coles simplificado
    lambda_home = home_attack * avg_home
    lambda_away = away_attack * avg_away
    
    # Distribución de Poisson para cada equipo
    home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals + 1)]
    
    # Matriz de probabilidades (resultado exacto)
    score_matrix = np.outer(home_probs, away_probs)
    
    # Probabilidades de resultado final
    prob_home_win = np.sum(np.tril(score_matrix, -1))  # Home > Away
    prob_draw = np.sum(np.diag(score_matrix))  # Home == Away
    prob_away_win = np.sum(np.triu(score_matrix, 1))  # Home < Away
    
    # Goles esperados
    expected_home_goals = lambda_home
    expected_away_goals = lambda_away
    
    # Probabilidad de más/menos 2.5 goles totales
    total_goals_probs = {}
    for total in range(max_goals * 2 + 1):
        prob = 0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h + a == total:
                    prob += home_probs[h] * away_probs[a]
        total_goals_probs[total] = prob
    
    prob_over_2_5 = sum(total_goals_probs.get(i, 0) for i in range(3, max_goals * 2 + 1))
    prob_under_2_5 = sum(total_goals_probs.get(i, 0) for i in range(0, 3))
    
    # Resultado más probable
    most_likely_score = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_home_goals": round(expected_home_goals, 2),
        "expected_away_goals": round(expected_away_goals, 2),
        "most_likely_score": f"{most_likely_score[0]}-{most_likely_score[1]}",
        "score_probability": round(float(score_matrix[most_likely_score]), 4),
        "result_probabilities": {
            "home_win": round(float(prob_home_win), 4),
            "draw": round(float(prob_draw), 4),
            "away_win": round(float(prob_away_win), 4)
        },
        "total_goals": {
            "over_2_5": round(float(prob_over_2_5), 4),
            "under_2_5": round(float(prob_under_2_5), 4)
        },
        "home_goals_distribution": {i: round(float(p), 4) for i, p in enumerate(home_probs)},
        "away_goals_distribution": {i: round(float(p), 4) for i, p in enumerate(away_probs)}
    }


def predict_corners_distribution(home_team, away_team):
    """
    Predice la distribución de córners usando modelo de Poisson.
    
    Similar a goles pero usando promedios de córners.
    """
    df = load_data_from_catalog()
    df_pd = df.toPandas()
    
    # Convertir corners a numérico
    df_pd["Corners_num"] = pd.to_numeric(df_pd["Corners"], errors="coerce").fillna(0)
    
    # Calcular promedios por equipo
    home_data = df_pd[(df_pd["Team"] == home_team) & (df_pd["Venue"] == "Home")]
    away_data = df_pd[(df_pd["Team"] == away_team) & (df_pd["Venue"] == "Away")]
    
    lambda_home_corners = home_data["Corners_num"].mean() if len(home_data) > 0 else 5.0
    lambda_away_corners = away_data["Corners_num"].mean() if len(away_data) > 0 else 4.0
    
    expected_total_corners = lambda_home_corners + lambda_away_corners
    
    # Probabilidad de más/menos 10.5 corners
    total_corners_dist = poisson(expected_total_corners)
    prob_over_10_5 = 1 - total_corners_dist.cdf(10)
    prob_under_10_5 = total_corners_dist.cdf(10)
    
    return {
        "expected_home_corners": round(lambda_home_corners, 2),
        "expected_away_corners": round(lambda_away_corners, 2),
        "expected_total_corners": round(expected_total_corners, 2),
        "corners_over_under": {
            "over_10_5": round(float(prob_over_10_5), 4),
            "under_10_5": round(float(prob_under_10_5), 4)
        }
    }


if __name__ == "__main__":
    print("Calculando parámetros del modelo de Poisson...")
    print("="*60)
    
    try:
        save_poisson_parameters()
        print("\n✓ Parámetros calculados y guardados exitosamente")
        
        # Test
        print("\n" + "="*60)
        print("Test de predicción: Arsenal FC vs Chelsea FC")
        print("="*60)
        
        result = predict_goals_distribution("Arsenal FC", "Chelsea FC")
        print(f"\nGoles esperados:")
        print(f"  Arsenal FC: {result['expected_home_goals']}")
        print(f"  Chelsea FC: {result['expected_away_goals']}")
        print(f"\nResultado más probable: {result['most_likely_score']} ({result['score_probability']*100:.1f}%)")
        print(f"\nProbabilidades de resultado:")
        print(f"  Victoria local: {result['result_probabilities']['home_win']*100:.1f}%")
        print(f"  Empate: {result['result_probabilities']['draw']*100:.1f}%")
        print(f"  Victoria visitante: {result['result_probabilities']['away_win']*100:.1f}%")
        
        corners = predict_corners_distribution("Arsenal FC", "Chelsea FC")
        print(f"\nCórners esperados: {corners['expected_total_corners']}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
