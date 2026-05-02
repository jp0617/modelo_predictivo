"""
Sistema Híbrido de Predicción de Partidos de Premier League

Combina tres enfoques:
1. Regresión Logística Multinomial para probabilidades de Victoria/Empate/Derrota
2. Distribución de Poisson para predicción de número de goles y córners
3. Ingeniería de características con promedios móviles de últimos 5 partidos
"""

import sys
import os

# Agregar directorio al path
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # __file__ no está definido en notebooks de Databricks, usar ruta absoluta
    sys.path.insert(0, "/Users/mago-dios666@hotmail.com/modelo_predictivo/modelo/hibrido")

from feature_engineering import get_match_features
from logit_model import predict_match_result
from poisson_model import predict_goals_distribution, predict_corners_distribution


def predict_match(home_team, away_team):
    """
    Realiza una predicción completa de un partido usando el sistema híbrido.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
        
    Returns:
        dict con todas las predicciones del sistema híbrido
    """
    
    print(f"\n{'='*70}")
    print(f"PREDICCIÓN HÍBRIDA: {home_team} vs {away_team}")
    print(f"{'='*70}")
    
    # 1. Obtener características del partido (promedios móviles)
    print("\n[1/3] Calculando características con promedios móviles...")
    try:
        features = get_match_features(home_team, away_team)
        print(f"✓ Características calculadas:")
        print(f"  {home_team} (Local):")
        print(f"    - Goles promedio: {features['home_avg_goals']:.2f}")
        print(f"    - Tasa de victoria: {features['home_win_rate']*100:.1f}%")
        print(f"  {away_team} (Visitante):")
        print(f"    - Goles promedio: {features['away_avg_goals']:.2f}")
        print(f"    - Tasa de victoria: {features['away_win_rate']*100:.1f}%")
    except Exception as e:
        print(f"✗ Error al calcular características: {e}")
        return None
    
    # 2. Predicción de resultado con Regresión Logística
    print("\n[2/3] Predicción de resultado con Regresión Logística Multinomial...")
    try:
        logit_result = predict_match_result(features)
        print(f"✓ Probabilidades (Logit):")
        print(f"  Victoria Local: {logit_result.get('H', 0)*100:.1f}%")
        print(f"  Empate: {logit_result.get('D', 0)*100:.1f}%")
        print(f"  Victoria Visitante: {logit_result.get('A', 0)*100:.1f}%")
    except Exception as e:
        print(f"✗ Error en modelo Logit: {e}")
        logit_result = None
    
    # 3. Predicción de goles con Distribución de Poisson
    print("\n[3/3] Predicción de goles con Distribución de Poisson...")
    try:
        poisson_result = predict_goals_distribution(home_team, away_team)
        print(f"✓ Predicción de goles (Poisson):")
        print(f"  Goles esperados {home_team}: {poisson_result['expected_home_goals']}")
        print(f"  Goles esperados {away_team}: {poisson_result['expected_away_goals']}")
        print(f"  Resultado más probable: {poisson_result['most_likely_score']}")
        print(f"  Probabilidad: {poisson_result['score_probability']*100:.1f}%")
        
        print(f"\n  Probabilidades de resultado (Poisson):")
        print(f"    Victoria Local: {poisson_result['result_probabilities']['home_win']*100:.1f}%")
        print(f"    Empate: {poisson_result['result_probabilities']['draw']*100:.1f}%")
        print(f"    Victoria Visitante: {poisson_result['result_probabilities']['away_win']*100:.1f}%")
        
        print(f"\n  Más/Menos 2.5 goles:")
        print(f"    Más de 2.5: {poisson_result['total_goals']['over_2_5']*100:.1f}%")
        print(f"    Menos de 2.5: {poisson_result['total_goals']['under_2_5']*100:.1f}%")
    except Exception as e:
        print(f"✗ Error en modelo Poisson: {e}")
        poisson_result = None
    
    # 3b. Predicción de córners
    try:
        corners_result = predict_corners_distribution(home_team, away_team)
        print(f"\n  Córners esperados:")
        print(f"    {home_team}: {corners_result['expected_home_corners']}")
        print(f"    {away_team}: {corners_result['expected_away_corners']}")
        print(f"    Total: {corners_result['expected_total_corners']}")
        print(f"    Más de 10.5: {corners_result['corners_over_under']['over_10_5']*100:.1f}%")
    except Exception as e:
        print(f"✗ Error al predecir córners: {e}")
        corners_result = None
    
    # Resultado combinado
    print(f"\n{'='*70}")
    print("RESUMEN - SISTEMA HÍBRIDO")
    print(f"{'='*70}")
    
    if logit_result and poisson_result:
        # Combinar ambos modelos (promedio ponderado: 60% Logit, 40% Poisson)
        combined_home_win = 0.6 * logit_result.get('H', 0) + 0.4 * poisson_result['result_probabilities']['home_win']
        combined_draw = 0.6 * logit_result.get('D', 0) + 0.4 * poisson_result['result_probabilities']['draw']
        combined_away_win = 0.6 * logit_result.get('A', 0) + 0.4 * poisson_result['result_probabilities']['away_win']
        
        print(f"\nProbabilidades combinadas (60% Logit + 40% Poisson):")
        print(f"  🏠 Victoria Local: {combined_home_win*100:.1f}%")
        print(f"  🤝 Empate: {combined_draw*100:.1f}%")
        print(f"  ✈️  Victoria Visitante: {combined_away_win*100:.1f}%")
        
        # Predicción final
        max_prob = max(combined_home_win, combined_draw, combined_away_win)
        if max_prob == combined_home_win:
            prediction = f"Victoria de {home_team}"
        elif max_prob == combined_draw:
            prediction = "Empate"
        else:
            prediction = f"Victoria de {away_team}"
        
        print(f"\n🎯 PREDICCIÓN FINAL: {prediction} ({max_prob*100:.1f}%)")
    
    if poisson_result:
        print(f"\n⚽ Marcador más probable: {poisson_result['most_likely_score']}")
        print(f"📊 Total de goles esperados: {poisson_result['expected_home_goals'] + poisson_result['expected_away_goals']:.2f}")
    
    if corners_result:
        print(f"🚩 Córners totales esperados: {corners_result['expected_total_corners']}")
    
    print(f"{'='*70}\n")
    
    # Construir respuesta estructurada
    response = {
        "match": {
            "home_team": home_team,
            "away_team": away_team
        },
        "features": features,
        "logit_prediction": logit_result,
        "poisson_prediction": poisson_result,
        "corners_prediction": corners_result,
    }
    
    if logit_result and poisson_result:
        response["combined_prediction"] = {
            "home_win": round(combined_home_win, 4),
            "draw": round(combined_draw, 4),
            "away_win": round(combined_away_win, 4),
            "final_prediction": prediction,
            "confidence": round(max_prob, 4)
        }
    
    return response


def get_available_teams():
    """Obtiene la lista de equipos disponibles en el sistema."""
    from feature_engineering import load_data_from_catalog
    
    df = load_data_from_catalog()
    teams = df.select("Team").distinct().toPandas()["Team"].tolist()
    return sorted(teams)


# Este bloque NO se ejecuta automáticamente en Databricks
# Solo muestra instrucciones de uso
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SISTEMA HÍBRIDO DE PREDICCIÓN - Premier League")
    print("="*70)
    print("\n📌 Para hacer predicciones, usa:")
    print("   from hybrid_predict import predict_match")
    print('   predict_match("Arsenal FC", "Chelsea FC")')
    print("\n📌 Para ver equipos disponibles:")
    print("   from hybrid_predict import get_available_teams")
    print("   teams = get_available_teams()")
    print("="*70 + "\n")
