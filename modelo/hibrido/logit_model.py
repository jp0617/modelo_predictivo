"""
Modelo de Regresión Logística Multinomial

Predice la probabilidad de Victoria Local / Empate / Victoria Visitante
usando características de promedios móviles de los últimos 5 partidos.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

# Agregar el directorio padre al path
sys.path.insert(0, os.getcwd())

# Ruta del volumen de Unity Catalog para modelos entrenados
MODELS_VOLUME_PATH = "/Volumes/workspace/gold/premier_modelos_entrenados"

try:
    from feature_engineering import load_data_from_catalog, calculate_rolling_stats
except ImportError:
    print("⚠ No se pudo importar feature_engineering. Asegúrate de ejecutar desde el directorio correcto.")


def prepare_training_data():
    """
    Prepara datos de entrenamiento para el modelo logístico.
    
    Returns:
        X: DataFrame con características
        y: Series con resultado (H/D/A)
    """
    # Cargar datos con features
    df = load_data_from_catalog()
    df_features = calculate_rolling_stats(df)
    
    # Convertir a pandas
    df_pd = df_features.toPandas()
    
    # Filtrar solo partidos con historial suficiente (al menos 3 partidos previos)
    df_pd = df_pd[df_pd["n_matches"] >= 3].copy()
    
    # Crear dataset por partido (necesitamos home vs away)
    # Primero, obtenemos todos los partidos únicos
    from pyspark.sql import functions as F
    
    df_original = load_data_from_catalog()
    
    # Crear identificador único de partido
    df_matches = df_original.withColumn(
        "match_id",
        F.concat_ws("_", F.col("MatchDate"), F.col("Season"))
    )
    
    # Obtener pares Home-Away
    df_home = df_matches.filter(F.col("Venue") == "Home").select(
        F.col("match_id"),
        F.col("MatchDate"),
        F.col("Team").alias("HomeTeam"),
        F.col("Goals").alias("HomeGoals"),
        F.col("FullTimeResult")
    )
    
    df_away = df_matches.filter(F.col("Venue") == "Away").select(
        F.col("match_id"),
        F.col("Team").alias("AwayTeam"),
        F.col("Goals").alias("AwayGoals")
    )
    
    # Join para obtener partidos completos
    df_complete = df_home.join(df_away, on="match_id", how="inner")
    df_complete_pd = df_complete.toPandas()
    
    # Merge con features
    df_features_pd = df_features.toPandas()
    
    # Features para home
    df_features_home = df_features_pd[df_features_pd["Venue"] == "Home"].copy()
    df_features_home.columns = ["Season", "MatchDate", "HomeTeam", "Venue_h", "Goals_h", "HTGoals_h", 
                                  "Corners_h", "Shots_h", "Result_h", "home_avg_goals", "home_avg_htgoals",
                                  "home_avg_corners", "home_avg_shots", "home_win_rate", "home_n_matches"]
    
    # Features para away
    df_features_away = df_features_pd[df_features_pd["Venue"] == "Away"].copy()
    df_features_away.columns = ["Season", "MatchDate", "AwayTeam", "Venue_a", "Goals_a", "HTGoals_a", 
                                  "Corners_a", "Shots_a", "Result_a", "away_avg_goals", "away_avg_htgoals",
                                  "away_avg_corners", "away_avg_shots", "away_win_rate", "away_n_matches"]
    
    # Merge
    df_complete_pd["MatchDate"] = pd.to_datetime(df_complete_pd["MatchDate"])
    df_features_home["MatchDate"] = pd.to_datetime(df_features_home["MatchDate"])
    df_features_away["MatchDate"] = pd.to_datetime(df_features_away["MatchDate"])
    
    df_train = df_complete_pd.merge(
        df_features_home[["MatchDate", "HomeTeam", "home_avg_goals", "home_avg_corners", 
                          "home_avg_shots", "home_win_rate", "home_n_matches"]],
        on=["MatchDate", "HomeTeam"],
        how="left"
    ).merge(
        df_features_away[["MatchDate", "AwayTeam", "away_avg_goals", "away_avg_corners",
                          "away_avg_shots", "away_win_rate", "away_n_matches"]],
        on=["MatchDate", "AwayTeam"],
        how="left"
    )
    
    # Remover NaNs
    df_train = df_train.dropna()
    
    # Crear características diferenciales
    df_train["goal_diff"] = df_train["home_avg_goals"] - df_train["away_avg_goals"]
    df_train["win_rate_diff"] = df_train["home_win_rate"] - df_train["away_win_rate"]
    df_train["corner_diff"] = df_train["home_avg_corners"] - df_train["away_avg_corners"]
    df_train["shot_diff"] = df_train["home_avg_shots"] - df_train["away_avg_shots"]
    
    # Features para el modelo
    feature_cols = [
        "home_avg_goals", "home_win_rate", "home_avg_corners", "home_avg_shots",
        "away_avg_goals", "away_win_rate", "away_avg_corners", "away_avg_shots",
        "goal_diff", "win_rate_diff", "corner_diff", "shot_diff"
    ]
    
    X = df_train[feature_cols]
    y = df_train["FullTimeResult"]
    
    return X, y, feature_cols


def train_logit_model():
    """
    Entrena el modelo de regresión logística multinomial.
    """
    print("Preparando datos de entrenamiento...")
    X, y, feature_cols = prepare_training_data()
    
    print(f"Dataset: {len(X)} partidos")
    print(f"Distribución de resultados:\n{y.value_counts()}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo logístico multinomial
    print("\nEntrenando Regresión Logística Multinomial...")
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluación
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"{'='*50}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"]))
    
    # Guardar modelo en Unity Catalog Volume
    print(f"\nGuardando modelos en Unity Catalog Volume: {MODELS_VOLUME_PATH}")
    joblib.dump(model, os.path.join(MODELS_VOLUME_PATH, "logit_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_VOLUME_PATH, "logit_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_VOLUME_PATH, "logit_features.pkl"))
    
    print(f"✓ Modelo guardado en {MODELS_VOLUME_PATH}/logit_model.pkl")
    print(f"✓ Scaler guardado en {MODELS_VOLUME_PATH}/logit_scaler.pkl")
    print(f"✓ Features guardadas en {MODELS_VOLUME_PATH}/logit_features.pkl")
    
    return model, scaler, feature_cols


def predict_match_result(features_dict):
    """
    Predice el resultado de un partido usando el modelo logístico.
    
    Args:
        features_dict: dict con características del partido
        
    Returns:
        dict con probabilidades {"H": prob_home_win, "D": prob_draw, "A": prob_away_win}
    """
    # Cargar modelo desde Unity Catalog Volume
    model = joblib.load(os.path.join(MODELS_VOLUME_PATH, "logit_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_VOLUME_PATH, "logit_scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_VOLUME_PATH, "logit_features.pkl"))
    
    # Preparar features
    X = pd.DataFrame([features_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Predicción
    proba = model.predict_proba(X_scaled)[0]
    classes = model.classes_
    
    result = {cls: float(prob) for cls, prob in zip(classes, proba)}
    
    return result


if __name__ == "__main__":
    print("Entrenando modelo de Regresión Logística Multinomial...")
    print("="*60)
    
    try:
        train_logit_model()
        print("\n✓ Entrenamiento completado exitosamente")
    except Exception as e:
        print(f"\n✗ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
