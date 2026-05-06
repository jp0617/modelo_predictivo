"""
Feature Engineering para Sistema Híbrido de Predicción de Premier League

Calcula promedios móviles de los últimos N partidos para cada equipo,
usando datos de Unity Catalog (workspace.gold.equipos_premier).
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd

N_GAMES = 5  # Número de partidos históricos para promedios móviles


def get_spark():
    """Obtiene la sesión de Spark activa."""
    return SparkSession.builder.getOrCreate()


def load_data_from_catalog():
    """Carga datos desde Unity Catalog."""
    spark = get_spark()
    df = spark.table("workspace.gold.equipos_premier")
    
    # Filtrar solo partidos jugados (con datos completos)
    df = df.filter(F.col("Goals").isNotNull())
    
    # Convertir FullTimeResult desde perspectiva del equipo
    # H = Victoria como local, A = Victoria como visitante, D = Empate
    df = df.withColumn(
        "Result",
        F.when(
            (F.col("Venue") == "Home") & (F.col("FullTimeResult") == "H"), "W"
        ).when(
            (F.col("Venue") == "Away") & (F.col("FullTimeResult") == "A"), "W"
        ).when(
            F.col("FullTimeResult") == "D", "D"
        ).otherwise("L")
    )
    
    return df


def calculate_rolling_stats(df, n_games=N_GAMES):
    """
    Calcula estadísticas móviles de los últimos n_games para cada equipo.
    
    Returns:
        Spark DataFrame con columnas:
        - Team
        - MatchDate
        - Venue (Home/Away)
        - Goals (del partido actual)
        - avg_goals_scored (promedio de últimos n partidos)
        - avg_goals_conceded (promedio de últimos n partidos)
        - avg_corners (promedio de últimos n partidos)
        - win_rate (tasa de victoria últimos n partidos)
        - n_matches (número de partidos usados para el promedio)
    """
    
    # Ventana particionada por equipo, ordenada por fecha
    window_spec = Window.partitionBy("Team").orderBy("MatchDate").rowsBetween(-n_games, -1)
    
    # Calcular estadísticas móviles
    df_features = df.withColumn(
        "avg_goals_scored",
        F.avg("Goals").over(window_spec)
    ).withColumn(
        "avg_htgoals",
        F.avg("HTGoals").over(window_spec)
    ).withColumn(
        # Conteo de victorias en ventana móvil
        "wins_in_window",
        F.sum(F.when(F.col("Result") == "W", 1).otherwise(0)).over(window_spec)
    ).withColumn(
        # Número de partidos en la ventana
        "n_matches",
        F.count("*").over(window_spec)
    ).withColumn(
        # Tasa de victoria
        "win_rate",
        F.col("wins_in_window") / F.col("n_matches")
    )
    
    # Convertir corners/shots a numérico (actualmente son strings)
    df_features = df_features.withColumn(
        "Corners_num",
        F.when(F.col("Corners").isNotNull(), F.col("Corners").cast("int")).otherwise(0)
    ).withColumn(
        "ShotsOnTarget_num",
        F.when(F.col("ShotsOnTarget").isNotNull(), F.col("ShotsOnTarget").cast("int")).otherwise(0)
    )
    
    # Promedio de corners y tiros
    df_features = df_features.withColumn(
        "avg_corners",
        F.avg("Corners_num").over(window_spec)
    ).withColumn(
        "avg_shots_on_target",
        F.avg("ShotsOnTarget_num").over(window_spec)
    )
    
    # Seleccionar columnas relevantes
    return df_features.select(
        "Season",
        "MatchDate",
        "Team",
        "Venue",
        "Goals",
        "HTGoals",
        "Corners_num",
        "ShotsOnTarget_num",
        "Result",
        "avg_goals_scored",
        "avg_htgoals",
        "avg_corners",
        "avg_shots_on_target",
        "win_rate",
        "n_matches"
    )


def get_team_recent_stats(team_name, venue="Home", n_games=N_GAMES):
    """
    Obtiene las estadísticas recientes de un equipo específico.
    
    Args:
        team_name: Nombre del equipo
        venue: "Home" o "Away"
        n_games: Número de partidos históricos a considerar
    
    Returns:
        dict con estadísticas del equipo
    """
    df = load_data_from_catalog()
    df_features = calculate_rolling_stats(df, n_games)
    
    # Filtrar por equipo y venue
    team_df = df_features.filter(
        (F.col("Team") == team_name) & (F.col("Venue") == venue)
    ).orderBy(F.col("MatchDate").desc()).limit(1)
    
    # Convertir a pandas para facilitar manejo
    team_stats = team_df.toPandas()
    
    if team_stats.empty:
        return None
    
    row = team_stats.iloc[0]
    
    return {
        "team": team_name,
        "venue": venue,
        "avg_goals_scored": float(row["avg_goals_scored"]) if pd.notna(row["avg_goals_scored"]) else 0.0,
        "avg_htgoals": float(row["avg_htgoals"]) if pd.notna(row["avg_htgoals"]) else 0.0,
        "avg_corners": float(row["avg_corners"]) if pd.notna(row["avg_corners"]) else 0.0,
        "avg_shots_on_target": float(row["avg_shots_on_target"]) if pd.notna(row["avg_shots_on_target"]) else 0.0,
        "win_rate": float(row["win_rate"]) if pd.notna(row["win_rate"]) else 0.0,
        "n_matches": int(row["n_matches"]) if pd.notna(row["n_matches"]) else 0
    }


def get_match_features(home_team, away_team):
    """
    Prepara las características para un partido entre dos equipos.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
    
    Returns:
        dict con características del partido
    """
    home_stats = get_team_recent_stats(home_team, "Home")
    away_stats = get_team_recent_stats(away_team, "Away")
    
    if home_stats is None or away_stats is None:
        raise ValueError(f"No hay suficientes datos históricos para {home_team} o {away_team}")
    
    # Crear características comparativas
    features = {
        "home_team": home_team,
        "away_team": away_team,
        "home_avg_goals": home_stats["avg_goals_scored"],
        "home_win_rate": home_stats["win_rate"],
        "home_avg_corners": home_stats["avg_corners"],
        "home_avg_shots": home_stats["avg_shots_on_target"],
        "away_avg_goals": away_stats["avg_goals_scored"],
        "away_win_rate": away_stats["win_rate"],
        "away_avg_corners": away_stats["avg_corners"],
        "away_avg_shots": away_stats["avg_shots_on_target"],
        # Características diferenciales (ventaja relativa)
        "goal_diff": home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
        "win_rate_diff": home_stats["win_rate"] - away_stats["win_rate"],
        "corner_diff": home_stats["avg_corners"] - away_stats["avg_corners"],
        "shot_diff": home_stats["avg_shots_on_target"] - away_stats["avg_shots_on_target"],
        "home_n_matches": home_stats["n_matches"],
        "away_n_matches": away_stats["n_matches"]
    }
    
    return features


if __name__ == "__main__":
    # Test
    print("Testing feature engineering...")
    
    try:
        features = get_match_features("AFC Bournemouth", "Crystal Palace FC")
        print("\n✓ Características calculadas exitosamente:")
        for k, v in features.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"✗ Error: {e}")
