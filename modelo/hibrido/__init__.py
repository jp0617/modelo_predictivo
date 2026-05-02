"""
Sistema Híbrido de Predicción de Premier League

Módulos:
- feature_engineering: Cálculo de características con promedios móviles
- logit_model: Regresión logística para predicción de resultado
- poisson_model: Distribución de Poisson para goles y córners
- hybrid_predict: Sistema principal que combina todos los modelos
"""

__version__ = "1.0.0"
__author__ = "Databricks ML Team"

from .feature_engineering import get_match_features, get_team_recent_stats
from .logit_model import predict_match_result, train_logit_model
from .poisson_model import predict_goals_distribution, predict_corners_distribution
from .hybrid_predict import predict_match, get_available_teams

__all__ = [
    "get_match_features",
    "get_team_recent_stats",
    "predict_match_result",
    "train_logit_model",
    "predict_goals_distribution",
    "predict_corners_distribution",
    "predict_match",
    "get_available_teams",
]
