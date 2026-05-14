import pandas as pd
import numpy as np
import joblib
import os
import sys
from rapidfuzz import process, fuzz, utils as fuzz_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.normalize import normalize_df

N_GAMES = 5
FUZZY_THRESHOLD = 75  # score mínimo para aceptar una coincidencia

_model = None
_le = None
_feature_cols = None
_shots_medians = None
_known_teams = None


def _load_artifacts():
    global _model, _le, _feature_cols, _shots_medians
    model_dir = os.path.dirname(__file__)
    _model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    _le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    _feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
    _shots_medians = joblib.load(os.path.join(model_dir, 'shots_medians.pkl'))


def _build_team_stats():
    """Construye el historial reciente de cada equipo a partir de los CSVs."""
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    df1 = normalize_df(pd.read_csv(os.path.join(base, 'epl_final.csv')))
    df2 = normalize_df(pd.read_csv(os.path.join(base, 'PL_2025_actual.csv')))
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)

    home_history = {}
    away_history = {}

    for _, row in df.iterrows():
        ht = row['HomeTeam']
        at = row['AwayTeam']

        home_history.setdefault(ht, []).append({
            'goals_scored': row['FullTimeHomeGoals'],
            'goals_conceded': row['FullTimeAwayGoals'],
            'shots': row['HomeShotsOnTarget'] if not pd.isna(row.get('HomeShotsOnTarget', float('nan'))) else None,
            'result': row['FullTimeResult'],
            'win_val': 'H',
        })
        away_history.setdefault(at, []).append({
            'goals_scored': row['FullTimeAwayGoals'],
            'goals_conceded': row['FullTimeHomeGoals'],
            'shots': row['AwayShotsOnTarget'] if not pd.isna(row.get('AwayShotsOnTarget', float('nan'))) else None,
            'result': row['FullTimeResult'],
            'win_val': 'A',
        })

    return home_history, away_history


def _compute_stats(history, win_val, n=N_GAMES):
    recent = history[-n:] if len(history) >= n else history
    if not recent:
        return None

    goals_avg = np.mean([h['goals_scored'] for h in recent])
    conceded_avg = np.mean([h['goals_conceded'] for h in recent])
    shots_list = [h['shots'] for h in recent if h['shots'] is not None]
    shots_avg = np.mean(shots_list) if shots_list else None
    win_rate = np.mean([1 if h['result'] == win_val else 0 for h in recent])
    n_played = len(recent)

    return {
        'goals_avg': goals_avg,
        'conceded_avg': conceded_avg,
        'shots_avg': shots_avg,
        'win_rate': win_rate,
        'n_played': n_played,
    }


def get_available_teams():
    home, away = _get_histories()
    return sorted(set(list(home.keys()) + list(away.keys())))


_home_history = None
_away_history = None


def _get_histories():
    global _home_history, _away_history, _known_teams
    if _home_history is None:
        _home_history, _away_history = _build_team_stats()
        _known_teams = sorted(set(list(_home_history.keys()) + list(_away_history.keys())))
    return _home_history, _away_history


def _resolve_team(name: str) -> str:
    """Devuelve el nombre canónico más parecido al texto recibido (case-insensitive)."""
    match, score, _ = process.extractOne(
        name, _known_teams,
        scorer=fuzz.WRatio,
        processor=fuzz_utils.default_process,
    )
    if score < FUZZY_THRESHOLD:
        raise ValueError(
            f"No se encontró ningún equipo parecido a '{name}'. "
            f"Mejor coincidencia: '{match}' ({score:.0f}%)."
        )
    return match


def team_stats(team_name: str) -> dict:
    """Devuelve las estadísticas recientes de un equipo (local y visitante)."""
    home_hist, away_hist = _get_histories()

    resolved = _resolve_team(team_name)

    home_s = _compute_stats(home_hist.get(resolved, []), win_val='H')
    away_s = _compute_stats(away_hist.get(resolved, []), win_val='A')

    def fmt(s):
        if s is None:
            return None
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}

    return {
        'team': resolved,
        'home': fmt(home_s),
        'away': fmt(away_s),
    }


def top5() -> list:
    """Devuelve los 5 equipos con mayor tasa de victorias reciente (local + visitante)."""
    home_hist, away_hist = _get_histories()
    all_teams = sorted(set(list(home_hist.keys()) + list(away_hist.keys())))

    rankings = []
    for team in all_teams:
        home_s = _compute_stats(home_hist.get(team, []), win_val='H')
        away_s = _compute_stats(away_hist.get(team, []), win_val='A')

        rates = [s['win_rate'] for s in [home_s, away_s] if s is not None]
        if not rates:
            continue

        overall_win_rate = round(sum(rates) / len(rates), 4)
        rankings.append({
            'team': team,
            'overall_win_rate': overall_win_rate,
            'home_win_rate': round(home_s['win_rate'], 4) if home_s else None,
            'away_win_rate': round(away_s['win_rate'], 4) if away_s else None,
        })

    rankings.sort(key=lambda x: x['overall_win_rate'], reverse=True)
    return rankings[:5]


def init():
    """Carga el modelo y precalcula historial. Llamar al iniciar la app."""
    _load_artifacts()
    _get_histories()


def predict(home_team: str, away_team: str) -> dict:
    if _model is None:
        init()

    _get_histories()

    home_resolved = _resolve_team(home_team)
    away_resolved = _resolve_team(away_team)

    home_hist, away_hist = _get_histories()
    home_h = home_hist.get(home_resolved)
    away_h = away_hist.get(away_resolved)

    home_s = _compute_stats(home_h, win_val='H')
    away_s = _compute_stats(away_h, win_val='A')

    row = {
        'home_goals_avg': home_s['goals_avg'],
        'home_conceded_avg': home_s['conceded_avg'],
        'home_shots_avg': home_s['shots_avg'] if home_s['shots_avg'] is not None else _shots_medians['home_shots_avg'],
        'home_win_rate': home_s['win_rate'],
        'home_n_played': home_s['n_played'],
        'away_goals_avg': away_s['goals_avg'],
        'away_conceded_avg': away_s['conceded_avg'],
        'away_shots_avg': away_s['shots_avg'] if away_s['shots_avg'] is not None else _shots_medians['away_shots_avg'],
        'away_win_rate': away_s['win_rate'],
        'away_n_played': away_s['n_played'],
    }

    X = pd.DataFrame([row])[_feature_cols]
    proba = _model.predict_proba(X)[0]
    classes = _le.classes_

    result = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    return {
        'Equipo_local': home_resolved,
        'Equipo_visitante': away_resolved,
        'Probabilidades': {
            'Victoria local': f"{result.get('H', 0)*100:.1f}%",
            'Empate': f"{result.get('D', 0)*100:.1f}%",
            'Victoria visitante': f"{result.get('A', 0)*100:.1f}%",
        },
        'Estadísticas recientes del equipo local': {k: round(v, 3) if v is not None else None for k, v in home_s.items()},
        'Estadísticas recientes del equipo visitante': {k: round(v, 3) if v is not None else None for k, v in away_s.items()},
    }
