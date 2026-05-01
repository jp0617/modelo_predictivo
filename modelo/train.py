import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

N_GAMES = 5  # últimos partidos para calcular estadísticas


def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    df1 = pd.read_csv(os.path.join(base, 'epl_final.csv'))
    df2 = pd.read_csv(os.path.join(base, 'PL_2025_actual.csv'))
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)
    # Conservar todos los partidos para construir historial, pero marcar los jugados
    df['_played'] = df['FullTimeResult'].notna()
    return df


def rolling_team_stats(df, team_col, goals_scored_col, goals_conceded_col,
                        shots_col, result_win_val, n=N_GAMES):
    """Para cada partido, calcula las medias de los últimos n partidos del equipo."""
    records = []
    team_history = {}

    for _, row in df.iterrows():
        team = row[team_col]
        history = team_history.get(team, [])

        if len(history) >= n:
            recent = history[-n:]
        else:
            recent = history

        if recent:
            goals_avg = np.mean([h['goals_scored'] for h in recent])
            conceded_avg = np.mean([h['goals_conceded'] for h in recent])
            shots_avg = np.mean([h['shots'] for h in recent]) if any(h['shots'] is not None for h in recent) else np.nan
            win_rate = np.mean([1 if h['result'] == result_win_val else 0 for h in recent])
            n_played = len(recent)
        else:
            goals_avg = np.nan
            conceded_avg = np.nan
            shots_avg = np.nan
            win_rate = np.nan
            n_played = 0

        records.append({
            'goals_avg': goals_avg,
            'conceded_avg': conceded_avg,
            'shots_avg': shots_avg,
            'win_rate': win_rate,
            'n_played': n_played,
        })

        shots_val = row.get(shots_col)
        if pd.isna(shots_val):
            shots_val = None

        entry = {
            'goals_scored': row[goals_scored_col],
            'goals_conceded': row[goals_conceded_col],
            'shots': shots_val,
            'result': row['FullTimeResult'],
        }
        team_history[team] = team_history.get(team, []) + [entry]

    return pd.DataFrame(records)


def build_features(df):
    home_stats = rolling_team_stats(
        df,
        team_col='HomeTeam',
        goals_scored_col='FullTimeHomeGoals',
        goals_conceded_col='FullTimeAwayGoals',
        shots_col='HomeShotsOnTarget',
        result_win_val='H',
    )
    home_stats.columns = ['home_' + c for c in home_stats.columns]

    away_stats = rolling_team_stats(
        df,
        team_col='AwayTeam',
        goals_scored_col='FullTimeAwayGoals',
        goals_conceded_col='FullTimeHomeGoals',
        shots_col='AwayShotsOnTarget',
        result_win_val='A',
    )
    away_stats.columns = ['away_' + c for c in away_stats.columns]

    features = pd.concat([home_stats, away_stats], axis=1)
    features['target'] = df['FullTimeResult'].values
    features['_played'] = df['_played'].values
    return features


def train():
    df = load_data()
    features = build_features(df)

    feature_cols = [c for c in features.columns if c not in ('target', '_played')]
    X = features[feature_cols]
    y = features['target']
    played = df['_played'].values

    # Solo entrenar con partidos jugados y que tengan historial previo
    mask = played & (features['home_n_played'] >= 1) & (features['away_n_played'] >= 1)
    X = X[mask]
    y = y[mask]

    # Rellenar NaN en shots con la media
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    model_dir = os.path.dirname(__file__)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))

    # Guardar medias de shots para imputar en predicción
    shots_medians = {
        'home_shots_avg': X['home_shots_avg'].median(),
        'away_shots_avg': X['away_shots_avg'].median(),
    }
    joblib.dump(shots_medians, os.path.join(model_dir, 'shots_medians.pkl'))

    print("Modelo guardado en modelo/model.pkl")
    return model, le, feature_cols, shots_medians


if __name__ == '__main__':
    train()
