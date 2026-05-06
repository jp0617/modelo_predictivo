"""
normalize.py — Normalización de nombres de equipos de la Premier League

Convierte cualquier variante de nombre (nombre corto del histórico, nombre
de la API, etc.) al formato oficial completo: "Manchester City FC".

Uso típico (se llama automáticamente desde train.py y predict.py):

    from data.normalize import normalize_df

    df = pd.read_csv(...)
    df = normalize_df(df)           # normaliza HomeTeam y AwayTeam in-place

También puede ejecutarse directamente para normalizar los CSVs en disco:

    python data/normalize.py
"""

import os
import pandas as pd

# Mapa exhaustivo: nombre raw → nombre oficial
TEAM_NAME_MAP: dict[str, str] = {
    # ── Nombres cortos del histórico (epl_final.csv) ──────────────────────
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
    # ── Variantes que puede devolver la API de football-data.org ──────────
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Newcastle United": "Newcastle United FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "West Ham United": "West Ham United FC",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Aston Villa": "Aston Villa FC",
    "Leicester City": "Leicester City FC",
    "Leeds United": "Leeds United FC",
    "Sheffield Utd": "Sheffield United FC",
    "Nott'm Forest": "Nottingham Forest FC",
    # ── Nombres ya correctos (identidad, evita KeyError) ──────────────────
    "Arsenal FC": "Arsenal FC",
    "Aston Villa FC": "Aston Villa FC",
    "AFC Bournemouth": "AFC Bournemouth",
    "Brentford FC": "Brentford FC",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion FC",
    "Burnley FC": "Burnley FC",
    "Chelsea FC": "Chelsea FC",
    "Crystal Palace FC": "Crystal Palace FC",
    "Everton FC": "Everton FC",
    "Fulham FC": "Fulham FC",
    "Leeds United FC": "Leeds United FC",
    "Leicester City FC": "Leicester City FC",
    "Liverpool FC": "Liverpool FC",
    "Luton Town FC": "Luton Town FC",
    "Manchester City FC": "Manchester City FC",
    "Manchester United FC": "Manchester United FC",
    "Middlesbrough FC": "Middlesbrough FC",
    "Newcastle United FC": "Newcastle United FC",
    "Nottingham Forest FC": "Nottingham Forest FC",
    "Queens Park Rangers FC": "Queens Park Rangers FC",
    "Sheffield United FC": "Sheffield United FC",
    "Southampton FC": "Southampton FC",
    "Sunderland AFC": "Sunderland AFC",
    "Tottenham Hotspur FC": "Tottenham Hotspur FC",
    "Watford FC": "Watford FC",
    "West Bromwich Albion FC": "West Bromwich Albion FC",
    "West Ham United FC": "West Ham United FC",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers FC",
    "Ipswich Town FC": "Ipswich Town FC",
    "Huddersfield Town FC": "Huddersfield Town FC",
    "Hull City FC": "Hull City FC",
    "Birmingham City FC": "Birmingham City FC",
    "Blackburn Rovers FC": "Blackburn Rovers FC",
    "Bolton Wanderers FC": "Bolton Wanderers FC",
    "Bradford City FC": "Bradford City FC",
    "Cardiff City FC": "Cardiff City FC",
    "Charlton Athletic FC": "Charlton Athletic FC",
    "Coventry City FC": "Coventry City FC",
    "Derby County FC": "Derby County FC",
    "Norwich City FC": "Norwich City FC",
    "Portsmouth FC": "Portsmouth FC",
    "Reading FC": "Reading FC",
    "Stoke City FC": "Stoke City FC",
    "Swansea City FC": "Swansea City FC",
    "Wigan Athletic FC": "Wigan Athletic FC",
    "Blackpool FC": "Blackpool FC",
}


def normalize_name(name: str) -> str:
    """
    Devuelve el nombre oficial de un equipo.
    Si no está en el mapa, devuelve el nombre tal como viene (sin crash).
    """
    if pd.isna(name):
        return name
    clean = str(name).strip()
    return TEAM_NAME_MAP.get(clean, clean)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las columnas HomeTeam y AwayTeam de un DataFrame.
    Opera sobre una copia para no mutar el original.
    """
    df = df.copy()
    if "HomeTeam" in df.columns:
        df["HomeTeam"] = df["HomeTeam"].apply(normalize_name)
    if "AwayTeam" in df.columns:
        df["AwayTeam"] = df["AwayTeam"].apply(normalize_name)
    return df


# ── Ejecución directa: normaliza los CSVs en disco ───────────────────────────

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))

    files = {
        "epl_final.csv": os.path.join(data_dir, "epl_final.csv"),
        "PL_2025_actual.csv": os.path.join(data_dir, "PL_2025_actual.csv"),
    }

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"  [skip] {name} no encontrado")
            continue

        df = pd.read_csv(path)
        original_teams = set(df.get("HomeTeam", pd.Series()).tolist() +
                             df.get("AwayTeam", pd.Series()).tolist())

        df = normalize_df(df)
        df.to_csv(path, index=False)

        normalized_teams = set(df.get("HomeTeam", pd.Series()).tolist() +
                               df.get("AwayTeam", pd.Series()).tolist())
        changed = {t for t in original_teams
                   if TEAM_NAME_MAP.get(str(t).strip(), str(t).strip()) != str(t).strip()}

        print(f"  {name}: {len(df)} filas, {len(changed)} nombres cambiados")

    print("Normalizacion completada.")
