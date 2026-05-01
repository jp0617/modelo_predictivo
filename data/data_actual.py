import requests
import csv
import json
from datetime import datetime

url = "https://api.football-data.org/v4/competitions/PL/matches?season=2025"

headers = {
    'X-Auth-Token': 'd272ac5581b44517a3850d3e374915f8'
}

response = requests.get(url, headers=headers)
data = response.json()

COLUMNS = [
    "Season", "MatchDate", "HomeTeam", "AwayTeam",
    "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
    "HalfTimeHomeGoals", "HalfTimeAwayGoals", "HalfTimeResult",
    "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
    "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls",
    "HomeYellowCards", "AwayYellowCards", "HomeRedCards", "AwayRedCards",
]

WINNER_MAP = {"HOME_TEAM": "H", "AWAY_TEAM": "A", "DRAW": "D"}

STAT_KEYS = {
    "HomeShots": ("SHOTS_TOTAL", "home"),
    "AwayShots": ("SHOTS_TOTAL", "away"),
    "HomeShotsOnTarget": ("SHOTS_ON_GOAL", "home"),
    "AwayShotsOnTarget": ("SHOTS_ON_GOAL", "away"),
    "HomeCorners": ("CORNER_KICKS", "home"),
    "AwayCorners": ("CORNER_KICKS", "away"),
    "HomeFouls": ("FOULS", "home"),
    "AwayFouls": ("FOULS", "away"),
    "HomeYellowCards": ("YELLOW_CARDS", "home"),
    "AwayYellowCards": ("YELLOW_CARDS", "away"),
    "HomeRedCards": ("RED_CARDS", "home"),
    "AwayRedCards": ("RED_CARDS", "away"),
}


def extract_stats(match):
    stats_lookup = {}
    raw_stats = match.get("statistics") or []
    for stat in raw_stats:
        stat_type = stat.get("type") or stat.get("name", "")
        stats_lookup[stat_type] = stat
    return stats_lookup


def parse_match(match):
    score = match.get("score", {})
    full_time = score.get("fullTime", {})
    half_time = score.get("halfTime", {})
    winner = score.get("winner")

    ft_home = full_time.get("home")
    ft_away = full_time.get("away")
    ht_home = half_time.get("home")
    ht_away = half_time.get("away")

    # Derive HalfTimeResult from half-time goals if winner field absent
    def result_from_goals(h, a):
        if h is None or a is None:
            return None
        return "H" if h > a else ("A" if a > h else "D")

    ft_result = WINNER_MAP.get(winner) if winner else result_from_goals(ft_home, ft_away)
    ht_result = result_from_goals(ht_home, ht_away)

    raw_date = match.get("utcDate", "")
    try:
        match_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        match_date = raw_date

    stats_lookup = extract_stats(match)

    row = {
        "Season": 2025,
        "MatchDate": match_date,
        "HomeTeam": match.get("homeTeam", {}).get("name"),
        "AwayTeam": match.get("awayTeam", {}).get("name"),
        "FullTimeHomeGoals": ft_home,
        "FullTimeAwayGoals": ft_away,
        "FullTimeResult": ft_result,
        "HalfTimeHomeGoals": ht_home,
        "HalfTimeAwayGoals": ht_away,
        "HalfTimeResult": ht_result,
    }

    for col, (stat_type, side) in STAT_KEYS.items():
        stat_entry = stats_lookup.get(stat_type)
        row[col] = stat_entry.get(side) if stat_entry else None

    return row


matches = data.get("matches", [])
rows = [parse_match(m) for m in matches]

output_path = "data/PL_2025_actual.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Guardado {len(rows)} partidos en {output_path}")
