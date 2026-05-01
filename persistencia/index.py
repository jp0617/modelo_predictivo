import pandas as pd

df1 =pd.read_csv('data/epl_final.csv')
df2 = pd.read_csv('data/PL_2025_actual.csv')

resultado = pd.concat([df1, df2], axis=0)

cols_comunes = ['Season', 'MatchDate', 'FullTimeResult', 'HalfTimeResult']

# Columnas que queremos para cada equipo (sin el prefijo Home/Away)
cols_stats = ['Team', 'Goals', 'HTGoals', 'Shots', 'ShotsOnTarget', 'Corners', 'Fouls', 'YellowCards', 'RedCards']

# 3. Crear el DataFrame de equipos Locales
home_df = resultado[cols_comunes + ['HomeTeam', 'FullTimeHomeGoals', 'HalfTimeHomeGoals', 'HomeShots', 'HomeShotsOnTarget', 'HomeCorners', 'HomeFouls', 'HomeYellowCards', 'HomeRedCards']].copy()
home_df.columns = cols_comunes + cols_stats
home_df['Venue'] = 'Home'

# 4. Crear el DataFrame de equipos Visitantes
away_df = resultado[cols_comunes + ['AwayTeam', 'FullTimeAwayGoals', 'HalfTimeAwayGoals', 'AwayShots', 'AwayShotsOnTarget', 'AwayCorners', 'AwayFouls', 'AwayYellowCards', 'AwayRedCards']].copy()
away_df.columns = cols_comunes + cols_stats
away_df['Venue'] = 'Away'

# 5. Concatenar ambos para tener una lista de "partidos por equipo"
df_final = pd.concat([home_df, away_df], axis=0).sort_values(by=['MatchDate', 'Team'])

# Ejemplo: Ver todos los datos de un equipo específico
print(df_final[df_final['Team'] == 'Man City'])

