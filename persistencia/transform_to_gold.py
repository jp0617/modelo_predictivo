from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# Crear sesión de Spark
spark = SparkSession.builder.getOrCreate()

# 1. Leer la tabla consolidada de la capa silver
df_consolidado = spark.table("workspace.silver.consolidado_premier")

# 2. Definir columnas comunes
cols_comunes = ['season', 'MatchDate', 'FullTimeResult', 'HalfTimeResult']

# 3. Crear DataFrame para equipos Locales (Home)
home_df = df_consolidado.select(
    col('season').alias('Season'),
    col('MatchDate'),
    col('FullTimeResult'),
    col('HalfTimeResult'),
    col('HomeTeam').alias('Team'),
    col('FullTimeHomeGoals').alias('Goals'),
    col('HalfTimeHomeGoals').alias('HTGoals'),
    col('HomeShots').alias('Shots'),
    col('HomeShotsOnTarget').alias('ShotsOnTarget'),
    col('HomeCorners').alias('Corners'),
    col('HomeFouls').alias('Fouls'),
    col('HomeYellowCards').alias('YellowCards'),
    col('HomeRedCards').alias('RedCards'),
    lit('Home').alias('Venue')
)

# 4. Crear DataFrame para equipos Visitantes (Away)
away_df = df_consolidado.select(
    col('season').alias('Season'),
    col('MatchDate'),
    col('FullTimeResult'),
    col('HalfTimeResult'),
    col('AwayTeam').alias('Team'),
    col('FullTimeAwayGoals').alias('Goals'),
    col('HalfTimeAwayGoals').alias('HTGoals'),
    col('AwayShots').alias('Shots'),
    col('AwayShotsOnTarget').alias('ShotsOnTarget'),
    col('AwayCorners').alias('Corners'),
    col('AwayFouls').alias('Fouls'),
    col('AwayYellowCards').alias('YellowCards'),
    col('AwayRedCards').alias('RedCards'),
    lit('Away').alias('Venue')
)

# 5. Unir ambos DataFrames y ordenar por fecha y equipo
df_final = home_df.union(away_df).orderBy('MatchDate', 'Team')

# 6. Guardar en la capa gold
df_final.write.mode('overwrite').saveAsTable('workspace.gold.equipos_premier')

# 7. Mostrar un resumen
print("Datos transformados y guardados en workspace.gold.equipos_premier")
print(f"Total de registros: {df_final.count()}")

# Ejemplo: Ver datos de un equipo específico
print("\nEjemplo - Datos de Man City:")
df_final.filter(col('Team') == 'Man City').show(10)
