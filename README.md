# Modelo Predictivo — Premier League

API REST + interfaz web para predecir resultados de partidos de la Premier League combinando un clasificador Random Forest y simulación Monte Carlo con distribuciones de Poisson.

---

## Arquitectura

```
index.py               # Servidor Flask — endpoints y página web
wsgi.py                # Punto de entrada para Gunicorn
modelo/
  train.py             # Entrena el Random Forest y guarda artefactos
  predict.py           # Carga el modelo y calcula predicciones
  montecarlo.py        # Simulación Monte Carlo + Poisson
data/
  epl_final.csv        # Histórico de temporadas anteriores
  PL_2025_actual.csv   # Temporada en curso (se puede agregar PL_YYYY_actual.csv)
  normalize.py         # Normaliza nombres de equipos a formato oficial
```

### Modelos

| Modelo | Descripción |
|--------|-------------|
| **Random Forest** | Clasificador con 200 árboles entrenado sobre estadísticas rolling de los últimos 5 partidos de cada equipo (goles, tiros a puerta, tasa de victorias). Predice H / D / A con sus probabilidades. |
| **Monte Carlo + Poisson** | Simula 50 000 partidos usando lambdas de Poisson calibradas por la fortaleza ofensiva/defensiva de cada equipo relativa al promedio de la liga. Devuelve probabilidades combinadas (analítico + simulación), goles esperados, Over/Under 2.5, BTTS y top 5 marcadores más probables. |

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Top 5 equipos por tasa de victorias reciente |
| `GET` | `/predict` | Interfaz web para seleccionar equipos y ver predicción |
| `GET` | `/teams` | Lista de equipos disponibles (JSON) |
| `GET` | `/team/<local>/<visitante>` | Predicción completa: Random Forest + Monte Carlo |
| `GET` | `/team/<equipo>` | Estadísticas recientes de un equipo |

### Ejemplo `/team/Arsenal FC/Liverpool FC`

```json
{
  "predict": {
    "Equipo_local": "Arsenal FC",
    "Equipo_visitante": "Liverpool FC",
    "Probabilidades": {
      "Victoria local": "42.3%",
      "Empate": "26.1%",
      "Victoria visitante": "31.6%"
    },
    "Estadísticas recientes del equipo local": { ... },
    "Estadísticas recientes del equipo visitante": { ... }
  },
  "montecarlo": {
    "Victoria_local": "38.5%",
    "Empate": "25.2%",
    "Victoria_visitante": "36.3%",
    "Goles_esperados_local": 1.47,
    "Goles_esperados_visitante": 1.38,
    "Over_2.5": "55.0%",
    "BTTS": "61.2%",
    "Prediccion_final": "Victoria Arsenal FC",
    "Confianza": "40.4%"
  }
}
```

---

## Instalación local

**Requisitos:** Python 3.11+

```bash
# Clonar el repositorio
git clone <repo-url>
cd "Modelo predictivo"

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar en modo desarrollo
python wsgi.py
# → http://localhost:5001
```

Al arrancar por primera vez, si no existe `modelo/model.pkl`, el modelo se entrena automáticamente.

---

## Datos

### Fuentes

| Archivo | Fuente | Cómo obtenerlo |
|---------|--------|----------------|
| `epl_final.csv` | [Kaggle — English Premier League Match Data 2000-2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025) | Ver `data/Historico_PL.py` |
| `PL_YYYY_actual.csv` | [football-data.org API v4](https://www.football-data.org/) | Ver `data/data_actual.py` |

### Histórico de temporadas (`epl_final.csv`)

Descarga automática vía `kagglehub`:

```bash
pip install kagglehub
python data/Historico_PL.py
```

El script descarga el dataset de Kaggle y lo guarda en `data/epl_final.csv` (también lo copia a Unity Catalog Volume si se ejecuta en Databricks).

### Temporada en curso (`PL_YYYY_actual.csv`)

Se obtiene desde la API gratuita de [football-data.org](https://www.football-data.org/). Requiere registrarse para obtener un token gratuito:

1. Crear cuenta en [football-data.org](https://www.football-data.org/) → obtener el API token.
2. Reemplazar el token en `data/data_actual.py`:
   ```python
   headers = {'X-Auth-Token': 'TU_TOKEN_AQUI'}
   ```
3. Ejecutar el script:
   ```bash
   python data/data_actual.py
   ```

El script detecta automáticamente la temporada activa (año anterior si el mes es antes de junio) y guarda el resultado como `data/PL_YYYY_actual.csv`.

### Estructura de columnas requerida

Los CSVs deben contener al menos:

```
Season, MatchDate, HomeTeam, AwayTeam,
FullTimeHomeGoals, FullTimeAwayGoals, FullTimeResult,
HomeShotsOnTarget, AwayShotsOnTarget
```

`predict.py` lee **todos los archivos `.csv`** del directorio `data/` y los une automáticamente, ordenados alfabéticamente. Para agregar una nueva temporada basta con copiar el archivo:

```
data/PL_2026_actual.csv   ← se incluye al reiniciar la app
```

Los nombres de equipos se normalizan automáticamente al formato oficial completo (`Man City` → `Manchester City FC`) mediante `data/normalize.py`.

---

## Despliegue

El proyecto está configurado para **Render** (`render.yaml`):

```yaml
buildCommand: pip install -r requirements.txt
startCommand:  gunicorn wsgi:app
```

---

## Reentrenar el modelo

```bash
python modelo/train.py
```

Genera y sobreescribe los artefactos en `modelo/`:

| Archivo | Contenido |
|---------|-----------|
| `model.pkl` | Random Forest entrenado |
| `label_encoder.pkl` | Codificador H / D / A |
| `feature_cols.pkl` | Orden de columnas de entrada |
| `shots_medians.pkl` | Medianas de tiros para imputación |
