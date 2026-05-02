# Sistema Híbrido de Predicción de Premier League

Sistema completo de predicción de resultados de partidos que combina múltiples enfoques estadísticos y de machine learning.

## 🎯 Componentes del Sistema

### 1. **Feature Engineering** (`feature_engineering.py`)
- Calcula promedios móviles de los últimos 5 partidos para cada equipo
- Usa datos de Unity Catalog (`workspace.gold.equipos_premier`)
- Genera características como:
  - Promedio de goles anotados/recibidos
  - Tasa de victorias
  - Promedio de córners y tiros a puerta
  - Estadísticas separadas por venue (Home/Away)

### 2. **Modelo Logit** (`logit_model.py`)
- **Regresión Logística Multinomial** para predecir resultado del partido
- Predice probabilidades de: **Victoria Local / Empate / Victoria Visitante**
- Entrenado con características de promedios móviles
- Usa normalización StandardScaler para mejorar convergencia
- **Modelos guardados en**: `/Volumes/workspace/gold/premier_modelos_entrenados/`
  - `logit_model.pkl`
  - `logit_scaler.pkl`
  - `logit_features.pkl`

### 3. **Modelo Poisson** (`poisson_model.py`)
- **Distribución de Poisson** para predecir número de goles
- Calcula fortalezas ofensivas relativas de cada equipo
- Genera:
  - Goles esperados para cada equipo
  - Marcador más probable
  - Probabilidad de Más/Menos 2.5 goles totales
  - Distribución completa de probabilidades por número de goles
  - Predicción de córners totales
- **Parámetros guardados en**: `/Volumes/workspace/gold/premier_modelos_entrenados/`
  - `poisson_params.pkl`

### 4. **Sistema Híbrido** (`hybrid_predict.py`)
- **Combina ambos modelos** con ponderación 60% Logit + 40% Poisson
- Proporciona predicción final consolidada
- Incluye análisis completo del partido

## 📊 Flujo de Predicción

```
┌─────────────────────────────────────────────┐
│  Unity Catalog: workspace.gold.equipos_premier  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Feature Engineering   │
        │ (Promedios Móviles)  │
        └─────────┬────────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│  Logit Model    │  │ Poisson Model   │
│ (Resultado)     │  │ (Goles/Corners) │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Hybrid Predict   │
         │ (Combinación)    │
         └──────────────────┘
                  │
                  ▼
     /Volumes/workspace/gold/
     premier_modelos_entrenados/
```

## 💾 Almacenamiento de Modelos

Los modelos entrenados se guardan en **Unity Catalog Volume** para:
- **Persistencia**: Los modelos sobreviven a reinicios de cluster
- **Compartición**: Múltiples notebooks/jobs pueden acceder a los mismos modelos
- **Versionado**: Fácil gestión y actualización de modelos
- **Gobernanza**: Control de acceso mediante Unity Catalog

**Ubicación**: `/Volumes/workspace/gold/premier_modelos_entrenados/`

**Archivos generados**:
* `logit_model.pkl` - Modelo de regresión logística entrenado
* `logit_scaler.pkl` - StandardScaler para normalización
* `logit_features.pkl` - Lista de columnas de características
* `poisson_params.pkl` - Parámetros del modelo de Poisson (fortalezas de equipos)

## 🚀 Uso

### Entrenar los modelos

```python
# 1. Entrenar modelo logístico
from logit_model import train_logit_model
train_logit_model()

# 2. Calcular parámetros de Poisson
from poisson_model import save_poisson_parameters
save_poisson_parameters()
```

### Hacer predicciones

```python
from hybrid_predict import predict_match

# Predicción completa
result = predict_match("Arsenal FC", "Chelsea FC")
```

### Desde línea de comandos

```bash
python hybrid_predict.py "Arsenal FC" "Chelsea FC"
```

## 📈 Salida del Sistema

El sistema proporciona:

1. **Probabilidades de resultado** (combinadas):
   - Victoria Local (%)
   - Empate (%)
   - Victoria Visitante (%)

2. **Predicción de goles**:
   - Goles esperados por equipo
   - Marcador más probable
   - Probabilidad de Más/Menos 2.5 goles

3. **Predicción de córners**:
   - Córners esperados por equipo
   - Total de córners esperados
   - Probabilidad de Más/Menos 10.5 córners

4. **Características del partido**:
   - Estadísticas recientes de ambos equipos
   - Ventajas relativas (diferencias entre equipos)

## 🔧 Requisitos

- PySpark (para acceso a Unity Catalog)
- scikit-learn (para regresión logística)
- scipy (para distribución de Poisson)
- pandas, numpy
- joblib (para guardar/cargar modelos)
- **Acceso a Unity Catalog Volume**: `workspace.gold.premier_modelos_entrenados`

## 📝 Notas Importantes

- El sistema compara equipos de forma **relativa** (Home vs Away)
- Usa **promedios móviles** de los últimos 5 partidos para capturar forma reciente
- La **combinación de modelos** mejora la robustez de las predicciones
- Los datos provienen de **Unity Catalog**, asegurando consistencia
- Los modelos se guardan en **Unity Catalog Volume** para persistencia y compartición

## 🎲 Ventajas del Sistema Híbrido

1. **Logit** captura patrones complejos en los datos históricos
2. **Poisson** modela la naturaleza estocástica de los goles en fútbol
3. **Promedios móviles** capturan la forma reciente de los equipos
4. **Combinación** reduce el sesgo de un solo modelo
5. **Unity Catalog Volume** permite persistencia y compartición de modelos

## 📊 Ejemplo de Salida

```
======================================================================
PREDICCIÓN HÍBRIDA: Arsenal FC vs Chelsea FC
======================================================================

[1/3] Calculando características con promedios móviles...
✓ Características calculadas:
  Arsenal FC (Local):
    - Goles promedio: 2.20
    - Tasa de victoria: 60.0%
  Chelsea FC (Visitante):
    - Goles promedio: 1.40
    - Tasa de victoria: 40.0%

[2/3] Predicción de resultado con Regresión Logística Multinomial...
✓ Probabilidades (Logit):
  Victoria Local: 52.3%
  Empate: 25.1%
  Victoria Visitante: 22.6%

[3/3] Predicción de goles con Distribución de Poisson...
✓ Predicción de goles (Poisson):
  Goles esperados Arsenal FC: 1.85
  Goles esperados Chelsea FC: 1.20
  Resultado más probable: 2-1
  Probabilidad: 18.5%

======================================================================
RESUMEN - SISTEMA HÍBRIDO
======================================================================

Probabilidades combinadas (60% Logit + 40% Poisson):
  🏠 Victoria Local: 50.5%
  🤝 Empate: 24.8%
  ✈️  Victoria Visitante: 24.7%

🎯 PREDICCIÓN FINAL: Victoria de Arsenal FC (50.5%)
⚽ Marcador más probable: 2-1
📊 Total de goles esperados: 3.05
🚩 Córners totales esperados: 10.5
======================================================================
```

## 🔄 Reentrenamiento

Para actualizar los modelos con nuevos datos:

```python
# Ejecutar ambos scripts de entrenamiento
%run /Users/mago-dios666@hotmail.com/modelo_predictivo/modelo/hibrido/logit_model.py
%run /Users/mago-dios666@hotmail.com/modelo_predictivo/modelo/hibrido/poisson_model.py
```

Los modelos se sobrescribirán automáticamente en el volumen de Unity Catalog.
