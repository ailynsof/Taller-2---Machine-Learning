# Taller 2: Machine Learning I — Premier League 2025/26
**Integrantes:** Alejandro Pardo (Cód. XXXXX), Ailyn Gomez (Cód. YYYYY) & Tomas Rincon (Cód. ZZZZZ)
**URL Dashboard:** [https://dashboard-rust-kappa-37.vercel.app/](https://dashboard-rust-kappa-37.vercel.app/)

Este repositorio contiene la solución completa para el Taller 2 de ML1. Hemos construido un pipeline de machine learning para predecir variables clave de la Premier League 2025-26, incluyendo goles esperados (xG), marcador final y resultado del partido (H/D/A), junto con un EDA profundo y Dashboard interactivo.

---

## 🧠 Approach y Pipeline (Enfoque)

El enfoque de nuestro análisis consiste en construir un **Pipeline end-to-end** dividiendo el proyecto en tres etapas:
1. **Setup & Data Engineering (01_setup.ipynb):** Descarga de datos, filtrado de eventos espaciales, y extracción de más de 10 qualifiers (features booleanas). Calculamos métricas geométricas de los tiros como distancia y ángulo.
2. **Exploratory Data Analysis (02_eda.ipynb):** Analizamos la conversión por distancia/tipo, correlaciones clave, el sesgo de árbitros, y el xG esperado vs precio de jugadores para descubrir insights reales para la predicción.
3. **Modelamiento Predictivo (03_models.ipynb):** Implementamos un modelo logístico base para xG y modelos avanzados (XGBoost y Random Forest). Además, usamos Regresión Lineal Ridge para predecir la cantidad de goles totales en los partidos y una Logística Multinomial para el Match Predictor (clasificación H/D/A), empleando *Rolling Features* para evitar el Data Leakage y capturar el "estado de forma" de cada equipo. Finalmente, un clustering segmenta a los jugadores según sus atributos de rendimiento.

---

## 🧩 Features Usadas por Modelo

### Modelo xG (Logistic Regression, XGBoost, Random Forest)
- **Geométricas:** `distance_to_goal`, `angle_to_goal`, `x_norm`, `y_centered`
- **Qualifiers (Booleanas):** `is_big_chance`, `is_header`, `is_right_foot`, `is_left_foot`, `is_penalty`, `is_counter`, `from_corner`, `is_volley`, `first_touch`

### Match Predictor (Ridge Regression & Logistic Multinomial H/D/A)
- **Odds Base:** `b365h`, `b365d`, `b365a`
- **Rolling Features (shift 1, window=5):** 
  - `home_team_rolling_scored`, `home_team_rolling_conceded`, `home_team_rolling_wins`
  - `away_team_rolling_scored`, `away_team_rolling_conceded`, `away_team_rolling_wins`

### Clustering de Jugadores (KMeans)
- Features: `expected_goals`, `expected_assists`, `goals_scored`, `assists`, `minutes`, `ict_index`, `threat`, `creativity`

---

## 📂 Estructura del Proyecto

*   `notebooks/01_setup.ipynb`: Descarga y preparación inicial de features.
*   `notebooks/02_eda.ipynb`: Exploratory Data Analysis.
*   `notebooks/03_models.ipynb`: Modelamiento predictivo y generación de JSONs para dashboard.
*   `models/`: Modelos exportados en formato `.pkl`.
*   `dashboard/`: Aplicación web con HTML, CSS, JS y Chart.js interactivo.
*   `figures/`: Gráficos PNG generados en EDA y Models.
*   `data/`: Datasets limpios y procesados.

---

## 🚀 Instrucciones de Ejecución Paso a Paso

1.  **Crear y activar el Entorno Virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ejecutar los notebooks en orden:**
    ```bash
    jupyter nbconvert --to notebook --execute notebooks/01_setup.ipynb --output 01_setup_executed.ipynb
    jupyter nbconvert --to notebook --execute notebooks/02_eda.ipynb --output 02_eda_executed.ipynb
    jupyter nbconvert --to notebook --execute notebooks/03_models.ipynb --output 03_models_executed.ipynb
    ```
    *(Alternativamente, abrir Jupyter Lab/Notebook y ejecutar "Restart & Run All" en cada archivo respetando el orden numérico).*
4.  **Visualizar Resultados en el Dashboard:**
    - Levantar un servidor local en la raíz del proyecto o abrir directamente el `dashboard/index.html` en el navegador web.
    - El dashboard cargará dinámicamente los datos `.json` generados en el paso 3.

---
**Machine Learning I — 2026**
