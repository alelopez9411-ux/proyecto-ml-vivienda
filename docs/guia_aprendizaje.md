# Guía de aprendizaje: de cero a tu primer proyecto de ML

Esta guía te lleva de la mano para entender el proyecto, aprender los conceptos clave y publicarlo en GitHub. Está pensada para alguien que ya trabaja con datos pero empieza en estadística y Machine Learning.

---

## Parte 1. Primeros pasos (hoy)

1. **Instala Python 3.10+** y un editor (VS Code recomendado).
2. Descarga la carpeta del proyecto, abre una terminal dentro de ella y ejecuta:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python run_all.py
   ```
3. Abre el notebook: `jupyter notebook notebooks/analisis_vivienda.ipynb`
4. **Léelo completo sin modificar nada** y ejecuta las celdas en orden. Anota en una libreta todo término que no entiendas (usa el glosario de abajo).

---

## Parte 2. Plan de estudio de 8 semanas

Dedica 5-6 horas por semana. Cada semana: **teoría corta → práctica en este repo → mini-ejercicio**.

| Semana | Tema | Qué practicar en este proyecto |
|---|---|---|
| 1 | Estadística descriptiva: media, mediana, desviación, percentiles, asimetría | Lee `src/stats.py → descriptive()`. Calcula a mano la media y mediana de `median_income` y compara con la tabla. |
| 2 | Probabilidad básica y distribuciones (normal, asimetrías, outliers) | Mira `02_distribuciones.png`. ¿Qué variables no son normales? Prueba `np.log1p` en `population`. |
| 3 | Inferencia: intervalos de confianza y pruebas de hipótesis (p-valor, tamaño del efecto) | Estudia `hypothesis_tests()`. Repite el t-test entre `NEAR BAY` y `NEAR OCEAN`. |
| 4 | Correlación y regresión lineal | Lee el resumen OLS (`reports/tables/ols_resumen.txt`). Añade `total_rooms` al modelo y observa qué cambia. |
| 5 | Fundamentos de ML: train/test, overfitting, fuga de datos, validación cruzada | En `models.py` localiza dónde se divide train/test y por qué el preprocesamiento va dentro del `Pipeline`. |
| 6 | Modelos de árboles: Random Forest y Gradient Boosting | Cambia `n_estimators` y `min_samples_leaf` del Random Forest y mira cómo cambia el RMSE en CV. |
| 7 | Métricas, interpretación e hiperparámetros | Explica con tus palabras RMSE, MAE y R². Prueba `n_iter=30` en `tune_boosting()`. |
| 8 | Comunicar resultados | Reescribe las conclusiones del README para un público no técnico (3-4 frases). |

**Retos finales (opcionales, suben mucho el nivel del portafolio):**
- Predecir `log(precio)` y comparar resultados.
- Eliminar los distritos con precio topado y medir el cambio.
- Hacer validación espacial: entrenar con unas zonas y probar en otras.
- Armar un segundo proyecto de **clasificación** con otro dataset.

### Recursos gratuitos recomendados
- **Kaggle Learn** (cursos cortos): Python, Pandas, Intro to Machine Learning, Intermediate ML.
- **Documentación de scikit-learn**: la *User Guide* y los ejemplos son excelentes.
- **StatQuest (YouTube)**: explica estadística y ML de forma muy visual.
- **Libro "Hands-On Machine Learning" (Aurélien Géron)**: este proyecto sigue la misma línea que su capítulo de vivienda.

---

## Parte 3. Glosario rápido

| Término | Explicación corta |
|---|---|
| **Variable objetivo (target)** | Lo que queremos predecir (aquí, el precio). |
| **Feature** | Variable de entrada del modelo. |
| **Regresión / Clasificación** | Predecir un número / predecir una categoría. |
| **Train / Test** | Datos para aprender / datos "nunca vistos" para evaluar. |
| **Overfitting** | El modelo memoriza el entrenamiento y falla con datos nuevos. |
| **Data leakage (fuga)** | Información del test que se cuela en el entrenamiento y da resultados falsamente buenos. |
| **Validación cruzada** | Evaluar rotando qué parte de los datos se usa para validar. |
| **Hiperparámetro** | Ajuste del modelo que defines tú (no se aprende solo). |
| **RMSE / MAE** | Error típico de predicción, en las unidades del objetivo. |
| **R²** | Proporción de variabilidad explicada (0 a 1). |
| **p-valor** | Probabilidad de ver un resultado así si la hipótesis nula fuera cierta. |
| **Tamaño del efecto** | Qué tan grande es una diferencia, más allá de si es "significativa". |
| **Intervalo de confianza** | Rango plausible para un valor poblacional. |
| **One-hot encoding** | Convertir categorías en columnas 0/1. |
| **Imputación** | Rellenar valores nulos con un valor razonable. |

---

## Parte 4. Subir el proyecto a GitHub

### Opción A: desde la terminal

1. Crea una cuenta en [github.com](https://github.com) si no la tienes.
2. En GitHub: **New repository** → nombre `proyecto-ml-vivienda` → déjalo público → **no** marques "Add README" (ya lo tienes) → **Create repository**.
3. En la terminal, dentro de la carpeta del proyecto:
   ```bash
   git init
   git add .
   git commit -m "Análisis completo de precios de vivienda en California"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/proyecto-ml-vivienda.git
   git push -u origin main
   ```
   (Si pide contraseña, GitHub exige un *Personal Access Token* o iniciar sesión con GitHub CLI: `gh auth login`.)

### Opción B: sin terminal
En tu repositorio nuevo, usa **Add file → Upload files** y arrastra el contenido de la carpeta. Respeta la estructura de subcarpetas.

### Antes de publicar, revisa
- [ ] En `README.md`, cambia `<tu-usuario>` por tu usuario real.
- [ ] Añade un archivo `LICENSE` (en GitHub: **Add file → Create new file → LICENSE** y elige la plantilla MIT).
- [ ] Verifica que el notebook se vea con los gráficos en GitHub (ya está ejecutado).
- [ ] Añade en la descripción del repo y en *Topics*: `data-analysis`, `machine-learning`, `python`, `scikit-learn`, `statistics`.
- [ ] Fija (pin) el repositorio en tu perfil de GitHub.

---

## Parte 5. Cómo presentarlo (portafolio / entrevistas)

Estructura de 60 segundos para explicarlo:

1. **Problema:** "Predije el valor de la vivienda en California a partir de ubicación e ingreso."
2. **Método:** "Revisé calidad de datos, hice EDA y pruebas estadísticas, y comparé modelos con validación cruzada evitando fuga de datos."
3. **Resultado:** "Gradient Boosting alcanzó R² 0,85 y error medio de 29 000 USD, frente a 90 000 del modelo base."
4. **Honestidad:** "Los datos son de 1990 y hay precios topados, así que lo tomo como ejercicio metodológico, no para decidir hoy."

Ese último punto es lo que más valoran los reclutadores: saber qué *no* puede afirmar tu análisis.
