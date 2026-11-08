"""Entrenamiento, comparación y evaluación de modelos de regresión."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, FIG_DIR, MODEL_DIR, RANDOM_STATE, TABLE_DIR, TARGET, TEST_SIZE

sns.set_theme(style="whitegrid", context="notebook")


def split(df: pd.DataFrame):
    """Separa en entrenamiento y prueba ANTES de ajustar nada, para evitar fuga de datos."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputa nulos con la mediana, estandariza numéricas y codifica categóricas (one-hot)."""
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidate_models() -> dict:
    return {
        "Baseline (media)": DummyRegressor(strategy="mean"),
        "Regresión lineal": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=2, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compare_models(X_train, y_train, cv=5) -> pd.DataFrame:
    """Validación cruzada de cada modelo SOLO sobre el set de entrenamiento."""
    rows = []
    for name, est in candidate_models().items():
        pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", est)])
        scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        rows.append({"modelo": name, "rmse_cv_media": scores.mean(), "rmse_cv_std": scores.std()})
    return pd.DataFrame(rows).sort_values("rmse_cv_media").reset_index(drop=True)


def tune_boosting(X_train, y_train) -> RandomizedSearchCV:
    """Búsqueda aleatoria de hiperparámetros para el modelo de boosting."""
    pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
    space = {
        "model__learning_rate": uniform(0.03, 0.17),
        "model__max_iter": randint(200, 600),
        "model__max_leaf_nodes": randint(15, 60),
        "model__min_samples_leaf": randint(10, 60),
        "model__l2_regularization": uniform(0, 1),
    }
    search = RandomizedSearchCV(pipe, space, n_iter=10, cv=3, scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=2)
    return search.fit(X_train, y_train)


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {"rmse": rmse(y_test, pred), "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred))}


def plot_model_comparison(cv_table):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cv_table, y="modelo", x="rmse_cv_media", ax=ax, color="#1f78b4")
    ax.errorbar(cv_table["rmse_cv_media"], range(len(cv_table)), xerr=cv_table["rmse_cv_std"], fmt="none", color="black", capsize=3)
    ax.set(title="Error (RMSE) en validación cruzada: menor es mejor", xlabel="RMSE (USD)", ylabel="")
    fig.savefig(FIG_DIR / "08_comparacion_modelos.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(y_test, pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, pred, alpha=0.25, s=8, color="#1b9e77")
    lim = [0, max(y_test.max(), pred.max())]
    axes[0].plot(lim, lim, "r--", label="Predicción perfecta")
    axes[0].set(title="Real vs. predicho (test)", xlabel="Precio real", ylabel="Precio predicho")
    axes[0].legend()
    resid = y_test - pred
    axes[1].scatter(pred, resid, alpha=0.25, s=8, color="#7570b3")
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set(title="Residuos vs. predicho", xlabel="Precio predicho", ylabel="Residuo (real - predicho)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_real_vs_predicho.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_importance(model, X_test, y_test):
    res = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=2)
    imp = pd.DataFrame({"variable": X_test.columns, "importancia": res.importances_mean, "std": res.importances_std}).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["variable"], imp["importancia"], xerr=imp["std"], color="#d95f02")
    ax.set(title="Importancia por permutación (aumento del RMSE al barajar la variable)", xlabel="Aumento de RMSE (USD)")
    fig.savefig(FIG_DIR / "10_importancia_variables.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    imp.sort_values("importancia", ascending=False).round(2).to_csv(TABLE_DIR / "importancia_variables.csv", index=False)
    return imp


def run_models(df: pd.DataFrame) -> dict:
    """Pipeline completo: split -> comparar -> ajustar -> evaluar -> guardar."""
    X_train, X_test, y_train, y_test = split(df)

    cv_table = compare_models(X_train, y_train)
    cv_table.round(1).to_csv(TABLE_DIR / "comparacion_cv.csv", index=False)
    plot_model_comparison(cv_table)

    search = tune_boosting(X_train, y_train)
    best = search.best_estimator_

    # Resultados finales en test de todos los modelos (ajustados con todo el train)
    rows = []
    for name, est in candidate_models().items():
        pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", est)]).fit(X_train, y_train)
        rows.append({"modelo": name, **evaluate(pipe, X_test, y_test)})
    rows.append({"modelo": "Gradient Boosting (ajustado)", **evaluate(best, X_test, y_test)})
    test_table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    test_table.round(3).to_csv(TABLE_DIR / "metricas_test.csv", index=False)

    pred = best.predict(X_test)
    plot_predictions(y_test, pred)
    importance = plot_importance(best, X_test, y_test)
    joblib.dump(best, MODEL_DIR / "mejor_modelo.joblib")

    return {"cv": cv_table, "test": test_table, "best_params": search.best_params_, "importance": importance}
