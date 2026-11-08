"""Estadística descriptiva e inferencial sobre el dataset."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import TABLE_DIR, TARGET


def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """Media, mediana, desviación, asimetría y curtosis de cada variable numérica."""
    num = df.select_dtypes("number")
    out = num.describe().T
    out["mediana"] = num.median()
    out["asimetria"] = num.skew()
    out["curtosis"] = num.kurt()
    return out.round(3)


def correlations_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson (relación lineal) y Spearman (relación monótona) contra el precio, con p-valor."""
    rows = []
    d = df.select_dtypes("number").dropna()
    for col in d.columns.drop(TARGET):
        r, p = stats.pearsonr(d[col], d[TARGET])
        rho, p_s = stats.spearmanr(d[col], d[TARGET])
        rows.append({"variable": col, "pearson_r": r, "p_pearson": p, "spearman_rho": rho, "p_spearman": p_s})
    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False).round(4).reset_index(drop=True)


def mean_ci_by_group(df: pd.DataFrame, group="ocean_proximity", conf=0.95) -> pd.DataFrame:
    """Precio medio por grupo con intervalo de confianza al 95 %."""
    rows = []
    for name, g in df.groupby(group):
        x = g[TARGET].to_numpy()
        m, se = x.mean(), stats.sem(x)
        lo, hi = stats.t.interval(conf, len(x) - 1, loc=m, scale=se)
        rows.append({group: name, "n": len(x), "media": m, "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(rows).sort_values("media").round(0).reset_index(drop=True)


def cohens_d(a, b) -> float:
    """Tamaño del efecto: cuántas desviaciones estándar separan dos medias."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Tres pruebas de hipótesis con su interpretación.

    1. ANOVA: ¿el precio medio es igual en todas las categorías de cercanía al océano?
    2. Kruskal-Wallis: lo mismo, sin suponer normalidad (más robusto).
    3. t de Welch: ¿cuestan distinto los distritos INLAND y los de <1H OCEAN?
    """
    groups = [g[TARGET].to_numpy() for _, g in df.groupby("ocean_proximity")]
    f, p_f = stats.f_oneway(*groups)
    h, p_h = stats.kruskal(*groups)
    inland = df.loc[df.ocean_proximity == "INLAND", TARGET]
    near = df.loc[df.ocean_proximity == "<1H OCEAN", TARGET]
    t, p_t = stats.ttest_ind(inland, near, equal_var=False)
    d = cohens_d(inland, near)
    return pd.DataFrame(
        [
            {"prueba": "ANOVA (todas las categorías)", "estadistico": f, "p_valor": p_f, "efecto": np.nan},
            {"prueba": "Kruskal-Wallis (todas)", "estadistico": h, "p_valor": p_h, "efecto": np.nan},
            {"prueba": "t de Welch INLAND vs <1H OCEAN", "estadistico": t, "p_valor": p_t, "efecto": d},
        ]
    )


def ols_summary(df: pd.DataFrame):
    """Regresión lineal (OLS) con statsmodels: coeficientes, errores, p-valores y R²."""
    d = df.dropna().copy()
    X = pd.get_dummies(d[["median_income", "housing_median_age", "rooms_per_household", "population_per_household", "ocean_proximity"]], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    model = sm.OLS(d[TARGET], X).fit(cov_type="HC3")  # errores robustos a heterocedasticidad
    return model


def save_all(df: pd.DataFrame) -> dict:
    """Calcula y guarda todas las tablas estadísticas en reports/tables."""
    res = {
        "descriptivas": descriptive(df),
        "correlaciones": correlations_with_target(df),
        "ic_por_grupo": mean_ci_by_group(df),
        "pruebas": hypothesis_tests(df),
    }
    for name, table in res.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=name not in {"correlaciones", "ic_por_grupo", "pruebas"})
    ols = ols_summary(df)
    (TABLE_DIR / "ols_resumen.txt").write_text(str(ols.summary()))
    res["ols"] = ols
    return res
