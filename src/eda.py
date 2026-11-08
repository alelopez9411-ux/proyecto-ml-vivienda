"""Análisis exploratorio (EDA): genera y guarda las figuras del proyecto."""
import matplotlib

matplotlib.use("Agg")  # sin ventana: guarda directo a archivo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import FIG_DIR, TARGET

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_missing(df):
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#d95f02")
    ax.set(xlabel="% de valores nulos", ylabel="", title="Valores nulos por columna")
    return _save(fig, "01_nulos.png")


def plot_distributions(df):
    num = df.select_dtypes("number").columns
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), num):
        sns.histplot(df[col], bins=40, ax=ax, color="#1b9e77")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(num):]:
        ax.axis("off")
    fig.suptitle("Distribución de las variables numéricas", y=1.01, fontsize=14)
    fig.tight_layout()
    return _save(fig, "02_distribuciones.png")


def plot_target(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], bins=50, ax=axes[0], color="#7570b3")
    axes[0].set(title="Precio mediano de la vivienda", xlabel="USD")
    sns.histplot(np.log1p(df[TARGET]), bins=50, ax=axes[1], color="#7570b3")
    axes[1].set(title="Mismo precio en escala logarítmica", xlabel="log(1 + USD)")
    fig.tight_layout()
    return _save(fig, "03_target.png")


def plot_correlation(df):
    corr = df.select_dtypes("number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de correlación (Pearson)")
    return _save(fig, "04_correlacion.png")


def plot_income_vs_price(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="median_income", y=TARGET, alpha=0.15, s=10, ax=ax, color="#1f78b4")
    ax.set(title="Ingreso mediano vs. precio de la vivienda", xlabel="Ingreso mediano (decenas de miles USD)", ylabel="Precio mediano (USD)")
    return _save(fig, "05_ingreso_vs_precio.png")


def plot_map(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df[TARGET], s=df["population"] / 100, cmap="viridis", alpha=0.4)
    fig.colorbar(sc, ax=ax, label="Precio mediano (USD)")
    ax.set(title="Mapa de California: precio y población (tamaño)", xlabel="Longitud", ylabel="Latitud")
    return _save(fig, "06_mapa_precios.png")


def plot_by_category(df):
    order = df.groupby("ocean_proximity")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="ocean_proximity", y=TARGET, order=order, ax=ax, palette="Set2", hue="ocean_proximity", legend=False)
    ax.set(title="Precio según cercanía al océano", xlabel="", ylabel="Precio mediano (USD)")
    return _save(fig, "07_precio_por_cercania.png")


def run_eda(df):
    """Genera todas las figuras y devuelve sus rutas."""
    return [
        plot_missing(df),
        plot_distributions(df),
        plot_target(df),
        plot_correlation(df),
        plot_income_vs_price(df),
        plot_map(df),
        plot_by_category(df),
    ]
