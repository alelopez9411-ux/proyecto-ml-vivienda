"""Ejecuta el análisis completo: python run_all.py"""
import pandas as pd

from src.data import add_features, censored_share, load_data, quality_report
from src.eda import run_eda
from src.models import run_models
from src.stats import save_all

pd.set_option("display.width", 140)


def main():
    df = add_features(load_data())
    print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
    print(f"Distritos con precio topado en el máximo: {censored_share(df):.1%}\n")
    print(quality_report(df), "\n")

    print("1/3 Generando figuras del EDA ...")
    run_eda(df)

    print("2/3 Calculando estadística ...")
    res = save_all(df)
    print(res["pruebas"].round(4), "\n")

    print("3/3 Entrenando y evaluando modelos (puede tardar unos minutos) ...")
    out = run_models(df)
    print("\nValidación cruzada (train):\n", out["cv"].round(1))
    print("\nResultados en test:\n", out["test"].round(3))
    print("\nMejores hiperparámetros:", out["best_params"])
    print("\nListo. Figuras en reports/figures, tablas en reports/tables.")


if __name__ == "__main__":
    main()
