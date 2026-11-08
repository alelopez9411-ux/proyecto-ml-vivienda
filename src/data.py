"""Carga y preparación de los datos."""
import numpy as np
import pandas as pd

from .config import DATA_RAW, TARGET


def load_data(path=DATA_RAW) -> pd.DataFrame:
    """Lee el CSV original sin modificarlo."""
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de calidad: tipo, nulos, % de nulos y valores únicos por columna."""
    return pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "pct_nulos": (df.isna().mean() * 100).round(2),
            "unicos": df.nunique(),
        }
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas (feature engineering) con sentido de negocio.

    Los totales por distrito dependen de su tamaño; las razones por hogar
    son más comparables entre distritos.
    """
    out = df.copy()
    out["rooms_per_household"] = out["total_rooms"] / out["households"]
    out["bedrooms_per_room"] = out["total_bedrooms"] / out["total_rooms"]
    out["population_per_household"] = out["population"] / out["households"]
    return out


def censored_share(df: pd.DataFrame) -> float:
    """Proporción de distritos con precio topado en el máximo del dataset (500001)."""
    return float((df[TARGET] >= df[TARGET].max()).mean())
