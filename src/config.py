"""Configuración central del proyecto: rutas y constantes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "housing.csv"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"

TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]
RANDOM_STATE = 42  # semilla fija: los resultados son reproducibles
TEST_SIZE = 0.2

for d in (FIG_DIR, TABLE_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
