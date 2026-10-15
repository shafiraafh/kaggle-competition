"""Feature engineering sebagai transformer scikit-learn (aman dipakai di Pipeline)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from scripts import config

ENGINEERED_COLS = [
    "age_unknown",
    "height_per_floor",
    "area_per_floor",
    "height_area_ratio",
    "age_x_floors",
]


class BuildingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Membersihkan ``age_building`` dan menambah fitur turunan.

    Transformer ini *stateless* (tidak belajar dari data), sehingga tidak ada
    risiko kebocoran data antara train dan validasi.

    Fitur yang ditambahkan:
        - ``age_unknown``: 1 jika umur bernilai 999 (kode "tidak diketahui").
        - ``height_per_floor``: tinggi bangunan / jumlah lantai.
        - ``area_per_floor``: luas plinth / jumlah lantai.
        - ``height_area_ratio``: tinggi / luas plinth (indikator kelangsingan).
        - ``age_x_floors``: umur * jumlah lantai.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "BuildingFeatureEngineer":
        """Tidak ada yang dipelajari; hanya mengembalikan ``self``."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan fitur turunan ke salinan ``X``.

        Args:
            X: DataFrame fitur mentah (tanpa ID dan target).

        Returns:
            DataFrame baru dengan kolom tambahan ``ENGINEERED_COLS``.
        """
        df = X.copy()
        unknown = df["age_building"] == config.AGE_UNKNOWN_VALUE
        df["age_unknown"] = unknown.astype(int)
        df["age_building"] = df["age_building"].where(~unknown, np.nan)

        floors = df["count_floors_pre_eq"].clip(lower=1)
        df["height_per_floor"] = df["height_ft_pre_eq"] / floors
        df["area_per_floor"] = df["plinth_area_sq_ft"] / floors
        df["height_area_ratio"] = df["height_ft_pre_eq"] / (df["plinth_area_sq_ft"] + 1)
        df["age_x_floors"] = df["age_building"] * df["count_floors_pre_eq"]
        return df
