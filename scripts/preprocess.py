"""Loader data dan preprocessing berbasis scikit-learn ``ColumnTransformer``."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from scripts import config


def load_train() -> pd.DataFrame:
    """Baca ``data/train.csv`` (sudah berisi kolom target ``damage_grade``)."""
    return pd.read_csv(config.TRAIN_PATH)


def load_test() -> pd.DataFrame:
    """Baca ``data/test.csv`` (tanpa kolom target)."""
    return pd.read_csv(config.TEST_PATH)


def load_metadata() -> pd.DataFrame:
    """Baca ``data/metadata.csv`` (deskripsi tiap variabel)."""
    return pd.read_csv(config.METADATA_PATH)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Pisahkan fitur dan target dari data training.

    Args:
        df: DataFrame training lengkap (berisi ``building_id`` dan ``damage_grade``).

    Returns:
        Tuple ``(X, y)``; ``X`` tanpa kolom ID & target, ``y`` berupa label string
        ``"Grade 1"`` .. ``"Grade 5"``.
    """
    X = df.drop(columns=[config.ID_COL, config.TARGET])
    y = df[config.TARGET]
    return X, y


def build_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Bangun ``ColumnTransformer`` untuk encoding fitur.

    Kolom kategorikal di-encode dengan ``OrdinalEncoder`` (kategori tak dikenal
    saat prediksi menjadi ``NaN``), kolom numerik diteruskan apa adanya. Kolom
    kategorikal SELALU berada di depan sehingga indeks ``0..n_cat-1`` bisa
    dipakai sebagai ``categorical_features`` pada model.

    Args:
        categorical_cols: Daftar kolom kategorikal. Default ``config.CATEGORICAL_COLS``.
        numeric_cols: Daftar kolom numerik hasil feature engineering.

    Returns:
        ``ColumnTransformer`` yang belum di-fit.
    """
    categorical_cols = categorical_cols or config.CATEGORICAL_COLS
    numeric_cols = numeric_cols or get_numeric_cols()
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def get_numeric_cols() -> List[str]:
    """Daftar kolom numerik akhir: kolom asli + hasil feature engineering."""
    from scripts.feature_engineering import ENGINEERED_COLS

    return config.NUMERIC_COLS + ENGINEERED_COLS


def categorical_mask() -> List[int]:
    """Indeks kolom kategorikal pada output preprocessor (selalu di depan)."""
    return list(range(len(config.CATEGORICAL_COLS)))
