"""Prediksi data test dengan pipeline tersimpan dan tulis file submission."""

import joblib
import pandas as pd

from scripts import config
from scripts.preprocess import load_test

VALID_LABELS = [f"Grade {i}" for i in range(1, 6)]


def make_submission(pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    """Buat DataFrame submission dari pipeline dan data test.

    Args:
        pipeline: Pipeline yang sudah di-fit.
        test_df: Data test mentah (termasuk ``building_id``).

    Returns:
        DataFrame berkolom ``building_id`` dan ``damage_grade``.
    """
    X_test = test_df.drop(columns=[config.ID_COL])
    return pd.DataFrame(
        {config.ID_COL: test_df[config.ID_COL].values,
         config.TARGET: pipeline.predict(X_test)}
    )


def validate_submission(sub: pd.DataFrame) -> None:
    """Cek format submission terhadap ``sample_submission.csv``.

    Raises:
        AssertionError: jika kolom, jumlah baris, urutan ID, atau label tidak valid.
    """
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    assert sub.columns.tolist() == sample.columns.tolist(), "Kolom tidak sesuai"
    assert len(sub) == len(sample), "Jumlah baris tidak sesuai"
    assert set(sub[config.ID_COL]) == set(sample[config.ID_COL]), "building_id tidak sesuai"
    assert sub[config.TARGET].isin(VALID_LABELS).all(), "Label tidak valid"


def predict() -> pd.DataFrame:
    """Muat model, prediksi data test, validasi, dan simpan ke ``submissions/``.

    Returns:
        DataFrame submission yang disimpan.
    """
    pipeline = joblib.load(config.MODEL_PATH)
    sub = make_submission(pipeline, load_test())
    validate_submission(sub)
    config.SUBMISSIONS_DIR.mkdir(exist_ok=True)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission tersimpan: {config.SUBMISSION_PATH}  shape={sub.shape}")
    print(sub[config.TARGET].value_counts().sort_index())
    return sub


if __name__ == "__main__":
    predict()
