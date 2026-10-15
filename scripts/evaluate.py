"""Metrik evaluasi dan pembuatan gambar (confusion matrix, feature importance)."""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
)

from scripts import config  # noqa: E402


def micro_f1(y_true, y_pred) -> float:
    """Micro-averaged F1 (metrik resmi kompetisi)."""
    return f1_score(y_true, y_pred, average="micro")


def report(y_true, y_pred) -> None:
    """Cetak classification report dan micro-F1."""
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Micro F1: {micro_f1(y_true, y_pred):.5f}")


def plot_confusion(y_true, y_pred, save_path: Optional[Path] = None) -> None:
    """Gambar confusion matrix ter-normalisasi per baris (recall per kelas).

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        save_path: Jika diberikan, gambar disimpan ke path ini.
    """
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (normalized by true class)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_feature_importance(
    pipeline, X: pd.DataFrame, y, save_path: Optional[Path] = None,
    n_samples: int = 20000, n_repeats: int = 3,
) -> pd.DataFrame:
    """Hitung & gambar permutation importance pada sampel data.

    Permutation importance dipakai karena ``HistGradientBoosting`` tidak
    menyediakan ``feature_importances_``. Karena dihitung pada pipeline utuh,
    nilainya dinyatakan pada kolom mentah (sebelum feature engineering).

    Args:
        pipeline: Pipeline yang sudah di-fit.
        X: Fitur mentah.
        y: Target.
        save_path: Lokasi penyimpanan gambar (opsional).
        n_samples: Jumlah baris sampel untuk menghemat waktu.
        n_repeats: Jumlah pengulangan permutasi.

    Returns:
        DataFrame ``feature, importance`` terurut menurun.
    """
    idx = np.random.RandomState(config.RANDOM_STATE).choice(
        len(X), size=min(n_samples, len(X)), replace=False
    )
    result = permutation_importance(
        pipeline, X.iloc[idx], y.iloc[idx], scoring="accuracy",
        n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=1,
    )
    imp = (
        pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top = imp.head(15).sort_values("importance")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"], color="#3b6ea5")
    plt.xlabel("Penurunan akurasi saat fitur diacak")
    plt.title("Permutation Feature Importance (Top 15)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()
    return imp
