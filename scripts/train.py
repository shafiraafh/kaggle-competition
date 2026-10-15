"""Training: bangun Pipeline lengkap, cross-validation, fit final, simpan model.

Jalankan dari root repo::

    python -m scripts.train
    python -m scripts.train --no-cv          # lewati CV, langsung fit final
    python -m scripts.train --params models/best_params.json   # pakai hasil tuning
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from scripts import config
from scripts.evaluate import micro_f1, plot_confusion, plot_feature_importance, report
from scripts.feature_engineering import BuildingFeatureEngineer
from scripts.preprocess import (
    build_preprocessor,
    categorical_mask,
    load_train,
    split_xy,
)

DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
}


def build_pipeline(params: Optional[dict] = None) -> Pipeline:
    """Bangun Pipeline: feature engineering -> preprocessing -> model.

    Seluruh langkah berada dalam satu objek sehingga ``fit``/``predict``/CV/tuning
    selalu konsisten dan bebas kebocoran data.

    Args:
        params: Hyperparameter ``HistGradientBoostingClassifier``. Nilai yang
            diberikan menimpa ``DEFAULT_PARAMS``.

    Returns:
        ``sklearn.pipeline.Pipeline`` yang belum di-fit.
    """
    merged = {**DEFAULT_PARAMS, **(params or {})}
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask(),
        random_state=config.RANDOM_STATE,
        **merged,
    )
    return Pipeline(
        steps=[
            ("features", BuildingFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def cross_validate(pipeline: Pipeline, X, y, n_splits: int = config.N_FOLDS):
    """Stratified K-Fold CV dan kembalikan prediksi out-of-fold.

    Args:
        pipeline: Pipeline yang belum di-fit (akan di-clone tiap fold).
        X: Fitur mentah.
        y: Target.
        n_splits: Jumlah fold.

    Returns:
        Tuple ``(oof_pred, fold_scores)``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_STATE)
    oof = np.empty(len(y), dtype=object)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        pipe = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict(X.iloc[va])
        scores.append(micro_f1(y.iloc[va], oof[va]))
        print(f"Fold {fold}/{n_splits} micro-F1: {scores[-1]:.5f}")
    return oof, scores


def train(use_cv: bool = True, params: Optional[dict] = None,
          sample: Optional[int] = None) -> Pipeline:
    """Latih model, evaluasi dengan CV, fit final pada seluruh data, lalu simpan.

    Args:
        use_cv: Jalankan Stratified K-Fold CV sebelum fit final.
        params: Override hyperparameter model.
        sample: Jika diisi, hanya pakai N baris pertama-acak (untuk uji cepat).

    Returns:
        Pipeline final yang sudah di-fit pada seluruh data training.
    """
    train_df = load_train()
    if sample:
        train_df = train_df.sample(sample, random_state=config.RANDOM_STATE)
    X, y = split_xy(train_df)
    print(f"Train: {X.shape}")

    pipeline = build_pipeline(params)
    metrics = {}

    if use_cv:
        oof, scores = cross_validate(pipeline, X, y)
        print("\n=== OOF report ===")
        report(y, oof)
        plot_confusion(y, oof, save_path=config.FIGURES_DIR / "confusion_matrix_oof.png")
        metrics = {
            "cv_micro_f1_mean": float(np.mean(scores)),
            "cv_micro_f1_std": float(np.std(scores)),
            "cv_micro_f1_folds": [float(s) for s in scores],
            "n_folds": config.N_FOLDS,
        }

    print("\nFit final pada seluruh data training...")
    pipeline.fit(X, y)
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH, compress=3)
    print(f"Model tersimpan: {config.MODEL_PATH}")

    imp = plot_feature_importance(
        pipeline, X, y, save_path=config.FIGURES_DIR / "feature_importance.png"
    )
    print("\nTop 5 fitur:\n", imp.head(5).to_string(index=False))

    metrics["n_iter_final"] = int(pipeline.named_steps["model"].n_iter_)
    metrics["params"] = {**DEFAULT_PARAMS, **(params or {})}
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return pipeline


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Train pipeline model.")
    parser.add_argument("--no-cv", action="store_true", help="Lewati cross-validation.")
    parser.add_argument("--params", type=Path, help="JSON hyperparameter (hasil tune.py).")
    parser.add_argument("--sample", type=int, help="Pakai N baris acak saja (uji cepat).")
    args = parser.parse_args()
    params = json.loads(args.params.read_text()) if args.params else None
    train(use_cv=not args.no_cv, params=params, sample=args.sample)


if __name__ == "__main__":
    main()
