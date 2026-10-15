"""Hyperparameter tuning memakai Pipeline + ``RandomizedSearchCV``.

Parameter dicari pada langkah ``model`` dengan prefiks ``model__`` sehingga
feature engineering & preprocessing ikut di-fit ulang per fold (tanpa leakage).

Jalankan::

    python -m scripts.tune --n-iter 10 --sample 100000
"""

import argparse
import json

from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from scripts import config
from scripts.preprocess import load_train, split_xy
from scripts.train import build_pipeline

PARAM_DISTRIBUTIONS = {
    "model__learning_rate": loguniform(0.03, 0.2),
    "model__max_leaf_nodes": [31, 63, 127, 255],
    "model__min_samples_leaf": [20, 40, 80, 150],
    "model__l2_regularization": loguniform(1e-2, 1e1),
    "model__max_depth": [None, 8, 12, 16],
}


def tune(n_iter: int = 10, sample: int = None) -> dict:
    """Jalankan randomized search dan simpan parameter terbaik.

    Args:
        n_iter: Jumlah kombinasi hyperparameter yang dicoba.
        sample: Pakai N baris acak untuk mempercepat pencarian (opsional).

    Returns:
        Dict hyperparameter terbaik (tanpa prefiks ``model__``).
    """
    df = load_train()
    if sample:
        df = df.sample(sample, random_state=config.RANDOM_STATE)
    X, y = split_xy(df)

    search = RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="f1_micro",
        cv=StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                           random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=False,
    )
    search.fit(X, y)

    best = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()}
    print(f"\nBest CV micro-F1: {search.best_score_:.5f}")
    print("Best params:", best)

    out = config.MODELS_DIR / "best_params.json"
    out.write_text(json.dumps(best, indent=2))
    print(f"Disimpan ke {out}")
    return best


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Tune hyperparameter pipeline.")
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    tune(args.n_iter, args.sample)


if __name__ == "__main__":
    main()
