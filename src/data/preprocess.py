"""Loan-specific cleaning and train/test splitting.

Note: the feature pipeline (scaling + encoding) is NOT defined here.
It is built by calling:

    from src.features.transformers import build_feature_pipeline   # ← Mlops-Plumbing

That function reads configs/default.yaml and constructs the sklearn
ColumnTransformer automatically — zero boilerplate needed here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Loan-specific cleaning rules."""
    before = len(df)
    df = df.dropna()
    if removed := before - len(df):
        logger.warning("Dropped %d rows with nulls", removed)

    # Sanity guards
    df = df[df["credit_amount"] > 0]
    df = df[df["duration"] > 0]
    df = df[df["age"] >= 18]

    logger.info("%d rows after cleaning", len(df))
    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame, cfg: DictConfig) -> None:
    target: str = cfg.data.target_column
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(cfg.data.test_size),
        random_state=int(cfg.data.random_seed),
        stratify=y,
    )

    train = X_train.copy()
    train[target] = y_train.values
    test = X_test.copy()
    test[target] = y_test.values

    train_path = Path(cfg.data.train_path)
    test_path = Path(cfg.data.test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    logger.info(
        "Splits saved: train=%d rows, test=%d rows | default rate train=%.1f%%",
        len(train), len(test), y_train.mean() * 100,
    )


import hydra  # noqa: E402


@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_raw(cfg.data.raw_path)
    df = clean(df)
    split_and_save(df, cfg)


if __name__ == "__main__":
    main()
