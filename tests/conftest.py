"""Shared pytest fixtures for loan-default-predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.fixture()
def sample_cfg():
    return OmegaConf.create({
        "data": {
            "raw_path": "data/raw/loan_data.csv",
            "train_path": "data/processed/train.parquet",
            "test_path": "data/processed/test.parquet",
            "target_column": "default",
            "test_size": 0.2,
            "random_seed": 42,
            "smote": False,
        },
        "features": {
            "numeric_columns": ["duration", "credit_amount", "installment_rate", "age"],
            "categorical_columns": ["checking_account", "credit_history", "purpose"],
            "binary_columns": [],
            "drop_columns": [],
            "scaling": "standard",
        },
        "model": {
            "type": "random_forest",
            "output_path": "models/model.joblib",
            "feature_pipeline_path": "models/feature_pipeline.joblib",
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "n_jobs": 1,
                "random_state": 42,
            },
            "gradient_boosting": {
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 1.0,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test-loans",
            "model_name": "loan-test",
            "register_model": False,
        },
        "serving": {
            "default_threshold": 0.40,
        },
    })


@pytest.fixture()
def raw_df():
    """Minimal fake loan dataframe matching UCI German Credit schema."""
    rng = np.random.default_rng(0)
    n = 150
    return pd.DataFrame({
        "duration": rng.integers(6, 72, n),
        "credit_amount": rng.uniform(500, 18000, n),
        "installment_rate": rng.integers(1, 5, n),
        "age": rng.integers(18, 75, n),
        "existing_credits": rng.integers(1, 4, n),
        "num_dependents": rng.integers(1, 3, n),
        "checking_account": rng.choice(["A11", "A12", "A13", "A14"], n),
        "credit_history": rng.choice(["A30", "A31", "A32", "A33", "A34"], n),
        "purpose": rng.choice(["A40", "A41", "A42", "A43", "A49"], n),
        "savings_account": rng.choice(["A61", "A62", "A63", "A64", "A65"], n),
        "employment": rng.choice(["A71", "A72", "A73", "A74", "A75"], n),
        "personal_status": rng.choice(["A91", "A92", "A93", "A94"], n),
        "other_debtors": rng.choice(["A101", "A102", "A103"], n),
        "property": rng.choice(["A121", "A122", "A123", "A124"], n),
        "other_installments": rng.choice(["A141", "A142", "A143"], n),
        "housing": rng.choice(["A151", "A152", "A153"], n),
        "job": rng.choice(["A171", "A172", "A173", "A174"], n),
        "telephone": rng.choice(["A191", "A192"], n),
        "foreign_worker": rng.choice(["A201", "A202"], n),
        "default": rng.integers(0, 2, n),
    })
