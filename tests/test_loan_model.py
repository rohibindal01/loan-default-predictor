"""Tests for LoanDefaultModel — verifies BaseModel contract is fulfilled."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

# Import from Mlops-Plumbing (git dep) — same as production code
from src.features.transformers import build_feature_pipeline
from src.models.base import BaseModel
from src.models.loan_model import LoanDefaultModel


@pytest.fixture()
def X_y(raw_df, sample_cfg):
    target = sample_cfg.data.target_column
    X = raw_df.drop(columns=[target])
    y = raw_df[target]
    pipeline = build_feature_pipeline(sample_cfg)   # ← Mlops-Plumbing
    X_t = pipeline.fit_transform(X)
    return pd.DataFrame(X_t), y, pipeline


def test_loan_model_is_base_model_subclass():
    assert issubclass(LoanDefaultModel, BaseModel)


def test_fit_returns_self(X_y, sample_cfg):
    X, y, _ = X_y
    model = LoanDefaultModel(sample_cfg)
    result = model.fit(X, y)
    assert result is model


def test_predict_shape(X_y, sample_cfg):
    X, y, _ = X_y
    model = LoanDefaultModel(sample_cfg)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(X_y, sample_cfg):
    X, y, _ = X_y
    model = LoanDefaultModel(sample_cfg)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_sums_to_one(X_y, sample_cfg):
    X, y, _ = X_y
    model = LoanDefaultModel(sample_cfg)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_get_params_has_domain_key(sample_cfg):
    model = LoanDefaultModel(sample_cfg)
    params = model.get_params()
    assert params["domain"] == "loan_default"
    assert "model_type" in params


def test_feature_importances_not_none(X_y, sample_cfg):
    X, y, _ = X_y
    model = LoanDefaultModel(sample_cfg)
    model.fit(X, y)
    fi = model.feature_importances()
    assert fi is not None
    assert len(fi) == X.shape[1]


def test_gradient_boosting_variant(X_y, sample_cfg):
    cfg = OmegaConf.merge(sample_cfg, {"model": {"type": "gradient_boosting"}})
    X, y, _ = X_y
    model = LoanDefaultModel(cfg)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_unknown_model_type_raises(sample_cfg):
    cfg = OmegaConf.merge(sample_cfg, {"model": {"type": "xgboost_unknown"}})
    with pytest.raises(ValueError, match="Unknown model.type"):
        LoanDefaultModel(cfg)


def test_feature_pipeline_transform_compatible(X_y, sample_cfg):
    """End-to-end: plumbing pipeline output → loan model predict_proba."""
    X, y, pipeline = X_y
    model = LoanDefaultModel(sample_cfg)
    model.fit(X, y)
    # raw dataframe → pipeline.transform → model.predict_proba
    raw_subset = pd.read_parquet  # just confirm the pipeline object is reusable
    X_new_t = pipeline.transform(X.iloc[:5])
    proba = model.predict_proba(pd.DataFrame(X_new_t))
    assert proba.shape == (5, 2)
