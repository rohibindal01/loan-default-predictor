"""Verify that all Mlops-Plumbing imports resolve correctly.

These tests exist specifically to prove the git dependency is wired up
and every plumbing abstraction used by this project is importable.
If any of these fail it means Mlops-Plumbing is not installed or has changed.
"""

from __future__ import annotations


# ── 1. BaseModel abstract interface ───────────────────────────────────────────
def test_import_base_model():
    from src.models.base import BaseModel  # noqa: F401
    assert BaseModel is not None


# ── 2. Feature pipeline builder ───────────────────────────────────────────────
def test_import_build_feature_pipeline():
    from src.features.transformers import build_feature_pipeline  # noqa: F401
    assert callable(build_feature_pipeline)


def test_import_get_feature_names():
    from src.features.transformers import get_feature_names  # noqa: F401
    assert callable(get_feature_names)


# ── 3. Evaluation utilities ───────────────────────────────────────────────────
def test_import_threshold_predict():
    from src.evaluation.evaluator import threshold_predict  # noqa: F401
    assert callable(threshold_predict)


def test_import_save_roc_curve():
    from src.evaluation.plots import save_roc_curve  # noqa: F401
    assert callable(save_roc_curve)


def test_import_save_confusion_matrix():
    from src.evaluation.plots import save_confusion_matrix  # noqa: F401
    assert callable(save_confusion_matrix)


def test_import_save_feature_importance():
    from src.evaluation.plots import save_feature_importance  # noqa: F401
    assert callable(save_feature_importance)


# ── 4. LoanDefaultModel correctly extends BaseModel ───────────────────────────
def test_loan_model_is_subclass_of_base_model():
    from src.models.base import BaseModel
    from src.models.loan_model import LoanDefaultModel
    assert issubclass(LoanDefaultModel, BaseModel)


# ── 5. LoanDefaultModel implements all abstract methods ───────────────────────
def test_loan_model_implements_interface(sample_cfg):
    from src.models.loan_model import LoanDefaultModel
    model = LoanDefaultModel(sample_cfg)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "get_params")
    assert hasattr(model, "feature_importances")


# ── 6. build_feature_pipeline returns a sklearn Pipeline ─────────────────────
def test_build_feature_pipeline_returns_pipeline(sample_cfg):
    from sklearn.pipeline import Pipeline
    from src.features.transformers import build_feature_pipeline
    pipeline = build_feature_pipeline(sample_cfg)
    assert isinstance(pipeline, Pipeline)


# ── 7. threshold_predict correctness ─────────────────────────────────────────
def test_threshold_predict_values():
    import numpy as np
    from src.evaluation.evaluator import threshold_predict
    proba = np.array([0.1, 0.3, 0.45, 0.6, 0.9])
    result = threshold_predict(proba, threshold=0.4)
    expected = np.array([0, 0, 1, 1, 1])
    assert (result == expected).all()
