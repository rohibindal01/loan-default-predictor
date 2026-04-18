"""LoanDefaultModel — extends BaseModel from Mlops-Plumbing.

The ONLY model code that lives in this repo. Everything else
(the abstract interface, the factory pattern) comes from the plumbing.

Direct import from Mlops-Plumbing git dependency:

    from src.models.base import BaseModel
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# ── IMPORTED FROM MLOPS-PLUMBING ──────────────────────────────────────────────
from src.models.base import BaseModel  # noqa: E402  (from Mlops-Plumbing git dep)
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class LoanDefaultModel(BaseModel):
    """Loan default classifier.

    Extends ``BaseModel`` from Mlops-Plumbing, implementing all abstract
    methods with loan-domain logic on top.

    Supports two backends configurable via ``cfg.model.type``:
    - ``random_forest``     → sklearn RandomForestClassifier
    - ``gradient_boosting`` → sklearn GradientBoostingClassifier
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._model_type: str = cfg.model.type
        self._clf = self._build_clf(cfg)
        self._params: dict[str, Any] = self._extract_params(cfg)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_clf(cfg: DictConfig) -> Any:
        model_type = cfg.model.type
        if model_type == "random_forest":
            rf = cfg.model.random_forest
            return RandomForestClassifier(
                n_estimators=int(rf.n_estimators),
                max_depth=int(rf.max_depth),
                min_samples_split=int(rf.min_samples_split),
                min_samples_leaf=int(rf.min_samples_leaf),
                class_weight=str(rf.class_weight),
                n_jobs=int(rf.n_jobs),
                random_state=int(rf.random_state),
            )
        if model_type == "gradient_boosting":
            gb = cfg.model.gradient_boosting
            return GradientBoostingClassifier(
                n_estimators=int(gb.n_estimators),
                learning_rate=float(gb.learning_rate),
                max_depth=int(gb.max_depth),
                subsample=float(gb.subsample),
                min_samples_leaf=int(gb.min_samples_leaf),
                random_state=int(gb.random_state),
            )
        raise ValueError(f"Unknown model.type: {model_type!r}. Choose random_forest or gradient_boosting.")

    @staticmethod
    def _extract_params(cfg: DictConfig) -> dict[str, Any]:
        model_type = cfg.model.type
        sub = dict(cfg.model.random_forest if model_type == "random_forest" else cfg.model.gradient_boosting)
        return {"model_type": model_type, "domain": "loan_default", **sub}

    # ── BaseModel interface (all required by Mlops-Plumbing) ──────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LoanDefaultModel":
        logger.info(
            "Training %s on %d loan samples (default rate: %.1f%%) …",
            self._model_type, len(X), y.mean() * 100,
        )
        self._clf.fit(X, y)
        logger.info("Training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict(X)  # type: ignore[return-value]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(X)  # type: ignore[return-value]

    def get_params(self) -> dict[str, Any]:
        """Return params for MLflow logging — uses Mlops-Plumbing convention."""
        return self._params

    def feature_importances(self) -> np.ndarray | None:
        return getattr(self._clf, "feature_importances_", None)
