"""Training entry-point for loan-default-predictor.

Shows explicit imports from Mlops-Plumbing at the top of every section.
This file is intentionally annotated to make the dependency crystal-clear.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import f1_score, roc_auc_score

# ── IMPORTS FROM MLOPS-PLUMBING (git dependency) ──────────────────────────────
from src.features.transformers import build_feature_pipeline  # feature pipeline builder
from src.evaluation.evaluator import threshold_predict        # threshold-based prediction
from src.evaluation.plots import (                            # standardised plot functions
    save_roc_curve,
    save_confusion_matrix,
    save_feature_importance,
)
from src.features.transformers import get_feature_names       # feature name extractor
# ─────────────────────────────────────────────────────────────────────────────

# ── THIS REPO'S OWN MODEL ─────────────────────────────────────────────────────
from src.models.loan_model import LoanDefaultModel            # extends Mlops-Plumbing BaseModel
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def load_split(path: str | Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    return df.drop(columns=[target_col]), df[target_col]


def apply_smote(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not found — skipping SMOTE")
        return X, y
    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info("SMOTE: %d → %d samples", len(y), len(y_res))
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


import hydra  # noqa: E402


@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    target_col: str = cfg.data.target_column

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train = load_split(cfg.data.train_path, target_col)
    X_test, y_test = load_split(cfg.data.test_path, target_col)

    # ── Feature pipeline — from Mlops-Plumbing ────────────────────────────────
    # build_feature_pipeline() reads cfg.features.{numeric,categorical,scaling}
    # and returns a fitted sklearn ColumnTransformer Pipeline.
    feature_pipeline = build_feature_pipeline(cfg)          # ← Mlops-Plumbing
    X_train_t = feature_pipeline.fit_transform(X_train)
    X_test_t = feature_pipeline.transform(X_test)

    # ── SMOTE (optional) ──────────────────────────────────────────────────────
    if cfg.data.smote:
        X_train_df = pd.DataFrame(X_train_t)
        X_train_df, y_train = apply_smote(X_train_df, y_train, cfg.data.random_seed)
        X_train_t = X_train_df.values

    # ── MLflow ────────────────────────────────────────────────────────────────
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        logger.info("MLflow run: %s", run.info.run_id)

        # ── Train — LoanDefaultModel (extends Mlops-Plumbing BaseModel) ───────
        model = LoanDefaultModel(cfg)
        model.fit(pd.DataFrame(X_train_t), y_train)

        # ── Evaluate on test set ──────────────────────────────────────────────
        threshold = float(cfg.serving.default_threshold)
        y_proba = model.predict_proba(pd.DataFrame(X_test_t))[:, 1]

        # threshold_predict() — imported from Mlops-Plumbing  ←────────────────
        y_pred = threshold_predict(y_proba, threshold)

        metrics = {
            "test_auc": roc_auc_score(y_test, y_proba),
            "test_f1": f1_score(y_test, y_pred),
        }
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        logger.info("AUC=%.4f | F1=%.4f", metrics["test_auc"], metrics["test_f1"])

        # ── Plots — imported from Mlops-Plumbing ──────────────────────────────
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)

        roc_img = save_roc_curve(y_test, y_proba, report_dir)       # ← Mlops-Plumbing
        cm_img = save_confusion_matrix(y_test, y_pred, report_dir)  # ← Mlops-Plumbing

        importances = model.feature_importances()
        if importances is not None:
            feature_names = get_feature_names(feature_pipeline, cfg)  # ← Mlops-Plumbing
            fi_img = save_feature_importance(importances, feature_names, report_dir)  # ← Mlops-Plumbing
            mlflow.log_artifact(str(fi_img))

        mlflow.log_artifact(str(roc_img))
        mlflow.log_artifact(str(cm_img))

        # ── Save model artefacts ──────────────────────────────────────────────
        model_path = Path(cfg.model.output_path)
        pipeline_path = Path(cfg.model.feature_pipeline_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(feature_pipeline, pipeline_path)

        if cfg.mlflow.register_model:
            mlflow.sklearn.log_model(
                model._clf,
                artifact_path="model",
                registered_model_name=cfg.mlflow.model_name,
            )

    # ── DVC metrics ───────────────────────────────────────────────────────────
    Path("reports/train_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Done — artefacts written.")


if __name__ == "__main__":
    main()
