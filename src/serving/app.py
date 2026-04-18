"""FastAPI serving layer for loan-default-predictor.

Imports the Mlops-Plumbing feature pipeline at startup to transform
raw loan features before passing them to LoanDefaultModel.predict_proba().
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    DefaultPrediction,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
FEATURE_PIPELINE_PATH = os.getenv("FEATURE_PIPELINE_PATH", "models/feature_pipeline.joblib")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.40"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

_state: dict[str, Any] = {}


def _risk_grade(prob: float) -> str:
    """Map default probability to a letter grade (A–E)."""
    if prob < 0.20:
        return "A"
    if prob < 0.35:
        return "B"
    if prob < 0.50:
        return "C"
    if prob < 0.70:
        return "D"
    return "E"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _state["start_time"] = time.time()
    _state["model_loaded"] = False

    model_path = Path(MODEL_PATH)
    pipeline_path = Path(FEATURE_PIPELINE_PATH)

    if model_path.exists() and pipeline_path.exists():
        try:
            # LoanDefaultModel (this repo) — wraps Mlops-Plumbing BaseModel
            _state["model"] = joblib.load(model_path)
            # feature pipeline from Mlops-Plumbing build_feature_pipeline()
            _state["feature_pipeline"] = joblib.load(pipeline_path)
            _state["model_loaded"] = True
            logger.info("Loaded model from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
    else:
        logger.warning("Model files not found — /predict will return 503")

    yield
    _state.clear()


app = FastAPI(
    title="Loan Default Predictor API",
    description=(
        "Predicts loan default probability. "
        "Built on [Mlops-Plumbing](https://github.com/rohibindal01/Mlops-Plumbing) — "
        "feature pipeline and model abstractions imported directly from the plumbing repo."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model_loaded", False),
        model_version=MODEL_VERSION,
        uptime_seconds=time.time() - _state.get("start_time", time.time()),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    if not _state.get("model_loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model = _state["model"]
    # feature_pipeline was built by Mlops-Plumbing's build_feature_pipeline()
    feature_pipeline = _state["feature_pipeline"]

    records = [a.model_dump() for a in request.applications]
    df = pd.DataFrame(records)

    try:
        # Transform using the Mlops-Plumbing fitted ColumnTransformer
        X_t = feature_pipeline.transform(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature transform failed: {exc}") from exc

    # predict_proba() — defined in Mlops-Plumbing BaseModel interface,
    # implemented in LoanDefaultModel
    proba: np.ndarray = model.predict_proba(pd.DataFrame(X_t))[:, 1]

    predictions = [
        DefaultPrediction(
            application_index=i,
            will_default=bool(p >= DEFAULT_THRESHOLD),
            probability=round(float(p), 4),
            risk_grade=_risk_grade(p),
        )
        for i, p in enumerate(proba)
    ]

    return PredictResponse(
        predictions=predictions,
        model_version=MODEL_VERSION,
        threshold_used=DEFAULT_THRESHOLD,
    )
