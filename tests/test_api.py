"""Tests for the FastAPI loan prediction serving layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

VALID_APPLICATION = {
    "duration": 24,
    "credit_amount": 4500.0,
    "installment_rate": 3,
    "age": 35,
    "existing_credits": 1,
    "num_dependents": 1,
    "checking_account": "A11",
    "credit_history": "A32",
    "purpose": "A43",
    "savings_account": "A61",
    "employment": "A73",
    "personal_status": "A93",
    "other_debtors": "A101",
    "property": "A121",
    "other_installments": "A141",
    "housing": "A151",
    "job": "A173",
    "telephone": "A191",
    "foreign_worker": "A201",
}


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_required_fields():
    body = client.get("/health").json()
    assert "status" in body
    assert "model_loaded" in body
    assert "uptime_seconds" in body
    assert "model_version" in body


def test_health_status_is_ok():
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_predict_without_model_returns_503():
    """Model files don't exist in test env — should get 503, not 500."""
    resp = client.post("/predict", json={"applications": [VALID_APPLICATION]})
    assert resp.status_code == 503


def test_predict_empty_list_returns_422():
    resp = client.post("/predict", json={"applications": []})
    assert resp.status_code == 422


def test_predict_missing_required_field_returns_422():
    bad = {k: v for k, v in VALID_APPLICATION.items() if k != "duration"}
    resp = client.post("/predict", json={"applications": [bad]})
    assert resp.status_code == 422


def test_predict_invalid_age_returns_422():
    bad = {**VALID_APPLICATION, "age": 15}   # under 18
    resp = client.post("/predict", json={"applications": [bad]})
    assert resp.status_code == 422


def test_predict_zero_credit_amount_returns_422():
    bad = {**VALID_APPLICATION, "credit_amount": 0}
    resp = client.post("/predict", json={"applications": [bad]})
    assert resp.status_code == 422


def test_predict_batch_too_large_returns_422():
    resp = client.post("/predict", json={"applications": [VALID_APPLICATION] * 257})
    assert resp.status_code == 422


def test_openapi_schema_available():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
