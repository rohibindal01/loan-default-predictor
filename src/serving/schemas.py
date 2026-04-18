"""Pydantic schemas for the loan default prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """Features for a single loan application (UCI German Credit schema)."""

    duration: int = Field(..., gt=0, description="Loan duration in months")
    credit_amount: float = Field(..., gt=0, description="Loan amount in DM")
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate as % of income (1-4)")
    age: int = Field(..., ge=18, description="Applicant age in years")
    existing_credits: int = Field(..., ge=1, description="Number of existing credits at this bank")
    num_dependents: int = Field(..., ge=1, le=2, description="Number of dependents (1-2)")

    checking_account: str = Field(..., description="A11 | A12 | A13 | A14")
    credit_history: str = Field(..., description="A30-A34")
    purpose: str = Field(..., description="A40-A49")
    savings_account: str = Field(..., description="A61-A65")
    employment: str = Field(..., description="A71-A75")
    personal_status: str = Field(..., description="A91-A95")
    other_debtors: str = Field(default="A101", description="A101-A103")
    property: str = Field(default="A121", description="A121-A124")
    other_installments: str = Field(default="A141", description="A141-A143")
    housing: str = Field(default="A151", description="A151-A153")
    job: str = Field(default="A173", description="A171-A174")
    telephone: str = Field(default="A191", description="A191 | A192")
    foreign_worker: str = Field(default="A201", description="A201 | A202")

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }


class PredictRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=256)


class DefaultPrediction(BaseModel):
    application_index: int
    will_default: bool
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_grade: str  # A (lowest risk) → E (highest risk)


class PredictResponse(BaseModel):
    predictions: list[DefaultPrediction]
    model_version: str
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
