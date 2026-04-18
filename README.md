# loan-default-predictor

[![CI](https://github.com/your-org/loan-default-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/loan-default-predictor/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Built on Mlops-Plumbing](https://img.shields.io/badge/built%20on-Mlops--Plumbing-blue)](https://github.com/rohibindal01/Mlops-Plumbing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Loan Default Prediction** — predicts whether a loan applicant will default.

## How this project uses Mlops-Plumbing

This project **directly imports** core abstractions from
[rohibindal01/Mlops-Plumbing](https://github.com/rohibindal01/Mlops-Plumbing),
which is declared as a **git dependency** in `pyproject.toml`:

```toml
ml-project-template = { git = "https://github.com/rohibindal01/Mlops-Plumbing.git", branch = "main" }
```

### What is imported from Mlops-Plumbing

| Import | Used in | Purpose |
|--------|---------|---------|
| `from src.models.base import BaseModel` | `src/models/loan_model.py` | Abstract interface all models must implement |
| `from src.features.transformers import build_feature_pipeline` | `src/models/trainer.py` | sklearn ColumnTransformer pipeline builder |
| `from src.evaluation.evaluator import threshold_predict` | `src/models/trainer.py` | Threshold-based binary prediction |
| `from src.evaluation.plots import save_roc_curve, save_confusion_matrix, save_feature_importance` | `src/models/trainer.py` | Standardised evaluation plots |

### What this project adds on top

Only the loan-specific logic lives here:

- `src/data/ingest.py` — downloads the Lending Club / UCI loan dataset
- `src/data/preprocess.py` — domain cleaning (loan grades, DTI, FICO bands)
- `src/models/loan_model.py` — `LoanDefaultModel` that **extends** `BaseModel` from Mlops-Plumbing
- `src/serving/app.py` — FastAPI `/predict` endpoint with loan-specific schema
- `configs/default.yaml` — loan-domain hyperparameters and feature definitions

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/loan-default-predictor.git
cd loan-default-predictor

# 2. Install (pulls Mlops-Plumbing automatically via git dep)
make install

# 3. Environment
cp .env.example .env

# 4. Run pipeline
make pipeline

# 5. Tests
make test

# 6. Serve
make serve
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Mlops-Plumbing (git dep)                 │
│                                                             │
│  src.models.base.BaseModel  ◄──────────────────────────┐   │
│  src.features.transformers.build_feature_pipeline       │   │
│  src.evaluation.plots.*                                 │   │
│  src.evaluation.evaluator.threshold_predict             │   │
└─────────────────────────────────────────────────────────────┘
           ▲              ▲               ▲
           │              │               │
           │    imported directly in code │
           │              │               │
┌──────────┴──────────────┴───────────────┴──────────────────┐
│                 loan-default-predictor (this repo)          │
│                                                             │
│  src/data/ingest.py          ← loan dataset download        │
│  src/data/preprocess.py      ← loan-specific cleaning       │
│  src/models/loan_model.py    ← extends BaseModel            │
│  src/serving/app.py          ← loan prediction API          │
│  configs/default.yaml        ← loan hyperparameters         │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
loan-default-predictor/
├── .github/workflows/       CI + CD
├── configs/default.yaml     Hydra config — loan-specific params
├── src/
│   ├── data/
│   │   ├── ingest.py        Download UCI loan dataset
│   │   └── preprocess.py    Clean, encode, split
│   ├── models/
│   │   ├── loan_model.py    LoanDefaultModel extends BaseModel (from Mlops-Plumbing)
│   │   └── trainer.py       Training entry-point, imports plumbing utilities
│   └── serving/
│       ├── app.py           FastAPI app
│       └── schemas.py       Loan-specific Pydantic schemas
├── tests/
├── dvc.yaml
├── Makefile
└── pyproject.toml           ml-project-template listed as git dependency
```
