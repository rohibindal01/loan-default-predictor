# data/

DVC-tracked data directory.

- `raw/loan_data.csv` — UCI German Credit dataset (downloaded by `src/data/ingest.py`)
- `processed/train.parquet` — training split
- `processed/test.parquet` — test split

Run `make pipeline` or `dvc pull` to populate.
