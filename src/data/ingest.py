"""Download and parse the UCI German Credit (loan default) dataset.

The dataset has no header — we assign column names per the UCI spec.
Target: 1 = good (no default), 2 = bad (default)  →  we remap to 0/1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Column names from the UCI German Credit dataset specification
COLUMN_NAMES = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installments",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "default",   # 1 = good, 2 = bad
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    logger.info("Saved %d bytes", len(r.content))


def parse_raw(path: Path) -> pd.DataFrame:
    """Parse space-separated file with no header, remap target to 0/1."""
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)
    # UCI encodes: 1 = good loan, 2 = default  →  remap to 0 = good, 1 = default
    df["default"] = (df["default"] == 2).astype(int)
    logger.info("Parsed %d rows | default rate: %.1f%%", len(df), df["default"].mean() * 100)
    return df


import hydra  # noqa: E402


@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dest = Path(cfg.data.raw_path)
    if not dest.exists():
        download(cfg.data.source_url, dest)
    else:
        logger.info("Raw file exists at %s — skipping download", dest)

    df = parse_raw(dest)
    # Re-save as proper CSV with header so downstream steps work uniformly
    df.to_csv(dest, index=False)
    logger.info("Saved parsed CSV to %s", dest)


if __name__ == "__main__":
    main()
