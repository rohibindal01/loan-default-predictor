"""Tests for src/data/preprocess.py"""

from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.preprocess import clean, split_and_save


def test_clean_drops_nulls(raw_df):
    raw_df.loc[0, "credit_amount"] = None
    result = clean(raw_df)
    assert result.isnull().sum().sum() == 0


def test_clean_filters_underage(raw_df):
    raw_df.loc[0, "age"] = 16
    result = clean(raw_df)
    assert (result["age"] < 18).sum() == 0


def test_clean_filters_zero_credit(raw_df):
    raw_df.loc[0, "credit_amount"] = 0
    result = clean(raw_df)
    assert (result["credit_amount"] <= 0).sum() == 0


def test_clean_filters_zero_duration(raw_df):
    raw_df.loc[0, "duration"] = 0
    result = clean(raw_df)
    assert (result["duration"] <= 0).sum() == 0


def test_split_creates_parquet_files(raw_df, sample_cfg, tmp_path):
    cfg = OmegaConf.merge(sample_cfg, {
        "data": {
            "train_path": str(tmp_path / "train.parquet"),
            "test_path": str(tmp_path / "test.parquet"),
        }
    })
    split_and_save(raw_df, cfg)
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "test.parquet").exists()


def test_split_ratio(raw_df, sample_cfg, tmp_path):
    cfg = OmegaConf.merge(sample_cfg, {
        "data": {
            "train_path": str(tmp_path / "train.parquet"),
            "test_path": str(tmp_path / "test.parquet"),
        }
    })
    split_and_save(raw_df, cfg)
    train = pd.read_parquet(tmp_path / "train.parquet")
    test = pd.read_parquet(tmp_path / "test.parquet")
    ratio = len(test) / (len(train) + len(test))
    assert abs(ratio - 0.2) < 0.05


def test_split_target_preserved(raw_df, sample_cfg, tmp_path):
    cfg = OmegaConf.merge(sample_cfg, {
        "data": {
            "train_path": str(tmp_path / "train.parquet"),
            "test_path": str(tmp_path / "test.parquet"),
        }
    })
    split_and_save(raw_df, cfg)
    train = pd.read_parquet(tmp_path / "train.parquet")
    assert "default" in train.columns
    assert set(train["default"].unique()).issubset({0, 1})
