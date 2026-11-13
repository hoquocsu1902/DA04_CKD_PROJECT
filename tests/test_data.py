"""Kiểm thử E1: dữ liệu nạp vào phải khớp đặc tả đề DA-04."""
from pathlib import Path

import numpy as np
import pytest

from data.nap import (CATEGORICAL_COLS, FEATURE_COLS, LABEL_COL,
                      NUMERIC_COLS, load_ckd)

DATA = Path("data/raw/kidney_disease.csv")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="Chưa có file dữ liệu thật")


@pytest.fixture(scope="module")
def df():
    return load_ckd(DATA)


def test_shape(df):
    assert df.shape == (400, 25)
    assert list(df.columns) == FEATURE_COLS + [LABEL_COL]


def test_schema(df):
    assert len(NUMERIC_COLS) == 11 and len(CATEGORICAL_COLS) == 13
    for c in NUMERIC_COLS:
        assert df[c].dtype == np.float64


def test_missing_and_labels(df):
    assert int(df[FEATURE_COLS].isna().sum().sum()) == 1012
    assert df[LABEL_COL].value_counts().to_dict() == {"ckd": 250, "notckd": 150}
    assert int(df.duplicated().sum()) == 0


def test_khong_con_token_ban(df):
    for c in CATEGORICAL_COLS + [LABEL_COL]:
        vals = df[c].dropna()
        assert not vals.isin(["?", ""]).any()
        assert not vals.str.contains(r"^\s|\s$", regex=True).any()
