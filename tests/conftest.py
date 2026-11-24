"""Dữ liệu GIẢ có đúng schema CKD, chỉ dùng để kiểm thử mã. Không dùng cho kết quả báo cáo."""
import numpy as np
import pandas as pd
import pytest

from data.nap import CATEGORICAL_COLS, LABEL_COL, NUMERIC_COLS
from models import doi_chieu as dc


def tao_df_gia(n=400, ti_le_thieu=0.10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array(["ckd"] * (n * 5 // 8) + ["notckd"] * (n - n * 5 // 8))
    benh = (y == "ckd")
    df = pd.DataFrame(index=range(n))
    for c in NUMERIC_COLS:
        df[c] = rng.normal(0, 1, n) + 1.5 * benh * rng.choice([-1, 1]) + 10
    for c in ("sg",):
        df[c] = np.where(benh, rng.choice(["1.005", "1.010", "1.015"], n),
                         rng.choice(["1.020", "1.025"], n))
    for c in ("al", "su"):
        df[c] = np.where(benh, rng.choice(["0", "1", "2", "3"], n), "0")
    nhi_phan = {"rbc": ("normal", "abnormal"), "pc": ("normal", "abnormal"),
                "pcc": ("notpresent", "present"), "ba": ("notpresent", "present"),
                "htn": ("no", "yes"), "dm": ("no", "yes"), "cad": ("no", "yes"),
                "appet": ("good", "poor"), "pe": ("no", "yes"), "ane": ("no", "yes")}
    for c, (a, b) in nhi_phan.items():
        df[c] = np.where(rng.random(n) < np.where(benh, 0.6, 0.1), b, a)
    df[LABEL_COL] = y
    cot = [c for c in df.columns if c != LABEL_COL]
    # giống dữ liệu thật: ~60% bệnh nhân thiếu ít nhất 1 ô, còn lại đủ 24 ô
    hang_thieu = rng.random(n) < 0.6
    mask = (rng.random((n, len(cot))) < ti_le_thieu / 0.6) & hang_thieu[:, None]
    for j, c in enumerate(cot):
        df[c] = df[c].astype(object).where(~mask[:, j], np.nan) if c in CATEGORICAL_COLS \
            else df[c].where(~mask[:, j], np.nan)
    return df[[*(NUMERIC_COLS + CATEGORICAL_COLS), LABEL_COL]]


@pytest.fixture(scope="session")
def du_lieu_gia():
    return dc.ma_hoa(tao_df_gia())          # (X, y, ten_cot)
