"""
Bản scikit-learn / scipy để đối chiếu và làm backend tạm cho thực nghiệm.

HỢP ĐỒNG GIAO DIỆN (để thay bằng mã tự viết khi các module khác merge):
  - bộ điền khuyết:   .fit(X) -> self ;  .transform(X) -> ndarray không còn NaN
  - bộ chuẩn hoá:     .fit(X) -> self ;  .transform(X)   (fit bỏ qua NaN, giữ NaN khi transform)
  - PCA:              .fit(X) -> self ;  .transform(X) ; thuộc tính components_ (k x d)
  - bộ phân lớp:      .fit(X, y) -> self ;  .predict(X)
  - X luôn là ndarray float64, cột sắp theo data.nap.FEATURE_COLS (11 số trước, 13 danh mục sau).

scikit-learn chỉ xuất hiện ở file này; mã tự viết không được import file này.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data.nap import CATEGORICAL_COLS, FEATURE_COLS, LABEL_COL, NUMERIC_COLS

# ----------------------------------------------------------------------
# Mã hoá cố định (không học gì từ dữ liệu -> không gây rò rỉ)
# ----------------------------------------------------------------------
_NHI_PHAN = {
    "rbc": {"normal": 0.0, "abnormal": 1.0},
    "pc": {"normal": 0.0, "abnormal": 1.0},
    "pcc": {"notpresent": 0.0, "present": 1.0},
    "ba": {"notpresent": 0.0, "present": 1.0},
    "htn": {"no": 0.0, "yes": 1.0},
    "dm": {"no": 0.0, "yes": 1.0},
    "cad": {"no": 0.0, "yes": 1.0},
    "appet": {"good": 0.0, "poor": 1.0},
    "pe": {"no": 0.0, "yes": 1.0},
    "ane": {"no": 0.0, "yes": 1.0},
}


def ma_hoa(df: pd.DataFrame):
    """DataFrame từ data.nap.load_ckd -> (X float64 [n x 24], y {0,1} với ckd = 1, tên cột).

    sg/al/su là thang thứ bậc -> giữ nguyên giá trị số; 10 cột còn lại là nhị phân -> 0/1.
    Ô thiếu giữ là NaN. Việc điền khuyết và chuẩn hoá làm sau, trong từng fold.
    """
    cot = []
    for c in FEATURE_COLS:
        if c in NUMERIC_COLS:
            cot.append(df[c].astype("float64").to_numpy())
        elif c in _NHI_PHAN:
            cot.append(df[c].map(_NHI_PHAN[c]).astype("float64").to_numpy())
        else:  # sg, al, su
            cot.append(pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy())
    X = np.column_stack(cot)
    y = (df[LABEL_COL] == "ckd").astype(int).to_numpy()
    return X, y, list(FEATURE_COLS)


# ----------------------------------------------------------------------
# Thành phần pipeline
# ----------------------------------------------------------------------
class TrungViTanSuat:
    """Median cho n_so cột đầu (cột số), mode cho các cột còn lại (cột thứ bậc / nhị phân)."""

    def __init__(self, n_so: int):
        self.n_so = n_so

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        gt = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            v = X[~np.isnan(X[:, j]), j]
            if v.size == 0:
                raise ValueError(f"Cột {j} toàn NaN trong tập fit")
            if j < self.n_so:
                gt[j] = np.median(v)
            else:
                u, dem = np.unique(v, return_counts=True)
                gt[j] = u[np.argmax(dem)]   # hoà -> giá trị nhỏ nhất
        self.gia_tri_ = gt
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        vi_tri = np.isnan(X)
        X[vi_tri] = np.broadcast_to(self.gia_tri_, X.shape)[vi_tri]
        return X


def tao_chuan_hoa():
    return StandardScaler()          # fit bỏ qua NaN, transform giữ NaN


def tao_dien_khuyet(ten: str, n_so: int, k_knn: int):
    if ten in ("median_mode", "drop"):   # "drop" dùng bộ này làm dự phòng cho hàng test còn thiếu
        return TrungViTanSuat(n_so)
    if ten == "knn":
        return KNNImputer(n_neighbors=k_knn)
    raise ValueError(f"Nhánh điền khuyết lạ: {ten}")


def tao_pca():
    return PCA(svd_solver="full")    # giữ đủ thành phần; chọn k bằng cách cắt cột


def tao_phan_lop(k_knn: int):
    return KNeighborsClassifier(n_neighbors=k_knn)


def goc_chinh_do(W1, W2) -> np.ndarray:
    """Góc chính (độ) giữa hai không gian con sinh bởi các hàng của W1 và W2 (k x d)."""
    return np.degrees(subspace_angles(np.asarray(W1).T, np.asarray(W2).T))
