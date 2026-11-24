"""
Bốn baseline (DA-04, Nhóm 02). Dùng scikit-learn, vì baseline chỉ để làm mốc so sánh.

  da_so          luôn đoán lớp ckd (sàn của F1)
  logistic       median/mode -> chuẩn hoá -> hồi quy logistic, không PCA
  knn_khong_pca  median/mode -> chuẩn hoá -> KNN, không PCA (để thấy PCA giúp hay hại)
  rung_ngau_nhien median/mode -> random forest

Ghi results/csv/baseline_folds.csv (mỗi dòng = một cấu hình x một lượt fold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from experiments import cau_hinh as ch
from experiments import chay
from models import doi_chieu as dc

TEN_BASELINE = ("da_so", "logistic", "knn_khong_pca", "rung_ngau_nhien")


def _tao_mo_hinh(ten: str, lan_lap: int, fold: int):
    seed = int(ch.tao_rng(lan_lap, fold, 99).integers(2**31 - 1))   # hạt riêng cho từng lượt
    if ten == "da_so":
        return DummyClassifier(strategy="most_frequent")
    if ten == "logistic":
        return LogisticRegression(max_iter=1000)
    if ten == "knn_khong_pca":
        return dc.tao_phan_lop(ch.K_KNN_PHAN_LOP)
    if ten == "rung_ngau_nhien":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    raise ValueError(ten)


def thuc_hien(X, y, ten_cot, thu_muc=ch.KET_QUA_CSV) -> pd.DataFrame:
    n_so = chay.so_cot_so(ten_cot)
    dong = []
    for r, f, tr, te in chay.tao_folds(y):
        # cùng một bước tiền xử lý cho cả 4 baseline: fit trên train rồi áp lên test
        Z_tr, y_tr, Z_te, _ = chay.chuan_bi_nhanh("median_mode", X[tr], y[tr], X[te], n_so)
        for ten in TEN_BASELINE:
            mh = _tao_mo_hinh(ten, r, f).fit(Z_tr, y_tr)
            dong.append({"thi_nghiem": "baseline", "cau_hinh": ten, "lan_lap": r, "fold": f,
                         **chay.so_do(y[te], mh.predict(Z_te))})
    kq = pd.DataFrame(dong)
    chay.ghi_csv(kq, "baseline_folds", thu_muc)
    return kq
