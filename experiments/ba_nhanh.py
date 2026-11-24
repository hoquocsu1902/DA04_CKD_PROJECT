"""
E3 (ba nhánh) và E4 (quét số thành phần PCA k = 2..20).

Mỗi lượt fold, với từng nhánh Drop / Median-Mode / KNN Imputer:
    chuẩn hoá -> điền khuyết -> PCA -> KNN, mọi bước chỉ fit trên train.

Đầu ra:
    results/csv/ba_nhanh_folds.csv   E3: k = K_PCA_MAC_DINH
    results/csv/quet_pca_folds.csv   E4: mọi k trong K_PCA_QUET, kèm phương sai tích luỹ
    results/npz/ba_nhanh_thanh_phan.npz   các trục PCA (k x 24) của từng nhánh x lượt fold,
                                           để E5 (góc chính, Bảo Khang) tính góc lệch giữa nhánh
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import cau_hinh as ch
from experiments import chay


def thuc_hien(X, y, ten_cot, thu_muc_csv=ch.KET_QUA_CSV, thu_muc_npz=ch.KET_QUA_NPZ):
    n_so = chay.so_cot_so(ten_cot)
    ds_k = sorted(set(ch.K_PCA_QUET) | {ch.K_PCA_MAC_DINH})
    e3, e4 = [], []
    trục, nhan_lan, nhan_fold = [], [], []
    for r, f, tr, te in chay.tao_folds(y):
        trục_fold = []
        for nhanh in ch.NHANH:
            Z_tr, y_tr, Z_te, tt = chay.chuan_bi_nhanh(nhanh, X[tr], y[tr], X[te], n_so)
            kq = chay.pca_phan_lop(Z_tr, y_tr, Z_te, ds_k)
            for k, (y_dd, W, pt) in kq.items():
                d = {"nhanh": nhanh, "k": k, "lan_lap": r, "fold": f,
                     **chay.so_do(y[te], y_dd), "phuong_sai_tich_luy": pt, **tt}
                e4.append(d)
                if k == ch.K_PCA_MAC_DINH:
                    e3.append({"thi_nghiem": "E3", "cau_hinh": nhanh, **d})
                    trục_fold.append(W)
        trục.append(np.stack(trục_fold)); nhan_lan.append(r); nhan_fold.append(f)
    e3, e4 = pd.DataFrame(e3), pd.DataFrame(e4)
    chay.ghi_csv(e3, "ba_nhanh_folds", thu_muc_csv)
    chay.ghi_csv(e4, "quet_pca_folds", thu_muc_csv)
    chay.ghi_npz("ba_nhanh_thanh_phan", thu_muc_npz,
                 thanh_phan=np.stack(trục),            # (luot, nhanh, k, d)
                 lan_lap=np.array(nhan_lan), fold=np.array(nhan_fold),
                 nhanh=np.array(ch.NHANH), ten_cot=np.array(ten_cot))
    return e3, e4
