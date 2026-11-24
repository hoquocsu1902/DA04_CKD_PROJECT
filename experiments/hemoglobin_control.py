"""
E6 - Mô hình có dựa quá nhiều vào Hemoglobin không? (RQ3)

Cùng pipeline của nhánh NHANH_CHINH (chuẩn hoá -> điền khuyết -> PCA -> KNN), bốn biến thể cột:
    tat_ca          24 feature
    bo_hemo         bỏ hemo
    bo_huyet_hoc    bỏ hemo, pcv, rc (ba cột đo cùng một thứ: lượng hồng cầu), để kiểm tra
                    việc bỏ riêng hemo có bị pcv/rc "gánh thay" không
    chi_hemo        chỉ giữ hemo
Số thành phần PCA là min(K_PCA_MAC_DINH, số cột còn lại).

Ghi results/csv/hemoglobin_folds.csv.
"""
from __future__ import annotations

import pandas as pd

from experiments import cau_hinh as ch
from experiments import chay

BO_CAC_COT = {
    "tat_ca": (),
    "bo_hemo": ("hemo",),
    "bo_huyet_hoc": ("hemo", "pcv", "rc"),
    "chi_hemo": None,       # None = chỉ giữ hemo
}


def _chon_cot(ten_cot, bien_the):
    if BO_CAC_COT[bien_the] is None:
        return [i for i, c in enumerate(ten_cot) if c == "hemo"]
    return [i for i, c in enumerate(ten_cot) if c not in BO_CAC_COT[bien_the]]


def thuc_hien(X, y, ten_cot, thu_muc=ch.KET_QUA_CSV) -> pd.DataFrame:
    dong = []
    cot = {bt: _chon_cot(ten_cot, bt) for bt in BO_CAC_COT}
    for r, f, tr, te in chay.tao_folds(y):
        for bt, idx in cot.items():
            ten_con = [ten_cot[i] for i in idx]
            Z_tr, y_tr, Z_te, _ = chay.chuan_bi_nhanh(
                ch.NHANH_CHINH, X[tr][:, idx], y[tr], X[te][:, idx], chay.so_cot_so(ten_con))
            (k, (y_dd, _, _)), = chay.pca_phan_lop(Z_tr, y_tr, Z_te, [ch.K_PCA_MAC_DINH]).items()
            dong.append({"thi_nghiem": "E6", "cau_hinh": bt, "nhanh": ch.NHANH_CHINH,
                         "k": k, "n_cot": len(idx), "lan_lap": r, "fold": f,
                         **chay.so_do(y[te], y_dd)})
    kq = pd.DataFrame(dong)
    chay.ghi_csv(kq, "hemoglobin_folds", thu_muc)
    return kq
