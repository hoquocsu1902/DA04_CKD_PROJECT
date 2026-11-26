"""
Câu hỏi thứ tự: điền khuyết trước rồi giảm chiều, hay giảm chiều trước? (RQ5)

Ba cấu hình, cùng k = K_PCA_MAC_DINH và cùng bộ phân lớp KNN:
    dien_truoc_median   chuẩn hoá -> median/mode -> PCA -> KNN
    dien_truoc_knn      chuẩn hoá -> KNN Imputer -> PCA -> KNN
    chieu_truoc         PCA dựng trên các hàng đủ 24 ô của train; mỗi hàng (kể cả hàng thiếu ô)
                        được chiếu chỉ bằng các ô QUAN SÁT được (bình phương tối thiểu có ridge),
                        không điền ô nào cả -> KNN trên điểm chiếu.

Cách định nghĩa "chiếu trước" là một lựa chọn thiết kế (PCA không chạy được trên ma trận có NaN),
đã ghi trong báo cáo thiết kế thực nghiệm; cần cả nhóm duyệt.

Ghi results/csv/thu_tu_folds.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import cau_hinh as ch
from experiments import chay
from models import doi_chieu as dc


def chieu_hang_thieu(X, W, mu, lam=ch.LAMBDA_CHIEU):
    """Chiếu từng hàng của X (chuẩn hoá, có thể có NaN) lên k trục W (k x d).

    Hàng có tập ô quan sát O giải  min_z ||x_O - mu_O - W_O^T z||^2 + lam ||z||^2,
    z = (W_O W_O^T + lam I)^-1 W_O (x_O - mu_O). Hàng đủ ô và lam -> 0 cho kết quả
    trùng phép chiếu PCA thông thường.
    """
    k = W.shape[0]
    Z = np.empty((X.shape[0], k))
    for i, x in enumerate(X):
        o = ~np.isnan(x)
        Wo = W[:, o]
        Z[i] = np.linalg.solve(Wo @ Wo.T + lam * np.eye(k), Wo @ (x[o] - mu[o]))
    return Z


def chay_chieu_truoc(X_tr, y_tr, X_te, k):
    """Trả về (y_du_doan, so_hang_train_dung_de_dung_PCA)."""
    bo_cs = dc.tao_chuan_hoa().fit(X_tr)                  # thống kê chuẩn hoá bỏ qua NaN
    A, B = bo_cs.transform(X_tr), bo_cs.transform(X_te)
    day_du = ~np.isnan(A).any(axis=1)
    mu = A[day_du].mean(axis=0)
    pca = dc.tao_pca().fit(A[day_du])
    k = min(k, pca.components_.shape[0])
    W = pca.components_[:k]
    clf = dc.tao_phan_lop(ch.K_KNN_PHAN_LOP).fit(chieu_hang_thieu(A, W, mu), y_tr)
    return clf.predict(chieu_hang_thieu(B, W, mu)), int(day_du.sum())


def thuc_hien(X, y, ten_cot, thu_muc=ch.KET_QUA_CSV) -> pd.DataFrame:
    n_so, k = chay.so_cot_so(ten_cot), ch.K_PCA_MAC_DINH
    dong = []
    for r, f, tr, te in chay.tao_folds(y):
        co_dinh = {"thi_nghiem": "thu_tu", "k": k, "lan_lap": r, "fold": f}
        for ten, nhanh in (("dien_truoc_median", "median_mode"), ("dien_truoc_knn", "knn")):
            Z_tr, y_tr, Z_te, _ = chay.chuan_bi_nhanh(nhanh, X[tr], y[tr], X[te], n_so)
            (_, (y_dd, _, _)), = chay.pca_phan_lop(Z_tr, y_tr, Z_te, [k]).items()
            dong.append({**co_dinh, "cau_hinh": ten, **chay.so_do(y[te], y_dd)})
        y_dd, n_pca = chay_chieu_truoc(X[tr], y[tr], X[te], k)
        dong.append({**co_dinh, "cau_hinh": "chieu_truoc", "n_hang_dung_pca": n_pca,
                     **chay.so_do(y[te], y_dd)})
    kq = pd.DataFrame(dong)
    chay.ghi_csv(kq, "thu_tu_folds", thu_muc)
    return kq
