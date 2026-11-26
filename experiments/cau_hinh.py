"""
Cấu hình chung của mọi thực nghiệm (DA-04, Nhóm 02).

HAT_GIONG và mọi siêu tham số chỉ đặt ở file này. Các file khác import từ đây,
không tự đặt số riêng. Mọi bộ sinh số ngẫu nhiên lấy qua tao_rng().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Hạt giống và cách chia fold (ràng buộc 4)
# ----------------------------------------------------------------------
HAT_GIONG = 20262027
SO_FOLD = 5            # Repeated Stratified 5-fold
SO_LAN_LAP = 10        # x 10 lần lặp  -> 50 lượt đánh giá mỗi cấu hình

# Chỉ số chính đã chốt, không đổi: F1 của lớp ckd (ckd = lớp dương = 1)
CHI_SO_CHINH = "f1_ckd"
LOP_DUONG = "ckd"

# ----------------------------------------------------------------------
# Siêu tham số của pipeline
# ----------------------------------------------------------------------
NHANH = ("drop", "median_mode", "knn")   # ba cách điền khuyết của đề tài
NHANH_CHINH = "knn"                      # nhánh dùng cho E6 và so sánh phụ
K_PCA_MAC_DINH = 10                      # số thành phần PCA cho E3/E6/thứ tự điền chiếu;
                                         # xem lại sau khi có kết quả E4
K_PCA_QUET = tuple(range(2, 21))         # E4: k = 2..20
K_KNN_PHAN_LOP = 5                       # k của bộ phân lớp KNN
K_KNN_DIEN_KHUYET = 5                    # k của KNN Imputer
LAMBDA_CHIEU = 1e-3                      # hệ số ridge khi chiếu hàng thiếu ô (thứ tự "chiếu trước")

# Backend: "sklearn" dùng models/doi_chieu.py; "tu_viet" dùng mã NumPy của nhóm
# (chỉ bật khi các module tự viết đã merge và có adapter trong doi_chieu.py).
BACKEND = "sklearn"

# Cặp so sánh cho Wilcoxon
MUC_Y_NGHIA = 0.05

# ----------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------
GOC = Path(__file__).resolve().parent.parent
DU_LIEU_THO = GOC / "data" / "raw" / "kidney_disease.csv"
KET_QUA = GOC / "results"
KET_QUA_CSV = KET_QUA / "csv"
KET_QUA_NPZ = KET_QUA / "npz"


def tao_rng(*khoa: int) -> np.random.Generator:
    """Bộ sinh ngẫu nhiên tái lập được: cùng HAT_GIONG + cùng khoá -> cùng dãy số."""
    return np.random.default_rng(np.random.SeedSequence([HAT_GIONG, *khoa]))
