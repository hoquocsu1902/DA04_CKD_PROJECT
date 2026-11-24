"""
Khung chạy thực nghiệm (DA-04, Nhóm 02).

Chạy từ thư mục gốc dự án, với .venv đang bật:
    python -m experiments.chay --thi_nghiem baseline
    python -m experiments.chay --thi_nghiem tat_ca

File này chỉ ghi SỐ ra results/csv/*.csv và results/npz/*.npz. Việc vẽ hình do analysis/ve_hinh.py
đọc lại từ đó. Không sửa tay các file kết quả.
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import cau_hinh as ch
from models import doi_chieu as dc

# tên thực nghiệm -> module chứa hàm thuc_hien(X, y, ten_cot) -> DataFrame
THI_NGHIEM = {
    "baseline": "experiments.baseline",
    "ba_nhanh": "experiments.ba_nhanh",                 # E3 + E4
    "hemoglobin": "experiments.hemoglobin_control",     # E6
    "thu_tu": "experiments.thu_tu_dien_chieu",
}


# ----------------------------------------------------------------------
# Chia fold: Repeated Stratified 5-fold x 10, dùng chung cho mọi thực nghiệm
# ----------------------------------------------------------------------
def tao_folds(y, so_fold: int = ch.SO_FOLD, so_lap: int = ch.SO_LAN_LAP):
    """Trả về danh sách (lan_lap, fold, chi_so_train, chi_so_test).

    Mỗi lần lặp xáo trộn riêng từng lớp rồi chia vòng tròn vào các fold, nên tỉ lệ lớp
    giữ gần như nhau ở mọi fold. Cùng HAT_GIONG -> mọi thực nghiệm dùng đúng cùng bộ fold,
    nhờ vậy các kiểm định cặp (Wilcoxon, Friedman) so sánh trên cùng dữ liệu.
    """
    y = np.asarray(y)
    ra = []
    for r in range(so_lap):
        rng = ch.tao_rng(r)
        gan = np.empty(len(y), dtype=int)
        for lop in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == lop))
            gan[idx] = (np.arange(len(idx)) + rng.integers(so_fold)) % so_fold
        for f in range(so_fold):
            ra.append((r, f, np.flatnonzero(gan != f), np.flatnonzero(gan == f)))
    return ra


# ----------------------------------------------------------------------
# Chỉ số
# ----------------------------------------------------------------------
def so_do(y_that, y_du_doan) -> dict:
    """F1 lớp ckd (ckd = 1), độ chính xác và ma trận nhầm lẫn."""
    y_that, y_du_doan = np.asarray(y_that), np.asarray(y_du_doan)
    tp = int(np.sum((y_that == 1) & (y_du_doan == 1)))
    fp = int(np.sum((y_that == 0) & (y_du_doan == 1)))
    fn = int(np.sum((y_that == 1) & (y_du_doan == 0)))
    tn = int(np.sum((y_that == 0) & (y_du_doan == 0)))
    mau = 2 * tp + fp + fn
    return {
        ch.CHI_SO_CHINH: (2 * tp / mau) if mau else 0.0,
        "do_chinh_xac": (tp + tn) / len(y_that),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ----------------------------------------------------------------------
# Pipeline dùng chung: encode (đã làm ngoài) -> chuẩn hoá -> điền khuyết -> PCA -> KNN
# Mọi bước chỉ fit trên phần TRAIN của fold (ràng buộc 3).
# ----------------------------------------------------------------------
def so_cot_so(ten_cot) -> int:
    """Số cột số còn lại (các cột số luôn đứng trước trong ten_cot)."""
    from data.nap import NUMERIC_COLS
    return sum(c in NUMERIC_COLS for c in ten_cot)


def chuan_bi_nhanh(nhanh, X_tr, y_tr, X_te, n_so):
    """Chuẩn hoá + điền khuyết theo `nhanh`, fit trên train.

    Trả về (Z_tr, y_tr_dung, Z_te, thong_tin). Với nhánh drop, train chỉ giữ hàng đủ 24 ô;
    hàng test còn thiếu được điền bằng median/mode của tập train đó (dự phòng) vì mọi bệnh
    nhân test đều phải có dự đoán. thong_tin ghi số hàng bị bỏ và số hàng test phải dự phòng.
    """
    X_fit, y_fit = X_tr, y_tr
    if nhanh == "drop":
        giu = ~np.isnan(X_tr).any(axis=1)
        X_fit, y_fit = X_tr[giu], y_tr[giu]
    bo_cs = dc.tao_chuan_hoa().fit(X_fit)
    A, B = bo_cs.transform(X_fit), bo_cs.transform(X_te)
    bo_dk = dc.tao_dien_khuyet(nhanh, n_so, ch.K_KNN_DIEN_KHUYET).fit(A)
    tt = {"n_train_bo": int(len(y_tr) - len(y_fit)),
          "n_test_du_phong": int(np.isnan(B).any(axis=1).sum()) if nhanh == "drop" else 0}
    return bo_dk.transform(A), y_fit, bo_dk.transform(B), tt


def pca_phan_lop(Z_tr, y_tr, Z_te, danh_sach_k):
    """Fit PCA một lần trên train, rồi với mỗi k lấy k thành phần đầu, huấn luyện KNN và dự đoán.

    Các thành phần PCA lồng nhau (k thành phần đầu của PCA đầy đủ chính là PCA k), nên cách
    này cho đúng kết quả như fit lại từng k. Trả về {k_thuc: (y_du_doan, components_k, ti_le_pt)}.
    """
    pca = dc.tao_pca().fit(Z_tr)
    S_tr, S_te = pca.transform(Z_tr), pca.transform(Z_te)
    d = S_tr.shape[1]
    ra = {}
    for k in danh_sach_k:
        k = min(k, d)
        clf = dc.tao_phan_lop(ch.K_KNN_PHAN_LOP).fit(S_tr[:, :k], y_tr)
        ra[k] = (clf.predict(S_te[:, :k]), pca.components_[:k].copy(),
                 float(np.sum(pca.explained_variance_ratio_[:k])))
    return ra


# ----------------------------------------------------------------------
# Nạp dữ liệu và ghi kết quả
# ----------------------------------------------------------------------
def nap_du_lieu(duong_dan=ch.DU_LIEU_THO):
    from data.nap import load_ckd
    return dc.ma_hoa(load_ckd(duong_dan))


def ghi_csv(df: pd.DataFrame, ten: str, thu_muc: Path = ch.KET_QUA_CSV) -> Path:
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    p = thu_muc / f"{ten}.csv"
    df.to_csv(p, index=False)
    return p


def ghi_npz(ten: str, thu_muc: Path = ch.KET_QUA_NPZ, **mang) -> Path:
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    p = thu_muc / f"{ten}.npz"
    np.savez_compressed(p, **mang)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy thực nghiệm DA-04 và ghi số ra results/")
    ap.add_argument("--thi_nghiem", default="tat_ca", choices=[*THI_NGHIEM, "tat_ca"])
    ap.add_argument("--du_lieu", default=str(ch.DU_LIEU_THO))
    args = ap.parse_args()

    X, y, ten_cot = nap_du_lieu(args.du_lieu)
    cac = list(THI_NGHIEM) if args.thi_nghiem == "tat_ca" else [args.thi_nghiem]
    for ten in cac:
        t0 = time.time()
        print(f"[{ten}] đang chạy ...", flush=True)
        importlib.import_module(THI_NGHIEM[ten]).thuc_hien(X, y, ten_cot)
        print(f"[{ten}] xong sau {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
