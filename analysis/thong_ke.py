"""
Thống kê cho kết quả thực nghiệm (DA-04, Nhóm 02).

  - mean ± SD (SD mẫu, ddof = 1) trên 50 lượt fold
  - Wilcoxon signed-rank cho hai phương pháp (cặp theo lượt fold)
  - Friedman + hậu kiểm Nemenyi cho từ ba phương pháp trở lên

Lưu ý khi đọc p-value: 50 lượt fold của Repeated CV dùng chung dữ liệu nên không độc lập hoàn
toàn; p-value dễ nhỏ hơn thực tế. Báo cáo cần nói rõ điều này.

Chạy:  python -m analysis.thong_ke   (đọc results/csv/*_folds.csv, ghi *_tom_tat.csv, *_so_sanh.csv)
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments import cau_hinh as ch

# file kết quả -> (cột tên cấu hình, cột nhóm hoặc None). So sánh các cấu hình trong từng nhóm.
KE_HOACH = {
    "baseline_folds": ("cau_hinh", None),
    "ba_nhanh_folds": ("cau_hinh", None),
    "quet_pca_folds": ("k", "nhanh"),
    "hemoglobin_folds": ("cau_hinh", None),
    "thu_tu_folds": ("cau_hinh", None),
}


def mean_sd(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def dinh_dang(x, so_le: int = 4) -> str:
    m, s = mean_sd(x)
    return f"{m:.{so_le}f} ± {s:.{so_le}f}"


def wilcoxon_hai(a, b) -> dict:
    """Wilcoxon signed-rank hai phía trên các cặp (a_i, b_i)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    hieu = a - b
    ra = {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
          "hieu_trung_binh": float(hieu.mean()), "n_cap": len(a)}
    if np.allclose(hieu, 0):                 # hai dãy trùng nhau: không có gì để kiểm định
        return {**ra, "thong_ke": 0.0, "p": 1.0}
    kq = stats.wilcoxon(a, b)
    return {**ra, "thong_ke": float(kq.statistic), "p": float(kq.pvalue)}


def friedman_nemenyi(ma_tran, ten=None) -> dict:
    """ma_tran: (n_khoi x k_phuong_phap), mỗi hàng là một lượt fold.

    Trả về {"friedman_chi2", "friedman_p", "hang_trung_binh" (Series, 1 = tốt nhất),
            "nemenyi_p" (DataFrame đối xứng k x k)}.
    Nemenyi: z = |R_i - R_j| / sqrt(k(k+1)/(6n)), p = SF của studentized range tại z*sqrt(2),
    k bậc tự do thứ nhất, vô hạn bậc tự do thứ hai (Demšar 2006).
    """
    M = np.asarray(ma_tran, dtype=float)
    n, k = M.shape
    ten = list(ten) if ten is not None else [f"pp{i}" for i in range(k)]
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    # hạng 1 = giá trị lớn nhất (F1 cao là tốt); hoà -> hạng trung bình
    hang = np.vstack([stats.rankdata(-hang_i) for hang_i in M]).mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    P = np.ones((k, k))
    for i, j in itertools.combinations(range(k), 2):
        z = abs(hang[i] - hang[j]) / se
        P[i, j] = P[j, i] = float(stats.studentized_range.sf(z * np.sqrt(2), k, np.inf))
    return {"friedman_chi2": float(chi2), "friedman_p": float(p),
            "hang_trung_binh": pd.Series(hang, index=ten),
            "nemenyi_p": pd.DataFrame(P, index=ten, columns=ten)}


def _ma_tran(df, cot_cau_hinh, chi_so):
    """Pivot về (lượt fold x cấu hình); các cấu hình phải có đủ cùng bộ lượt fold."""
    pv = df.pivot_table(index=["lan_lap", "fold"], columns=cot_cau_hinh, values=chi_so, aggfunc="first")
    if pv.isna().any().any():
        raise ValueError("Các cấu hình không có cùng bộ lượt fold, không so sánh cặp được")
    return pv


def tom_tat(df, cot_cau_hinh, cot_nhom=None, chi_so_cac=(ch.CHI_SO_CHINH, "do_chinh_xac")) -> pd.DataFrame:
    khoa = ([cot_nhom] if cot_nhom else []) + [cot_cau_hinh]
    dong = []
    for gt, g in df.groupby(khoa, sort=False):
        gt = gt if isinstance(gt, tuple) else (gt,)
        d = dict(zip(khoa, gt))
        for cs in chi_so_cac:
            m, s = mean_sd(g[cs])
            d[f"{cs}_mean"], d[f"{cs}_sd"], d[cs] = m, s, dinh_dang(g[cs])
        for o in ("tn", "fp", "fn", "tp"):
            d[f"{o}_tong"] = int(g[o].sum())     # ma trận nhầm lẫn cộng dồn trên mọi lượt fold
        d["n_luot"] = len(g)
        dong.append(d)
    return pd.DataFrame(dong)


def so_sanh(df, cot_cau_hinh, cot_nhom=None, chi_so=ch.CHI_SO_CHINH) -> pd.DataFrame:
    """Bảng p-value: Wilcoxon từng cặp (và Friedman + Nemenyi khi có từ 3 cấu hình trở lên)."""
    nhom = df.groupby(cot_nhom, sort=False) if cot_nhom else [(None, df)]
    dong = []
    for ten_nhom, g in nhom:
        pv = _ma_tran(g, cot_cau_hinh, chi_so)
        cac = list(pv.columns)
        fn = friedman_nemenyi(pv.to_numpy(), cac) if len(cac) >= 3 else None
        for a, b in itertools.combinations(cac, 2):
            w = wilcoxon_hai(pv[a], pv[b])
            d = {"nhom": ten_nhom, "a": a, "b": b, "mean_a": w["mean_a"], "mean_b": w["mean_b"],
                 "hieu_trung_binh": w["hieu_trung_binh"], "wilcoxon_p": w["p"]}
            if fn:
                d["nemenyi_p"] = float(fn["nemenyi_p"].loc[a, b])
                d["friedman_p"] = fn["friedman_p"]
            dong.append(d)
    return pd.DataFrame(dong)


def main() -> None:
    thu_muc = Path(ch.KET_QUA_CSV)
    for ten, (cot, nhom) in KE_HOACH.items():
        p = thu_muc / f"{ten}.csv"
        if not p.exists():
            print(f"bỏ qua {ten}: chưa có {p.name} (chạy experiments.chay trước)")
            continue
        df = pd.read_csv(p)
        goc = ten.removesuffix("_folds")
        tom_tat(df, cot, nhom).to_csv(thu_muc / f"{goc}_tom_tat.csv", index=False)
        so_sanh(df, cot, nhom).to_csv(thu_muc / f"{goc}_so_sanh.csv", index=False)
        print(f"đã ghi {goc}_tom_tat.csv, {goc}_so_sanh.csv")


if __name__ == "__main__":
    main()
