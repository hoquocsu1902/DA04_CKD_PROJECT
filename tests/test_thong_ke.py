"""Kiểm thử analysis/thong_ke.py: Wilcoxon, Friedman, Nemenyi, mean ± SD."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis import thong_ke as tk


def test_mean_sd_dung_ddof1():
    m, s = tk.mean_sd([1, 2, 3, 4])
    assert m == 2.5 and s == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert tk.dinh_dang([0.5, 0.5, 0.5]) == "0.5000 ± 0.0000"
    assert tk.mean_sd([3.0])[1] == 0.0


def test_wilcoxon_khop_scipy_va_xu_ly_hai_day_trung():
    rng = np.random.default_rng(0)
    a = rng.normal(0.95, 0.02, 50)
    b = a - np.abs(rng.normal(0.01, 0.01, 50))
    r = tk.wilcoxon_hai(a, b)
    assert r["p"] == pytest.approx(stats.wilcoxon(a, b).pvalue) and r["p"] < 0.01
    assert tk.wilcoxon_hai(a, a)["p"] == 1.0          # scipy sẽ báo lỗi, ta trả p = 1


def test_friedman_khop_scipy_va_hang():
    rng = np.random.default_rng(1)
    M = np.column_stack([rng.normal(m, 0.01, 50) for m in (0.90, 0.95, 0.97)])
    r = tk.friedman_nemenyi(M, ["a", "b", "c"])
    assert r["friedman_p"] == pytest.approx(stats.friedmanchisquare(*M.T).pvalue)
    assert list(r["hang_trung_binh"].sort_values().index) == ["c", "b", "a"]   # hạng 1 = F1 cao nhất


def test_nemenyi_doi_xung_va_khop_cong_thuc_tay():
    rng = np.random.default_rng(2)
    n, k = 40, 4
    M = rng.normal(size=(n, k)) + np.array([0, 0.2, 0.4, 1.5])
    P = tk.friedman_nemenyi(M)["nemenyi_p"].to_numpy()
    assert np.allclose(P, P.T) and np.allclose(np.diag(P), 1.0)
    hang = np.vstack([stats.rankdata(-h) for h in M]).mean(axis=0)
    q = abs(hang[0] - hang[3]) / np.sqrt(k * (k + 1) / (12 * n))      # dạng q của Demšar: sqrt(k(k+1)/12n)
    assert P[0, 3] == pytest.approx(stats.studentized_range.sf(q, k, np.inf))
    assert P[0, 3] < 0.001 and P[0, 1] > 0.05                          # cách xa -> khác biệt; sát nhau -> không


def test_nemenyi_hai_pp_giong_nhau_cho_p_bang_1():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 1))
    M = np.hstack([x, x, x - 5])               # hai cột trùng hẳn -> hạng bằng nhau
    P = tk.friedman_nemenyi(M)["nemenyi_p"].to_numpy()
    assert P[0, 1] == pytest.approx(1.0) and P[0, 2] < 0.001


def _df_folds(rng, hieu_cau_hinh):
    dong = []
    for r in range(10):
        for f in range(5):
            nen = rng.normal(0.95, 0.02)
            for ten, dich in hieu_cau_hinh.items():
                dong.append({"cau_hinh": ten, "lan_lap": r, "fold": f, "f1_ckd": nen + dich,
                             "do_chinh_xac": nen, "tn": 30, "fp": 1, "fn": 2, "tp": 47})
    return pd.DataFrame(dong)


def test_tom_tat_va_so_sanh_tren_bang_ket_qua():
    df = _df_folds(np.random.default_rng(4), {"x": 0.0, "y": 0.01, "z": -0.05})
    t = tk.tom_tat(df, "cau_hinh")
    assert set(t["cau_hinh"]) == {"x", "y", "z"} and (t["n_luot"] == 50).all()
    assert t.loc[t.cau_hinh == "z", "tp_tong"].iloc[0] == 50 * 47
    s = tk.so_sanh(df, "cau_hinh")
    assert len(s) == 3 and {"wilcoxon_p", "nemenyi_p", "friedman_p"} <= set(s.columns)
    assert s.loc[(s.a == "x") & (s.b == "z"), "wilcoxon_p"].iloc[0] < 0.001


def test_so_sanh_hai_cau_hinh_chi_co_wilcoxon():
    df = _df_folds(np.random.default_rng(5), {"x": 0.0, "y": 0.02})
    s = tk.so_sanh(df, "cau_hinh")
    assert len(s) == 1 and "nemenyi_p" not in s.columns


def test_so_sanh_bao_loi_khi_thieu_luot_fold():
    df = _df_folds(np.random.default_rng(6), {"x": 0.0, "y": 0.02})
    with pytest.raises(ValueError):
        tk.so_sanh(df.drop(index=3), "cau_hinh")
