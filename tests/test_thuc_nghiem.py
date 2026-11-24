"""Smoke test các thực nghiệm trên dữ liệu GIẢ (ghi vào thư mục tạm, không đụng results/)."""
import numpy as np
import pandas as pd
import pytest

from analysis import thong_ke as tk
from experiments import baseline, ba_nhanh, hemoglobin_control, thu_tu_dien_chieu
from experiments import cau_hinh as ch
from experiments import chay


@pytest.fixture(scope="module")
def nho(du_lieu_gia):
    """Dữ liệu giả rút gọn 200 hàng để test chạy nhanh."""
    X, y, ten = du_lieu_gia
    idx = np.r_[0:125, 250:325]
    return X[idx], y[idx], ten


def test_baseline(nho, tmp_path):
    X, y, ten = nho
    kq = baseline.thuc_hien(X, y, ten, tmp_path)
    assert len(kq) == 50 * 4 and set(kq.cau_hinh) == set(baseline.TEN_BASELINE)
    f1 = kq.groupby("cau_hinh").f1_ckd.mean()
    assert f1["da_so"] == pytest.approx(2 * 0.625 / 1.625, abs=1e-6)    # luôn đoán ckd, ckd = 62,5%
    assert f1["rung_ngau_nhien"] > f1["da_so"]
    assert (tmp_path / "baseline_folds.csv").exists()


def test_baseline_tai_lap(nho, tmp_path):
    X, y, ten = nho
    a = baseline.thuc_hien(X, y, ten, tmp_path / "a")
    b = baseline.thuc_hien(X, y, ten, tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_e3_e4_va_thanh_phan(nho, tmp_path):
    X, y, ten = nho
    e3, e4 = ba_nhanh.thuc_hien(X, y, ten, tmp_path, tmp_path)
    assert len(e3) == 50 * 3 and set(e3.cau_hinh) == set(ch.NHANH)
    assert len(e4) == 50 * 3 * len(ch.K_PCA_QUET)
    assert sorted(e4.k.unique()) == list(ch.K_PCA_QUET)
    # E3 chính là hàng k = K_PCA_MAC_DINH của E4
    a = e3.sort_values(["nhanh", "lan_lap", "fold"]).f1_ckd.to_numpy()
    b = e4[e4.k == ch.K_PCA_MAC_DINH].sort_values(["nhanh", "lan_lap", "fold"]).f1_ckd.to_numpy()
    assert np.array_equal(a, b)
    # phương sai tích luỹ tăng theo k
    pt = e4[(e4.nhanh == "knn")].groupby("k").phuong_sai_tich_luy.mean()
    assert pt.is_monotonic_increasing and pt.iloc[-1] <= 1 + 1e-9
    # nhánh drop bỏ hàng; hai nhánh kia không bỏ
    assert (e3[e3.nhanh == "drop"].n_train_bo > 0).all()
    assert (e3[e3.nhanh != "drop"].n_train_bo == 0).all()
    z = np.load(tmp_path / "ba_nhanh_thanh_phan.npz")
    assert z["thanh_phan"].shape == (50, 3, ch.K_PCA_MAC_DINH, 24)
    W = z["thanh_phan"][0, 0]
    assert np.allclose(W @ W.T, np.eye(ch.K_PCA_MAC_DINH), atol=1e-8)       # trục trực chuẩn


def test_e4_gon_voi_pca_fit_lai_tung_k(nho):
    """Cắt k cột đầu của PCA đầy đủ == fit PCA k thành phần: kiểm tra cách tiết kiệm không đổi kết quả."""
    from sklearn.decomposition import PCA
    X, y, ten = nho
    _, _, tr, te = chay.tao_folds(y)[0]
    Z_tr, y_tr, Z_te, _ = chay.chuan_bi_nhanh("median_mode", X[tr], y[tr], X[te], chay.so_cot_so(ten))
    kq = chay.pca_phan_lop(Z_tr, y_tr, Z_te, [3, 8])
    for k in (3, 8):
        p = PCA(n_components=k, svd_solver="full").fit(Z_tr)
        S_tr, S_te = p.transform(Z_tr), p.transform(Z_te)
        from sklearn.neighbors import KNeighborsClassifier
        dd = KNeighborsClassifier(ch.K_KNN_PHAN_LOP).fit(S_tr, y_tr).predict(S_te)
        assert np.array_equal(dd, kq[k][0])


def test_e6(nho, tmp_path):
    X, y, ten = nho
    kq = hemoglobin_control.thuc_hien(X, y, ten, tmp_path)
    assert set(kq.cau_hinh) == set(hemoglobin_control.BO_CAC_COT) and len(kq) == 50 * 4
    n_cot = kq.groupby("cau_hinh").n_cot.first().to_dict()
    assert n_cot == {"tat_ca": 24, "bo_hemo": 23, "bo_huyet_hoc": 21, "chi_hemo": 1}
    assert kq[kq.cau_hinh == "chi_hemo"].k.eq(1).all()                       # k bị chặn bởi số cột


def test_chieu_hang_thieu_khop_pca_khi_du_o():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    A = rng.normal(size=(60, 8))
    pca = PCA(n_components=3, svd_solver="full").fit(A)
    mu = A.mean(axis=0)
    Z = thu_tu_dien_chieu.chieu_hang_thieu(A, pca.components_, mu, lam=0.0)
    assert np.allclose(Z, pca.transform(A), atol=1e-8)


def test_chieu_hang_thieu_chi_dung_o_quan_sat():
    rng = np.random.default_rng(1)
    W = np.linalg.qr(rng.normal(size=(8, 3)))[0].T          # 3 x 8, trực chuẩn
    mu = np.zeros(8)
    x = rng.normal(size=8)
    x1, x2 = x.copy(), x.copy()
    x1[[2, 5]] = np.nan
    x2[[2, 5]] = np.nan
    x2[0:1] = x[0:1]                                           # cùng ô quan sát
    z1 = thu_tu_dien_chieu.chieu_hang_thieu(x1[None], W, mu)
    # đổi giá trị ở ô bị thiếu (không thể ảnh hưởng vì là NaN) -> luôn không NaN và hữu hạn
    assert np.isfinite(z1).all()
    x3 = x1.copy(); x3[7] = 100.0                              # đổi một ô quan sát -> kết quả phải đổi
    assert not np.allclose(z1, thu_tu_dien_chieu.chieu_hang_thieu(x3[None], W, mu))


def test_thu_tu(nho, tmp_path):
    X, y, ten = nho
    kq = thu_tu_dien_chieu.thuc_hien(X, y, ten, tmp_path)
    assert set(kq.cau_hinh) == {"dien_truoc_median", "dien_truoc_knn", "chieu_truoc"}
    assert len(kq) == 150 and kq.f1_ckd.between(0, 1).all()
    assert kq[kq.cau_hinh == "chieu_truoc"].n_hang_dung_pca.min() > ch.K_PCA_MAC_DINH


def test_pipeline_thong_ke_tu_dau_den_cuoi(nho, tmp_path, monkeypatch):
    X, y, ten = nho
    baseline.thuc_hien(X, y, ten, tmp_path)
    ba_nhanh.thuc_hien(X, y, ten, tmp_path, tmp_path)
    monkeypatch.setattr(ch, "KET_QUA_CSV", tmp_path)
    tk.main()
    for f in ("baseline_tom_tat", "baseline_so_sanh", "ba_nhanh_tom_tat", "ba_nhanh_so_sanh",
              "quet_pca_tom_tat", "quet_pca_so_sanh"):
        assert (tmp_path / f"{f}.csv").exists(), f
    ss = pd.read_csv(tmp_path / "ba_nhanh_so_sanh.csv")
    assert len(ss) == 3 and ss.wilcoxon_p.between(0, 1).all()
