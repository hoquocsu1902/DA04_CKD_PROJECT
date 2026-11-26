"""Kiểm thử khung chạy: chia fold, chỉ số, chống rò rỉ của pipeline."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from experiments import cau_hinh as ch
from experiments import chay
from models import doi_chieu as dc


def test_hat_giong_dung_gia_tri_chot():
    assert ch.HAT_GIONG == 20262027
    assert (ch.SO_FOLD, ch.SO_LAN_LAP) == (5, 10)
    assert ch.CHI_SO_CHINH == "f1_ckd"


def test_tao_rng_tai_lap_va_khac_khoa():
    a, b = ch.tao_rng(3).random(5), ch.tao_rng(3).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ch.tao_rng(4).random(5))


def test_hat_giong_chi_dat_o_cau_hinh():
    from pathlib import Path
    for p in Path("experiments").glob("*.py"):
        if p.name != "cau_hinh.py":
            assert "20262027" not in p.read_text(encoding="utf-8"), p


def test_folds_chia_dung_va_phan_tang(du_lieu_gia):
    _, y, _ = du_lieu_gia
    folds = chay.tao_folds(y)
    assert len(folds) == 50
    for r in range(10):
        test_cua_lan = [te for (rr, _, _, te) in folds if rr == r]
        gop = np.concatenate(test_cua_lan)
        assert sorted(gop) == list(range(len(y)))                  # mỗi bệnh nhân test đúng 1 lần
        for te in test_cua_lan:
            assert abs(y[te].mean() - y.mean()) < 0.02            # tỉ lệ ckd giữ ~62,5%
    for _, _, tr, te in folds:
        assert not set(tr) & set(te)


def test_folds_tai_lap_va_khac_nhau_giua_cac_lan_lap(du_lieu_gia):
    _, y, _ = du_lieu_gia
    f1, f2 = chay.tao_folds(y), chay.tao_folds(y)
    assert all(np.array_equal(a[3], b[3]) for a, b in zip(f1, f2))
    assert not np.array_equal(f1[0][3], f1[5][3])                  # lần lặp 0 khác lần lặp 1


def test_so_do_khop_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(20):
        yt, yp = rng.integers(0, 2, 80), rng.integers(0, 2, 80)
        m = chay.so_do(yt, yp)
        assert m["f1_ckd"] == pytest.approx(f1_score(yt, yp, pos_label=1))
        assert m["do_chinh_xac"] == pytest.approx(accuracy_score(yt, yp))
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (tn, fp, fn, tp)
    assert chay.so_do([0, 0], [0, 0])["f1_ckd"] == 0.0               # không có ca dương nào: không chia 0


# ---------------------- chống rò rỉ (ràng buộc 3) ----------------------
@pytest.mark.parametrize("nhanh", ch.NHANH)
def test_transform_khong_dung_thong_ke_du_lieu_moi(du_lieu_gia, nhanh):
    """Đổi hẳn dữ liệu test (hoặc bỏ bớt hàng) thì biểu diễn của các hàng còn lại không đổi."""
    X, y, ten = du_lieu_gia
    n_so = chay.so_cot_so(ten)
    folds = chay.tao_folds(y)
    _, _, tr, te = folds[0]
    _, _, Z_te_a, _ = chay.chuan_bi_nhanh(nhanh, X[tr], y[tr], X[te], n_so)
    # (1) cùng train, test chỉ còn nửa đầu -> biểu diễn nửa đầu y hệt
    nua = len(te) // 2
    _, _, Z_te_b, _ = chay.chuan_bi_nhanh(nhanh, X[tr], y[tr], X[te][:nua], n_so)
    assert np.allclose(Z_te_a[:nua], Z_te_b, atol=1e-10)
    # (2) test bị thay bằng số cực lớn ở các hàng khác -> hàng đầu không đổi
    X_te2 = X[te].copy()
    X_te2[nua:] = 1e6
    _, _, Z_te_c, _ = chay.chuan_bi_nhanh(nhanh, X[tr], y[tr], X_te2, n_so)
    assert np.allclose(Z_te_a[0], Z_te_c[0], atol=1e-10)


def test_bo_dien_khuyet_chi_dung_thong_ke_cua_tap_fit(du_lieu_gia):
    X, _, ten = du_lieu_gia
    n_so = chay.so_cot_so(ten)
    bo = dc.TrungViTanSuat(n_so).fit(X[:200])
    gt_truoc = bo.gia_tri_.copy()
    out = bo.transform(X[200:] * 1000 + 7)             # dữ liệu mới rất khác, không được ảnh hưởng
    assert np.array_equal(bo.gia_tri_, gt_truoc)
    assert not np.isnan(out).any()


def test_trung_vi_tan_suat_khop_simple_imputer(du_lieu_gia):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    X, _, ten = du_lieu_gia
    n = chay.so_cot_so(ten)
    ref = ColumnTransformer([("a", SimpleImputer(strategy="median"), slice(0, n)),
                             ("b", SimpleImputer(strategy="most_frequent"), slice(n, X.shape[1]))])
    assert np.allclose(ref.fit_transform(X), dc.TrungViTanSuat(n).fit(X).transform(X))


def test_drop_chi_giu_hang_du_o_va_hang_test_van_duoc_du_doan(du_lieu_gia):
    X, y, ten = du_lieu_gia
    _, _, tr, te = chay.tao_folds(y)[0]
    Z_tr, y_tr, Z_te, tt = chay.chuan_bi_nhanh("drop", X[tr], y[tr], X[te], chay.so_cot_so(ten))
    assert len(Z_tr) == len(y_tr) == (~np.isnan(X[tr]).any(axis=1)).sum()
    assert tt["n_train_bo"] == len(tr) - len(y_tr)
    assert Z_te.shape[0] == len(te) and not np.isnan(Z_te).any()


def test_ma_hoa_giu_nan_va_thang_do(du_lieu_gia):
    X, y, ten = du_lieu_gia
    assert X.shape == (400, 24) and set(np.unique(y)) == {0, 1} and y.sum() == 250
    assert np.isnan(X).any()
    j = ten.index("htn")
    assert set(np.unique(X[~np.isnan(X[:, j]), j])) <= {0.0, 1.0}
