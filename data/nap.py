"""
E1 - Nạp và kiểm kê dữ liệu CKD (DA-04, Nhóm 02).

Chạy từ thư mục gốc dự án:
    python data/nap.py --input data/raw/kidney_disease.csv

Đầu ra:
    results/csv/inventory.csv
    results/csv/inventory_summary.json

Các file khác dùng lại hàm nạp này bằng:
    from data.nap import load_ckd, NUMERIC_COLS, CATEGORICAL_COLS, FEATURE_COLS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. SCHEMA CỐ ĐỊNH CỦA BỘ CKD (24 feature + 1 nhãn)
# ----------------------------------------------------------------------
NUMERIC_COLS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
                "hemo", "pcv", "wc", "rc"]                       # 11 cột số
CATEGORICAL_COLS = ["sg", "al", "su", "rbc", "pc", "pcc", "ba",
                    "htn", "dm", "cad", "appet", "pe", "ane"]    # 13 cột danh mục
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS                   # 24 feature
LABEL_COL = "class"
CLASS_VALUES = ("ckd", "notckd")

# sg, al, su trông như số nhưng là thang đo rời rạc (ordinal) -> xếp vào danh mục
ORDINAL_ALLOWED = {
    "sg": {1.005, 1.010, 1.015, 1.020, 1.025},
    "al": {0, 1, 2, 3, 4, 5},
    "su": {0, 1, 2, 3, 4, 5},
}
NOMINAL_ALLOWED = {
    "rbc": {"normal", "abnormal"},
    "pc": {"normal", "abnormal"},
    "pcc": {"present", "notpresent"},
    "ba": {"present", "notpresent"},
    "htn": {"yes", "no"},
    "dm": {"yes", "no"},
    "cad": {"yes", "no"},
    "appet": {"good", "poor"},
    "pe": {"yes", "no"},
    "ane": {"yes", "no"},
}

# Các chuỗi được coi là "thiếu" SAU KHI đã cắt khoảng trắng/tab (vd "\t?" -> "?")
MISSING_TOKENS = {"", "?", "nan", "NaN", "NA", "N/A", "null", "None"}


# ----------------------------------------------------------------------
# 2. HÀM NẠP DỮ LIỆU
# ----------------------------------------------------------------------
def _clean_text(s: pd.Series) -> pd.Series:
    """Cắt khoảng trắng/tab 2 đầu và đổi các token thiếu thành NaN."""
    s = s.astype(object)
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    return s.map(lambda v: np.nan if (isinstance(v, str) and v in MISSING_TOKENS) else v)


def _canonical_ordinal(col: str, s: pd.Series) -> pd.Series:
    """sg/al/su: kiểm tra giá trị hợp lệ rồi viết lại dạng chuẩn ('1.020', '0'...)."""
    values = pd.to_numeric(s, errors="coerce")
    bad = s.notna() & values.isna()
    if bad.any():
        raise ValueError(f"Cột {col} có giá trị không đọc được: {sorted(set(s[bad]))[:10]}")
    allowed = ORDINAL_ALLOWED[col]
    off = values.notna() & ~values.round(3).isin(allowed)
    if off.any():
        raise ValueError(f"Cột {col} có giá trị ngoài thang đo: {sorted(set(values[off]))[:10]}")
    if col == "sg":
        fmt = lambda v: f"{v:.3f}"
    else:
        fmt = lambda v: str(int(round(v)))
    return values.map(lambda v: fmt(v) if pd.notna(v) else np.nan).astype(object)


def load_ckd(path: str | Path) -> pd.DataFrame:
    """
    Đọc file CKD (bản Kaggle kidney_disease.csv) và trả về DataFrame sạch:
      - 400 hàng, 25 cột = 24 feature + 'class'
      - 11 cột số kiểu float64, ô thiếu là NaN
      - 13 cột danh mục kiểu object (chuỗi), ô thiếu là NaN
      - nhãn chỉ gồm 'ckd' / 'notckd'
    Không điền khuyết, không bỏ hàng, không scale -> các bước đó làm sau, trong từng fold.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file dữ liệu: {path.resolve()}")

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    if "id" in raw.columns:
        raw = raw.drop(columns="id")
    if "classification" in raw.columns:
        raw = raw.rename(columns={"classification": LABEL_COL})

    expected = set(FEATURE_COLS) | {LABEL_COL}
    if set(raw.columns) != expected:
        thieu = sorted(expected - set(raw.columns))
        thua = sorted(set(raw.columns) - expected)
        raise ValueError(f"Sai schema. Thiếu cột: {thieu} | Cột lạ: {thua}")

    df = pd.DataFrame(index=raw.index)

    # --- 11 cột số ---
    for col in NUMERIC_COLS:
        text = _clean_text(raw[col])
        values = pd.to_numeric(text, errors="coerce")
        bad = text.notna() & values.isna()
        if bad.any():
            raise ValueError(f"Cột số {col} có giá trị lạ: {sorted(set(text[bad]))[:10]}")
        df[col] = values.astype("float64")

    # --- 13 cột danh mục ---
    for col in CATEGORICAL_COLS:
        text = _clean_text(raw[col])
        if col in ORDINAL_ALLOWED:
            df[col] = _canonical_ordinal(col, text)
        else:
            text = text.map(lambda v: v.lower() if isinstance(v, str) else v)
            off = text.notna() & ~text.isin(NOMINAL_ALLOWED[col])
            if off.any():
                raise ValueError(f"Cột {col} có giá trị lạ: {sorted(set(text[off]))[:10]}")
            df[col] = text.astype(object)

    # --- nhãn ---
    label = _clean_text(raw[LABEL_COL]).map(lambda v: v.lower() if isinstance(v, str) else v)
    if label.isna().any():
        raise ValueError(f"Có {int(label.isna().sum())} hàng thiếu nhãn.")
    off = ~label.isin(CLASS_VALUES)
    if off.any():
        raise ValueError(f"Nhãn lạ: {sorted(set(label[off]))}")
    df[LABEL_COL] = label.astype(object)

    return df[FEATURE_COLS + [LABEL_COL]].reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. KIỂM KÊ (INVENTORY)
# ----------------------------------------------------------------------
def build_inventory(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in FEATURE_COLS:
        miss = int(df[col].isna().sum())
        rows.append({
            "Feature": col,
            "Kind": "numeric" if col in NUMERIC_COLS else "categorical",
            "Dtype": str(df[col].dtype),
            "Missing": miss,
            "Missing_%": round(100 * miss / n, 2),
            "Unique": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame, inventory: pd.DataFrame, source: Path) -> dict:
    n_rows, n_feat = len(df), len(FEATURE_COLS)
    missing_cells = int(df[FEATURE_COLS].isna().sum().sum())
    top5 = inventory.sort_values(["Missing", "Feature"], ascending=[False, True]).head(5)
    return {
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": n_rows,
        "features": n_feat,
        "numeric_features": len(NUMERIC_COLS),
        "categorical_features": len(CATEGORICAL_COLS),
        "missing_cells": missing_cells,
        "missing_percentage": round(100 * missing_cells / (n_rows * n_feat), 4),
        "rows_with_any_missing": int(df[FEATURE_COLS].isna().any(axis=1).sum()),
        "complete_rows": int(df[FEATURE_COLS].notna().all(axis=1).sum()),
        "duplicates": int(df.duplicated().sum()),
        "classes": {c: int((df[LABEL_COL] == c).sum()) for c in CLASS_VALUES},
        "top_5_missing_features": {r.Feature: r.Missing for r in top5.itertuples()},
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 - Nạp và kiểm kê dữ liệu CKD")
    parser.add_argument("--input", default="data/raw/kidney_disease.csv")
    parser.add_argument("--outdir", default="results/csv")
    args = parser.parse_args()

    source = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_ckd(source)
    inventory = build_inventory(df)
    summary = build_summary(df, inventory, source)

    inventory.to_csv(outdir / "inventory.csv", index=False)
    with open(outdir / "inventory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 120)
    print("=" * 60)
    print("E1 - KIỂM KÊ DỮ LIỆU CKD")
    print("=" * 60)
    print(inventory.to_string(index=False))
    print("-" * 60)
    for key in ["rows", "features", "missing_cells", "missing_percentage",
                "rows_with_any_missing", "complete_rows", "duplicates", "classes"]:
        print(f"{key:<24}: {summary[key]}")
    print("-" * 60)
    print(f"Đã ghi: {outdir / 'inventory.csv'}")
    print(f"Đã ghi: {outdir / 'inventory_summary.json'}")


if __name__ == "__main__":
    main()
