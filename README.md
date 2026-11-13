# DA-04 — Phát hiện bệnh thận mạn (CKD) từ kết quả xét nghiệm

Đồ án cuối kỳ môn **Học máy**, HK1 năm học 2026–2027, Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT).
**Nhóm 02 — lớp 15DHTH09.**

> Đây là dự án nghiên cứu phục vụ học tập. Kết quả mô hình **không** dùng để chẩn đoán lâm sàng.

## Câu hỏi nghiên cứu

> Ba cách xử lý dữ liệu khuyết tạo ra các hệ trục PCA lệch nhau bao nhiêu độ, và độ lệch ấy có làm thay đổi kết quả phân lớp CKD không?

- **RQ1.** Ba nhánh Drop / Median-Mode / KNN Imputer làm hệ trục PCA lệch nhau bao nhiêu độ (principal angles)?
- **RQ2.** Góc lệch đó có đi kèm thay đổi đáng kể F1 của lớp CKD không?
- **RQ3.** Mô hình có phụ thuộc quá nhiều vào Hemoglobin (`hemo`) không?
- **RQ4.** Bệnh nhân thiếu từ 5 ô trở lên có tập trung vào vùng/cụm PCA nào không?
- **RQ5.** Điền khuyết rồi giảm chiều, hay giảm chiều trước rồi mới điền: cách nào đúng hơn, và bằng chứng nào?

**Hướng mở rộng (chiều sâu).** Che dấu các ô vốn đầy đủ theo ba cơ chế MCAR / MAR / MNAR ở bốn tỉ lệ 5 / 10 / 20 / 30%, đo trực tiếp sai số điền khuyết và góc lệch so với hệ trục dựng từ dữ liệu chân lý gốc; thêm điền khuyết lặp và điền khuyết nhiều lần. Câu hỏi: cách điền có sai số nhỏ nhất có phải cách cho mô hình tốt nhất không?

## Thành viên

| Họ tên | MSSV | Vai trò |
|---|---|---|
| Lê Gia Bảo | 2001240028 | Nhóm trưởng — Experiment & Integration |
| Hồ Quốc Sử | 2001240412 | Data Pipeline, EDA, Web Foundation |
| Nguyễn Ngọc Bảo | 2045240021 | KNN, KNN Imputer tự viết; điền khuyết lặp |
| Phan Hoài Bảo Khang | 2045240120 | PCA-SVD, Principal Angles, Clustering, Visualization; điền khuyết nhiều lần |

## Tiến độ

| Hạng mục | Nội dung | Trạng thái |
|---|---|---|
| E1 | Nạp và kiểm kê dữ liệu (`data/nap.py`) | Hoàn thành |
| E2 | EDA — 10 biểu đồ, trọng tâm bản đồ ô thiếu 400 × 24 | Chưa bắt đầu |
| Tiền xử lý | Mã hoá, chuẩn hoá, ba nhánh điền khuyết, kiểm thử chống rò rỉ dữ liệu | Chưa bắt đầu |
| Mô hình tự viết | KNN, KNN Imputer, PCA qua SVD, principal angles (NumPy) | Chưa bắt đầu |
| E3–E8 | Ba nhánh, quét PCA k = 2..20, góc chính, Hemoglobin, artificial missing, clustering + ARI | Chưa bắt đầu |
| Mở rộng | Bộ che dấu MCAR/MAR/MNAR, điền khuyết lặp, điền khuyết nhiều lần | Chưa bắt đầu |
| Web | FastAPI, 6 màn hình, chạy offline | Chưa bắt đầu |
| Báo cáo | 15–25 trang PDF | Chưa bắt đầu |

## Dữ liệu
Tải file từ Kaggle và đặt đúng đường dẫn `data/raw/kidney_disease.csv`. Thư mục này không đưa lên git theo quy định của tài liệu 07b. Test `test_dung_file_du_lieu` kiểm tra mã SHA-256 để chắc mọi thành viên dùng cùng một file.

**Nguồn:** Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020 — giấy phép **CC BY 4.0**.
File dùng trong dự án là bản CSV trên Kaggle: https://www.kaggle.com/datasets/akshayksingh/kidney-disease-dataset, lưu tại `data/raw/kidney_disease.csv`.

Kết quả kiểm kê E1 (`results/csv/inventory_summary.json`):

| Chỉ số | Giá trị |
|---|---|
| Số bệnh nhân | 400 |
| Số feature | 24 (11 số, 13 danh mục; `sg`, `al`, `su` là thang thứ bậc) |
| Nhãn | `ckd` 250, `notckd` 150 |
| Ô thiếu | 1012 / 9600 (10,54%) |
| Bệnh nhân thiếu ít nhất 1 ô | 242 |
| Bệnh nhân đầy đủ | 158 |
| Hàng trùng lặp | 0 |
| Thiếu nhiều nhất | `rbc` 38,00%, `rc` 32,75%, `wc` 26,50%, `pot` 22,00%, `sod` 21,75% |

Ở bước nạp, `data/nap.py` chỉ làm sạch: cắt khoảng trắng/tab, đổi `?` và ô rỗng thành NaN, bỏ cột `id`, đổi tên `classification` thành `class`, kiểm tra giá trị hợp lệ. Bước này **không** điền khuyết và không scale; các việc đó làm trong từng fold.

## Yêu cầu hệ thống

- Python **3.11** (dùng bản ổn định, không dùng bản beta)
- Các thư viện ghi cố định phiên bản trong `requirements.txt`

## Cài đặt

macOS / Linux:

```bash
git clone https://github.com/hoquocsu1902/DA04_CKD_PROJECT.git
cd DA04_CKD_PROJECT
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/hoquocsu1902/DA04_CKD_PROJECT.git
cd DA04_CKD_PROJECT
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy và kiểm tra

Chạy mọi lệnh từ thư mục gốc của dự án, với `.venv` đang bật.

```bash
python data/nap.py --input data/raw/kidney_disease.csv   # E1 → results/csv/inventory.csv, inventory_summary.json
python -m pytest -q                                      # toàn bộ test phải pass
```

Nếu pytest báo `skipped` thay vì `passed` thì máy bạn chưa có file dữ liệu: các test đó **chưa** được chạy.

Các lệnh cho EDA, thực nghiệm và web sẽ được bổ sung khi từng phần hoàn thành.

## Cấu trúc thư mục

```text
DA04_CKD_PROJECT/
├── data/            # nạp, tiền xử lý, điền khuyết; raw/ chứa dữ liệu gốc (không đưa lên git)
├── models/          # KNN, KNN Imputer, PCA-SVD, principal angles, clustering (tự viết)
├── experiments/     # cấu hình và các thực nghiệm E3–E8
├── analysis/        # EDA, thống kê, vẽ hình
├── tests/           # pytest: dữ liệu, chống rò rỉ, đối chiếu với scikit-learn
├── results/         # csv/, hinh/, models/, npz/ — đầu ra sinh từ code
├── app/             # web FastAPI + HTML/CSS/JS, Plotly lưu cục bộ
├── docs/            # biên bản họp, wireframe, user test, khai báo AI
├── report/          # báo cáo PDF
└── scripts/         # script tiện ích
```

## Nguyên tắc kỹ thuật

- KNN, KNN Imputer, PCA qua SVD và principal angles **tự viết bằng NumPy**. scikit-learn chỉ dùng để đối chiếu và làm baseline.
- **Không rò rỉ dữ liệu:** scaler, imputer và PCA chỉ fit trên phần train của từng fold.
- Chỉ số chính là **F1 của lớp `ckd`**, đánh giá bằng Repeated Stratified 5-fold × 10, hạt giống `HAT_GIONG = 20262027` đặt duy nhất ở `experiments/cau_hinh.py`.
- Mọi so sánh có p-value trên cùng bộ fold: Wilcoxon cho hai phương pháp, Friedman + Nemenyi cho nhiều phương pháp.
- Mọi con số trong báo cáo và trên web đều sinh ra từ code trong repo; không sửa tay file kết quả.

## Quy trình làm việc nhóm

- Không push thẳng lên `main`. Mỗi việc làm trên một nhánh `feat/<ten>-<viec>` rồi mở Pull Request.
- Trước khi mở PR, `python -m pytest -q` phải pass. Mỗi PR có ít nhất một thành viên khác review.
- PR điền theo mẫu `.github/pull_request_template.md`.
- Biên bản họp lưu ở `docs/meeting_minutes/`, đặt tên `YYYY-MM-DD.md`, theo mẫu `_mau_bien_ban.md`.

## Khai báo sử dụng AI

Việc sử dụng công cụ AI trong dự án được ghi lại ở `docs/ai_usage.md`.
