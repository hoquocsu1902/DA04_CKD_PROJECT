# DA-04 CKD — Nhóm 02, lớp 15DHTH09 (HUIT)

Ba cách xử lý dữ liệu khuyết tạo ra các hệ trục PCA lệch nhau bao nhiêu độ,
và độ lệch ấy có làm thay đổi kết quả phân lớp CKD không?

## Cài đặt (Python 3.11)

    git clone https://github.com/hoquocsu1902/DA04_CKD_PROJECT.git
    cd DA04_CKD_PROJECT
    python3.11 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Kiểm tra

    python data/nap.py --input data/raw/kidney_disease.csv
    python -m pytest -q                 # phải ra: 4 passed

## Dữ liệu

data/raw/kidney_disease.csv — Kaggle: akshayksingh/kidney-disease-dataset
(bản gốc UCI: Chronic Kidney Disease, id 336). 400 bệnh nhân, 24 feature.
