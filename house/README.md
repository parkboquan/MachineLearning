# House Price Prediction - Linear Regression

Dự án dự báo giá nhà bằng **hồi quy tuyến tính (Linear Regression)** với Python và scikit-learn.

## Cấu trúc

```text
house/
├── house_prices.csv
├── house_price_prediction.py
└── README.md
```

## Dữ liệu

File `house_prices.csv` gồm 100 mẫu dữ liệu, với các biến:

- `area_m2`: diện tích nhà (m²)
- `bedrooms`: số phòng ngủ
- `bathrooms`: số phòng tắm
- `location_score`: điểm vị trí từ 1 đến 5
- `age_years`: tuổi căn nhà (năm)
- `distance_center_km`: khoảng cách tới trung tâm (km)
- `price_billion_vnd`: giá nhà, đơn vị tỷ VNĐ

> Đây là **dữ liệu mẫu phục vụ học tập**, không phải dữ liệu giao dịch bất động sản thực tế.

## Cài đặt

Yêu cầu Python 3.9+.

### 1. Tạo môi trường ảo

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài thư viện

```bash
pip install pandas scikit-learn
```

## Chạy chương trình

```bash
python house/house_price_prediction.py
```

Chương trình sẽ:

1. Đọc dữ liệu từ `house_prices.csv`.
2. Chia dữ liệu thành tập train/test.
3. Huấn luyện mô hình Linear Regression.
4. Tính MAE, RMSE và R².
5. In hệ số của mô hình.
6. Dự báo giá cho một căn nhà mẫu.

## Git workflow

Khởi tạo repository:

```bash
git init
git branch -M main
git add .
git commit -m "Initial version: house price prediction"
```

Kết nối GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/house-price-prediction.git
git push -u origin main
```

Tạo nhánh cho Version 2:

```bash
git checkout -b version-2-house-price
git push -u origin version-2-house-price
```

Sau khi hoàn thành Version 2:

```bash
git add .
git commit -m "Version 2: improve house price prediction"
git push
```

Sau đó có thể tạo Pull Request từ `version-2-house-price` vào `main` trên GitHub.

## Lưu ý

Nếu repository trên GitHub đã có README được tạo sẵn, khi đẩy một repository local có lịch sử Git riêng có thể phát sinh merge conflict. Cách đơn giản là tạo repository GitHub **trống**, không chọn khởi tạo README/.gitignore/License, rồi push repository local lên.
## Version 3

- Cập nhật tài liệu dự án.
- Tiếp tục sử dụng mô hình Linear Regression để dự đoán giá nhà.
