# Bài thực hành: Overfitting trong dự báo giá nhà

## 1. Mục tiêu

Sử dụng lại bộ dữ liệu `../house/house_prices.csv` của bài dự báo giá nhà để:

1. Tạo một mô hình bị **overfitting**.
2. Đo kết quả trên tập train và tập test.
3. Áp dụng 3 cách giảm overfitting.

## 2. Chạy chương trình

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy từ thư mục gốc của repository:

```bash
python overfitting/overfitting_house_price.py
```

## 3. Cách tạo overfitting

Chương trình dùng:

```python
DecisionTreeRegressor(random_state=42)
```

Không giới hạn `max_depth` và không đặt `min_samples_leaf`, vì vậy cây quyết định có thể phát triển rất sâu và ghi nhớ dữ liệu train.

Dấu hiệu overfitting:

- `Train R²` gần 1.000.
- `Test R²` thấp hơn đáng kể.
- Khoảng cách giữa Train R² và Test R² lớn.

## 4. Ba cách khắc phục

### Cách 1: Giới hạn độ sâu cây

```python
DecisionTreeRegressor(max_depth=4, random_state=42)
```

Cây đơn giản hơn, ít học thuộc dữ liệu train hơn.

### Cách 2: Tăng số mẫu tối thiểu ở mỗi lá

```python
DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
```

Mỗi lá phải có ít nhất 5 mẫu, giúp dự đoán ổn định hơn.

### Cách 3: Sử dụng Random Forest

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
)
```

Random Forest kết hợp nhiều cây để giảm phương sai và thường tổng quát hóa tốt hơn một cây đơn lẻ.

## 5. Ý nghĩa các chỉ số

- **Train R²:** mức độ phù hợp trên dữ liệu huấn luyện.
- **Test R²:** khả năng dự đoán trên dữ liệu chưa từng thấy.
- **MAE:** sai số tuyệt đối trung bình, đơn vị tỷ VNĐ.
- **RMSE:** phạt các sai số lớn mạnh hơn MAE.

Không nên chỉ nhìn Train R². Khi phát hiện overfitting, cần ưu tiên kết quả trên Test R² và khoảng cách Train/Test.
