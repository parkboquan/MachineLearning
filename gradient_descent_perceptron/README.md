# Machine Learning — Bài tập 3.26 đến 3.30

## Bài 3.26 — Gradient Descent

Với `f(x) = x² - 4x + 5 = (x - 2)² + 1`:

`f'(x) = 2x - 4`

Công thức cập nhật với `η = 0.2` là:

`x(k+1) = x(k) - 0.2(2x(k) - 4)`

| Bước k | x(k) | f'(x(k)) | f(x(k)) |
|---:|---:|---:|---:|
| 0 | 5.000000 | 6.000000 | 10.000000 |
| 1 | 3.800000 | 3.600000 | 4.240000 |
| 2 | 3.080000 | 2.160000 | 2.166400 |
| 3 | 2.648000 | 1.296000 | 1.419904 |
| 4 | 2.388800 | 0.777600 | 1.151165 |

Sau mỗi bước, `x(k) - 2 = 0.6(x(k-1) - 2)`, nên sai số giảm theo cấp số
nhân với công bội `0.6`. Vì `|0.6| < 1`, thuật toán hội tụ tuyến tính về
điểm cực tiểu `x* = 2`, tại đó `f(x*) = 1`.

## Bài 3.27 — Dự đoán Perceptron

Với `w = [1, 2, -10]ᵀ`, `x = [3, 4, 1]ᵀ`:

`wᵀx = 1·3 + 2·4 - 10·1 = 1`

Theo quy ước `wᵀx >= 0` dự đoán nhãn `+1`, nhãn dự đoán là `+1`. Nhãn thật
là `-1`, vì vậy điểm dữ liệu **bị phân lớp sai**.

## Bài 3.28 — Cập nhật Perceptron

Với `w = [-2, 1, 0]ᵀ`, `x = [2, 3, 1]ᵀ`, `y = +1`:

`wᵀx = -2·2 + 1·3 + 0·1 = -1`

Mô hình dự đoán `-1`, khác nhãn thật `+1`, nên mẫu bị phân lớp sai. Chọn
learning rate Perceptron mặc định `η = 1`:

`w_new = w + ηyx = [-2, 1, 0]ᵀ + [2, 3, 1]ᵀ = [0, 4, 1]ᵀ`

Sau cập nhật: `w_newᵀx = 0·2 + 4·3 + 1·1 = 13`, do đó dự đoán mới là `+1`.

## Bài 3.29 — Lớp Perceptron

Lớp `Perceptron` tự cài đặt có:

- `fit(X, y)` để huấn luyện với nhãn `-1`, `+1`;
- `predict(X)` để dự báo;
- `decision_function(X)` để trả về điểm tuyến tính;
- dừng sớm khi một epoch không còn mẫu sai.

Phần demo dùng dữ liệu cổng AND để kiểm tra mô hình học được một bài toán phân
lớp tuyến tính đơn giản.

## Bài 3.30 — Phân lớp Breast Cancer Wisconsin

Code sử dụng tập Breast Cancer Wisconsin có sẵn trong scikit-learn (569 mẫu,
30 đặc trưng, hai lớp là u ác tính/lành tính). Quy trình gồm:

1. Chia train/test theo tỷ lệ 80/20 có phân tầng và `random_state=42`.
2. Chuẩn hóa đặc trưng bằng `StandardScaler` chỉ trên từng fold train để tránh
   rò rỉ dữ liệu.
3. Dùng Grid Search 5-fold trên tập train để chọn `penalty`, `alpha`, `eta0` và
   `class_weight`, tối ưu theo F1-score.
4. Báo cáo Accuracy, Precision, Recall và F1-score có trọng số trên tập test.

Kết quả tái lập trong môi trường kiểm thử (`random_state=42`):

| Độ đo trên tập test | Kết quả |
|---|---:|
| Accuracy | 0.9561 |
| Precision (weighted) | 0.9581 |
| Recall (weighted) | 0.9561 |
| F1-score (weighted) | 0.9564 |

Grid Search chọn `penalty=l1`, `alpha=1e-5`, `eta0=0.1`, không dùng
`class_weight`; F1 trung bình của 5 fold là `0.9773`. Ma trận nhầm lẫn trên
tập test là `[[41, 1], [4, 68]]`. Các con số có thể chênh lệch rất nhỏ giữa
các phiên bản thư viện; chương trình luôn in lại kết quả thực tế khi chạy.

## Cách chạy

Từ thư mục `gradient_descent_perceptron`:

```bash
python -m pip install -r requirements.txt
python -m chapter_03.bai_3_26_3_30
python -m unittest discover -s tests -v
```

Mã nguồn chính: `chapter_03/bai_3_26_3_30.py`.
