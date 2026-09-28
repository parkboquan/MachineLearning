# Machine Learning Projects

Repository tổng hợp ba bài thực hành Machine Learning:

| Thư mục | Nội dung | Thuật toán chính |
|---|---|---|
| [`house/`](house/) | Dự đoán giá nhà | Linear Regression |
| [`overfitting/`](overfitting/) | Minh họa overfitting và các cách khắc phục | Decision Tree, Random Forest |
| [`gradient_descent_perceptron/`](gradient_descent_perceptron/) | Bài tập 3.26–3.30 | Gradient Descent, Perceptron |

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

## Chạy từng bài

```bash
# Bài dự đoán giá nhà (có phần nhập dữ liệu tương tác)
python house/house_price_prediction.py

# Bài overfitting
python overfitting/overfitting_house_price.py

# Bài Gradient Descent và Perceptron
cd gradient_descent_perceptron
python -m chapter_03.bai_3_26_3_30
python -m unittest discover -s tests -v
```

Dữ liệu `house/house_prices.csv` là dữ liệu mẫu phục vụ học tập và được dùng
chung cho hai bài `house` và `overfitting`.
