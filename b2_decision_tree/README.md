# Bài 2 — Cây quyết định ID3 và CART

## 1. Dữ liệu

Bộ dữ liệu có 14 mẫu, 4 thuộc tính đầu vào (`age`, `income`, `student`,
`credit_rating`) và nhãn `buys_computer`. Có 9 mẫu `yes` và 5 mẫu `no`.

## 2. ID3 — Entropy và Information Gain

Công thức:

```text
Entropy(S) = -Σ pᵢ log₂(pᵢ)
Gain(S, A) = Entropy(S) - Σ(|Sᵥ|/|S|)Entropy(Sᵥ)
```

Entropy của toàn bộ dữ liệu:

```text
Entropy(S) = -(9/14)log₂(9/14) - (5/14)log₂(5/14) = 0.9403
```

Information Gain tại nút gốc:

| Thuộc tính | Information Gain |
|---|---:|
| age | **0.2467** |
| student | 0.1518 |
| credit_rating | 0.0481 |
| income | 0.0292 |

ID3 chọn `age` vì có Information Gain lớn nhất. Cây thu được:

```text
age?
├── <=30: student?
│   ├── no  → no
│   └── yes → yes
├── 31..40 → yes
└── >40: credit_rating?
    ├── fair      → yes
    └── excellent → no
```

## 3. CART — Gini Index

CART chỉ chia hai nhánh tại mỗi nút. Với thuộc tính phân loại, ta xét các cách
chia tập giá trị thành hai nhóm không rỗng.

```text
Gini(S) = 1 - Σpᵢ²
GiniSplit = Σ(|Sⱼ|/|S|)Gini(Sⱼ)
```

`Gini(S) = 1 - (9/14)² - (5/14)² = 0.4592`.

Cách chia tốt nhất của từng thuộc tính tại nút gốc:

| Thuộc tính | Nhóm tách nhị phân | GiniSplit |
|---|---|---:|
| age | `{31..40}` / `{<=30, >40}` | **0.3571** |
| income | `{high}` / `{low, medium}` | 0.4429 |
| student | `{no}` / `{yes}` | 0.3673 |
| credit_rating | `{excellent}` / `{fair}` | 0.4286 |

CART chọn `age ∈ {31..40}` vì GiniSplit nhỏ nhất. Cây đầy đủ:

```text
age ∈ {31..40}?
├── Đúng → yes
└── Sai: student ∈ {no}?
    ├── Đúng: age ∈ {<=30}?
    │   ├── Đúng → no
    │   └── Sai: credit_rating ∈ {excellent}?
    │       ├── Đúng → no
    │       └── Sai  → yes
    └── Sai: credit_rating ∈ {excellent}?
        ├── Đúng: age ∈ {<=30}?
        │   ├── Đúng → yes
        │   └── Sai  → no
        └── Sai → yes
```

Cả hai cây phân loại đúng 14/14 mẫu huấn luyện. Đây là kết quả trên tập huấn
luyện nhỏ, không phải bằng chứng rằng mô hình sẽ tổng quát hóa tốt trên dữ liệu
mới.

Tham khảo cách CART đánh giá phép chia bằng Gini Index:
[Big Data Uni — CART (Gini Index)](https://bigdatauni.com/tin-tuc/thuat-toan-cay-quyet-dinh-p-2-cart-gini-index.html).
