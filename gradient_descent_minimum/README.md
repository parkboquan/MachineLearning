# Hai bài tìm cực tiểu bằng Gradient Descent

Công thức cập nhật chung:

```text
x(k+1) = x(k) - ηf'(x(k))
```

Trong bài dùng `η = 0.1` và dừng khi `|f'(x)| < 10⁻³`.

## Bài 1

```text
f(x)  = x² - 2
f'(x) = 2x
x(0)  = 5
```

Công thức cập nhật:

```text
x(k+1) = x(k) - 0.1·2x(k) = 0.8x(k)
```

Do đó `x(k) = 5·0.8ᵏ`. Vì `|0.8| < 1`, dãy hội tụ về `x* = 0`.

```text
x*       ≈ 0.00042535
f(x*)    ≈ -1.99999982
số bước  = 42
```

Về lý thuyết, `f''(x) = 2 > 0`, nên `x = 0` là điểm cực tiểu toàn cục và
`f_min = -2`.

## Bài 2

```text
f(x)  = x³/3 - x
f'(x) = x² - 1
η     = 0.1
```

Công thức cập nhật:

```text
x(k+1) = x(k) - 0.1(x(k)² - 1)
```

Giải `f'(x) = 0` được hai điểm dừng `x = -1` và `x = 1`. Vì:

```text
f''(x) = 2x
f''(-1) = -2 < 0  → cực đại địa phương
f''(1)  =  2 > 0  → cực tiểu địa phương
```

Chạy Gradient Descent từ điểm khởi tạo dương:

| Điểm đầu | Nghiệm gần đúng | Giá trị hàm | Số bước |
|---:|---:|---:|---:|
| `x(0)=2` | `1.000476` | `-0.666666` | 32 |
| `x(0)=5` | `1.000482` | `-0.666666` | 34 |

Lưu ý: đây là cực tiểu **địa phương**. Hàm bậc ba không có cực tiểu toàn cục
vì `f(x) → -∞` khi `x → -∞`.

## Chạy chương trình

```bash
cd gradient_descent_minimum
python gradient_descent_minimum.py
python -m unittest -v
```
