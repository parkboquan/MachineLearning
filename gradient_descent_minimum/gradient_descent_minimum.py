"""Hai bài tìm cực tiểu bằng phương pháp Gradient Descent."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GDResult:
    x: float
    value: float
    gradient: float
    iterations: int
    converged: bool
    history: tuple[float, ...]


def gradient_descent(
    objective: Callable[[float], float],
    derivative: Callable[[float], float],
    x0: float,
    learning_rate: float = 0.1,
    tolerance: float = 1e-3,
    max_iterations: int = 1000,
) -> GDResult:
    """Lặp x(k+1) = x(k) - eta*f'(x(k)) đến khi đạo hàm đủ nhỏ."""
    if learning_rate <= 0:
        raise ValueError("learning_rate phải lớn hơn 0")
    if tolerance <= 0:
        raise ValueError("tolerance phải lớn hơn 0")
    if max_iterations <= 0:
        raise ValueError("max_iterations phải lớn hơn 0")

    x = float(x0)
    history = [x]
    for iteration in range(1, max_iterations + 1):
        x -= learning_rate * derivative(x)
        history.append(x)
        if abs(derivative(x)) < tolerance:
            return GDResult(
                x=x,
                value=objective(x),
                gradient=derivative(x),
                iterations=iteration,
                converged=True,
                history=tuple(history),
            )

    return GDResult(
        x=x,
        value=objective(x),
        gradient=derivative(x),
        iterations=max_iterations,
        converged=False,
        history=tuple(history),
    )


def f1(x: float) -> float:
    return x**2 - 2


def df1(x: float) -> float:
    return 2 * x


def f2(x: float) -> float:
    return x**3 / 3 - x


def df2(x: float) -> float:
    return x**2 - 1


def print_result(title: str, result: GDResult) -> None:
    print(title)
    print(f"  Hội tụ       : {'Có' if result.converged else 'Không'}")
    print(f"  Số lần cập nhật: {result.iterations}")
    print(f"  x            : {result.x:.8f}")
    print(f"  f(x)         : {result.value:.8f}")
    print(f"  f'(x)        : {result.gradient:.8f}\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("HAI BÀI TÌM CỰC TIỂU BẰNG GRADIENT DESCENT")
    print("Learning rate η = 0.1; điều kiện dừng |f'(x)| < 10^-3\n")

    result1 = gradient_descent(f1, df1, x0=5, learning_rate=0.1)
    print_result("Bài 1: f(x) = x² - 2, x(0) = 5", result1)

    # Chạy từ hai điểm khởi tạo dương thường gặp để kiểm tra cùng hội tụ về x=1.
    result2_from_2 = gradient_descent(f2, df2, x0=2, learning_rate=0.1)
    print_result("Bài 2: f(x) = x³/3 - x, x(0) = 2", result2_from_2)

    result2_from_5 = gradient_descent(f2, df2, x0=5, learning_rate=0.1)
    print_result("Kiểm tra thêm Bài 2 với x(0) = 5", result2_from_5)


if __name__ == "__main__":
    main()
