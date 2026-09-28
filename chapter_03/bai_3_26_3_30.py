"""Lời giải bằng code cho các bài tập 3.26 đến 3.30.

Chạy toàn bộ bài:
    python -m chapter_03.bai_3_26_3_30
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def objective(x: float) -> float:
    """f(x) = x^2 - 4x + 5 = (x - 2)^2 + 1."""
    return x**2 - 4 * x + 5


def derivative(x: float) -> float:
    """Đạo hàm f'(x) = 2x - 4."""
    return 2 * x - 4


@dataclass(frozen=True)
class GradientDescentStep:
    step: int
    x: float
    gradient: float
    value: float


def gradient_descent(
    x0: float = 5.0, learning_rate: float = 0.2, n_steps: int = 4
) -> list[GradientDescentStep]:
    """Thực hiện Gradient Descent và giữ cả trạng thái khởi tạo."""
    if learning_rate <= 0:
        raise ValueError("learning_rate phải lớn hơn 0")
    if n_steps < 0:
        raise ValueError("n_steps không được âm")

    x = float(x0)
    history = [GradientDescentStep(0, x, derivative(x), objective(x))]
    for step in range(1, n_steps + 1):
        x -= learning_rate * derivative(x)
        history.append(GradientDescentStep(step, x, derivative(x), objective(x)))
    return history


def perceptron_score(weights: ArrayLike, sample: ArrayLike) -> float:
    """Tính tích vô hướng w^T x."""
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(sample, dtype=float)))


def sign_label(score: float) -> int:
    """Quy ước Perceptron: score >= 0 thuộc lớp +1, ngược lại là -1."""
    return 1 if score >= 0 else -1


def perceptron_update(
    weights: ArrayLike, sample: ArrayLike, label: int, learning_rate: float = 1.0
) -> NDArray[np.float64]:
    """Cập nhật w <- w + eta*y*x cho một mẫu bị phân lớp sai."""
    if label not in (-1, 1):
        raise ValueError("label phải là -1 hoặc +1")
    return np.asarray(weights, dtype=float) + learning_rate * label * np.asarray(
        sample, dtype=float
    )


class Perceptron:
    """Perceptron nhị phân cài đặt từ đầu, nhận nhãn -1 và +1.

    Dữ liệu truyền vào ``fit`` không cần thêm bias; bias được lưu riêng trong
    ``bias_``. Mẫu được cập nhật khi y * (w^T x + b) <= 0.
    """

    def __init__(
        self, learning_rate: float = 0.01, n_epochs: int = 100, shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate phải lớn hơn 0")
        if n_epochs <= 0:
            raise ValueError("n_epochs phải lớn hơn 0")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError("X phải là ma trận hai chiều")
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y phải là vector và có cùng số mẫu với X")
        if not np.all(np.isin(y_array, (-1, 1))):
            raise ValueError("Perceptron này chỉ nhận nhãn -1 và +1")

        y_array = y_array.astype(int)
        self.weights_ = np.zeros(X_array.shape[1], dtype=float)
        self.bias_ = 0.0
        self.errors_: list[int] = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_epochs):
            indices = rng.permutation(len(X_array)) if self.shuffle else np.arange(len(X_array))
            errors = 0
            for index in indices:
                score = float(X_array[index] @ self.weights_ + self.bias_)
                if y_array[index] * score <= 0:
                    update = self.learning_rate * y_array[index]
                    self.weights_ += update * X_array[index]
                    self.bias_ += update
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_is_fitted()
        X_array = np.asarray(X, dtype=float)
        return X_array @ self.weights_ + self.bias_

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int64)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Mô hình chưa được huấn luyện; hãy gọi fit trước")


def solve_3_26() -> None:
    print("\nBài 3.26 - Gradient Descent")
    print("f'(x) = 2x - 4")
    print(f"{'Bước':>5} {'x':>12} {'f\'(x)':>12} {'f(x)':>12}")
    for item in gradient_descent():
        print(f"{item.step:>5} {item.x:>12.6f} {item.gradient:>12.6f} {item.value:>12.6f}")
    print("x tiến dần đến 2 và f(x) tiến dần đến giá trị cực tiểu 1.")


def solve_3_27() -> None:
    weights = np.array([1, 2, -10])
    sample = np.array([3, 4, 1])
    actual_label = -1
    score = perceptron_score(weights, sample)
    prediction = sign_label(score)
    print("\nBài 3.27 - Perceptron")
    print(f"w^T x = {score:g}; nhãn dự đoán = {prediction:+d}")
    print(f"Nhãn thật = {actual_label:+d}; phân lớp sai: {prediction != actual_label}")


def solve_3_28() -> None:
    weights = np.array([-2, 1, 0])
    sample = np.array([2, 3, 1])
    actual_label = 1
    old_score = perceptron_score(weights, sample)
    prediction = sign_label(old_score)
    is_wrong = prediction != actual_label
    new_weights = perceptron_update(weights, sample, actual_label) if is_wrong else weights
    new_score = perceptron_score(new_weights, sample)
    print("\nBài 3.28 - Một bước cập nhật Perceptron (eta = 1)")
    print(f"Trước cập nhật: w^T x = {old_score:g}, dự đoán = {prediction:+d}, sai = {is_wrong}")
    print(f"w mới = {new_weights}; w_mới^T x = {new_score:g}")


def solve_3_29_demo() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, -1, -1, 1])  # Cổng AND
    model = Perceptron(learning_rate=0.1, n_epochs=100, random_state=42).fit(X, y)
    print("\nBài 3.29 - Lớp Perceptron tự cài đặt")
    print(f"Dự đoán cổng AND: {model.predict(X).tolist()}")
    print(f"Số epoch đã chạy: {len(model.errors_)}")


def solve_3_30() -> dict[str, float]:
    """Huấn luyện và đánh giá Perceptron trên Breast Cancer Wisconsin."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SklearnPerceptron(random_state=42, max_iter=5000, tol=1e-4)),
        ]
    )
    param_grid = {
        "model__penalty": [None, "l1", "l2", "elasticnet"],
        "model__alpha": [1e-5, 1e-4, 1e-3],
        "model__eta0": [0.001, 0.01, 0.1, 1.0],
        "model__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }
    print("\nBài 3.30 - Breast Cancer Wisconsin")
    print(f"Siêu tham số tốt nhất theo CV F1: {search.best_params_}")
    print(f"F1 CV tốt nhất: {search.best_score_:.4f}")
    print("Kết quả trên tập kiểm tra:")
    for name, value in metrics.items():
        print(f"  {name.capitalize():<10}: {value:.4f}")
    print(f"Ma trận nhầm lẫn:\n{confusion_matrix(y_test, predictions)}")
    return metrics


def main() -> None:
    # Windows có thể dùng bảng mã cp1252 mặc định, không in được tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    solve_3_26()
    solve_3_27()
    solve_3_28()
    solve_3_29_demo()
    solve_3_30()


if __name__ == "__main__":
    main()

