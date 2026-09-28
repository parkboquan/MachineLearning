"""Demonstrate overfitting and three ways to reduce it.

Uses the house_prices.csv dataset in the sibling ``house`` exercise.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_FILE = Path(__file__).resolve().parent.parent / "house" / "house_prices.csv"
FEATURES = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "location_score",
    "age_years",
    "distance_center_km",
]
TARGET = "price_billion_vnd"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Train R² : {train_r2:.3f}")
    print(f"Test  R² : {test_r2:.3f}")
    print(f"Test MAE : {test_mae:.3f} tỷ VNĐ")
    print(f"Test RMSE: {test_rmse:.3f} tỷ VNĐ")
    print(f"Khoảng cách Train/Test R²: {train_r2 - test_r2:.3f}")
    return model, train_r2, test_r2


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=== OVERFITTING HOUSE PRICE PREDICTION ===")
    print("Sử dụng lại bộ dữ liệu house_prices.csv\n")

    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Số dòng dữ liệu: {len(df)}")
    print(f"Dữ liệu train: {len(X_train)} dòng")
    print(f"Dữ liệu test : {len(X_test)} dòng")

    # 1. Deliberately overfit: an unrestricted tree can memorize training data.
    evaluate_model(
        "1. MÔ HÌNH CỐ TÌNH OVERFITTING - Decision Tree không giới hạn",
        DecisionTreeRegressor(random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Fix 1: limit tree complexity.
    evaluate_model(
        "2. FIX 1 - Giới hạn độ sâu cây (max_depth=4)",
        DecisionTreeRegressor(max_depth=4, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Fix 2: require more samples in each leaf.
    evaluate_model(
        "3. FIX 2 - Tăng số mẫu tối thiểu ở lá (min_samples_leaf=5)",
        DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Fix 3: average many less-correlated trees.
    evaluate_model(
        "4. FIX 3 - Dùng Random Forest để giảm phương sai",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
        ),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nKẾT LUẬN")
    print("- Overfitting thường có Train R² rất cao nhưng Test R² thấp hơn nhiều.")
    print("- Có thể giảm overfitting bằng cách giảm độ phức tạp mô hình,")
    print("  tăng số mẫu mỗi lá, hoặc dùng Random Forest với tham số phù hợp.")


if __name__ == "__main__":
    main()
