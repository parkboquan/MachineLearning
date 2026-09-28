"""
House Price Prediction - Linear Regression
Version 2

Chạy:
    python house_price_prediction.py
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "house_prices.csv"

FEATURES = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "location_score",
    "age_years",
    "distance_center_km",
]
TARGET = "price_billion_vnd"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("=== HOUSE PRICE PREDICTION - VERSION 2 ===")
    print(f"MAE : {mae:.3f} tỷ VNĐ")
    print(f"RMSE: {rmse:.3f} tỷ VNĐ")
    print(f"R²  : {r2:.3f}")

    print("\nHệ số mô hình:")
    for feature, coef in zip(FEATURES, model.coef_):
        print(f"- {feature}: {coef:.4f}")

    # Dự báo thử cho một căn nhà mới.
    new_house = pd.DataFrame([{
        "area_m2": 100,
        "bedrooms": 3,
        "bathrooms": 2,
        "location_score": 4,
        "age_years": 5,
        "distance_center_km": 5,
    }])

    predicted_price = model.predict(new_house)[0]
    print(
        f"\nDự báo căn nhà mẫu: {predicted_price:.2f} tỷ VNĐ"
    )
    print("\n--- DỰ BÁO GIÁ NHÀ CHO NGƯỜI DÙNG ---")

    area = float(input("Diện tích (m2): "))
    bedrooms = int(input("Số phòng ngủ: "))
    bathrooms = int(input("Số phòng tắm: "))
    location_score = float(input("Điểm vị trí (1-10): "))
    age_years = float(input("Tuổi căn nhà (năm): "))
    distance_center_km = float(input("Khoảng cách đến trung tâm (km): "))

    user_house = pd.DataFrame({
        "area_m2": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "location_score": [location_score],
        "age_years": [age_years],
        "distance_center_km": [distance_center_km]
    })

    predicted_price = model.predict(user_house)[0]

    print(f"\nGiá nhà dự đoán: {predicted_price:.2f} tỷ VNĐ")


if __name__ == "__main__":
    main()
