"""
Train and evaluate ML models for student performance prediction.

This script:
1. Loads sample data from data/student_performance.csv
2. Fills missing values with column medians
3. Normalizes numeric features for Linear Regression
4. Trains Linear Regression and Random Forest
5. Compares metrics and checks simple over/under-fitting signal
6. Saves best model and metadata for Flask deployment
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "student_score_model.joblib"
META_PATH = MODEL_DIR / "model_info.json"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    numeric_cols = ["StudyHours", "SleepHours", "PracticeTests", "FinalScore"]
    # Convert all model columns to numeric and keep non-numeric as NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing numeric values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    return {
        "name": name,
        "train_r2": round(float(train_r2), 4),
        "test_r2": round(float(test_r2), 4),
        "test_mae": round(float(test_mae), 4),
        "overfit_gap": round(float(train_r2 - test_r2), 4),
        "y_true": y_test.tolist(),
        "y_pred": [float(v) for v in test_pred],
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    x = df[["StudyHours", "SleepHours", "PracticeTests"]]
    y = df["FinalScore"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # StandardScaler handles feature normalization for Linear Regression.
    linear_model = Pipeline(
        steps=[("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )
    forest_model = RandomForestRegressor(
        n_estimators=220, max_depth=8, min_samples_leaf=2, random_state=42
    )

    linear_metrics = evaluate_model(
        "Linear Regression", linear_model, x_train, x_test, y_train, y_test
    )
    forest_metrics = evaluate_model(
        "Random Forest", forest_model, x_train, x_test, y_train, y_test
    )

    # Choose best model by test R^2.
    best_model = forest_model if forest_metrics["test_r2"] >= linear_metrics["test_r2"] else linear_model
    best_name = "Random Forest" if best_model is forest_model else "Linear Regression"
    best_model.fit(x_train, y_train)
    joblib.dump(best_model, MODEL_PATH)

    # Feature importance can come from the random forest directly.
    forest_model.fit(x_train, y_train)
    feature_importance = dict(
        zip(
            ["StudyHours", "SleepHours", "PracticeTests"],
            [round(float(v), 4) for v in forest_model.feature_importances_],
        )
    )

    model_info = {
        "dataset_size": int(len(df)),
        "features": ["StudyHours", "SleepHours", "PracticeTests"],
        "target": "FinalScore",
        "models": [linear_metrics, forest_metrics],
        "best_model": best_name,
        "feature_importance": feature_importance,
        "methodology": [
            "Median imputation for missing numeric values",
            "Feature normalization with StandardScaler for Linear Regression",
            "Train/test split: 80/20",
            "Metrics: R^2 and MAE",
        ],
        "overfitting_note": (
            "Model may be overfitting if train R^2 is much larger than test R^2 "
            "(large overfit_gap)."
        ),
    }

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
