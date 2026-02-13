"""Crop Yield Prediction System training script."""

import json
import os
from typing import Dict, Tuple

import matplotlib
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


matplotlib.use("Agg")

DATA_PATH = os.path.join("data", "crop_data.csv")
MODEL_PATH = os.path.join("model", "crop_model.pkl")
METRICS_PATH = os.path.join("model", "model_metrics.json")
EDA_DIR = os.path.join("notebooks", "eda")


def generate_synthetic_data(n_rows: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for crop yield prediction."""
    rng = np.random.default_rng(random_state)

    states = [
        "Andhra Pradesh",
        "Bihar",
        "Gujarat",
        "Karnataka",
        "Maharashtra",
        "Punjab",
        "Rajasthan",
        "Tamil Nadu",
        "Uttar Pradesh",
        "West Bengal",
    ]
    crops = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Soybean", "Pulses"]
    soils = ["Loamy", "Clay", "Sandy", "Silt", "Peaty"]

    state = rng.choice(states, size=n_rows)
    crop = rng.choice(crops, size=n_rows)
    soil = rng.choice(soils, size=n_rows)

    rainfall = rng.normal(loc=900, scale=250, size=n_rows).clip(200, 2000)
    temperature = rng.normal(loc=26, scale=5, size=n_rows).clip(10, 40)
    fertilizer = rng.normal(loc=120, scale=40, size=n_rows).clip(20, 300)
    area = rng.normal(loc=4.5, scale=2.0, size=n_rows).clip(0.5, 20)

    crop_base = {
        "Wheat": 3.2,
        "Rice": 4.1,
        "Maize": 3.6,
        "Cotton": 2.4,
        "Sugarcane": 5.0,
        "Soybean": 2.8,
        "Pulses": 2.2,
    }
    soil_bonus = {"Loamy": 0.5, "Clay": 0.2, "Sandy": -0.3, "Silt": 0.1, "Peaty": 0.4}

    base_yield = np.array([crop_base[c] for c in crop])
    soil_effect = np.array([soil_bonus[s] for s in soil])

    rainfall_effect = (rainfall - 800) / 1200
    temp_effect = -((temperature - 25) ** 2) / 100
    fert_effect = np.log1p(fertilizer) / 5
    area_effect = np.sqrt(area) / 3

    noise = rng.normal(0, 0.35, size=n_rows)

    yield_tph = (
        base_yield
        + soil_effect
        + rainfall_effect
        + temp_effect
        + fert_effect
        + area_effect
        + noise
    )
    yield_tph = np.clip(yield_tph, 0.5, None)

    df = pd.DataFrame(
        {
            "State": state,
            "Crop Type": crop,
            "Rainfall": rainfall.round(1),
            "Temperature": temperature.round(1),
            "Soil Type": soil,
            "Fertilizer Usage": fertilizer.round(1),
            "Area": area.round(2),
            "Yield": yield_tph.round(2),
        }
    )

    for col in ["Rainfall", "Temperature", "Fertilizer Usage", "Soil Type"]:
        missing_mask = rng.random(n_rows) < 0.03
        df.loc[missing_mask, col] = np.nan

    return df


def save_dataset(df: pd.DataFrame) -> None:
    """Save dataset to disk."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


def perform_eda(df: pd.DataFrame) -> None:
    """Generate EDA plots and save them to disk."""
    os.makedirs(EDA_DIR, exist_ok=True)

    numeric_cols = ["Rainfall", "Temperature", "Fertilizer Usage", "Area", "Yield"]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"))
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="Rainfall", y="Yield", alpha=0.6)
    plt.title("Rainfall vs Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "rainfall_vs_yield.png"))
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="Temperature", y="Yield", alpha=0.6, color="tomato")
    plt.title("Temperature vs Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "temperature_vs_yield.png"))
    plt.close()


def build_preprocessor(
    categorical_features: list, numeric_features: list
) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )

    return preprocessor


def evaluate_model(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def select_best_model(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """Select best model based on highest R2 and lowest RMSE."""
    best_name = None
    best_score = (-np.inf, np.inf)
    for name, score in metrics.items():
        r2 = score["r2"]
        rmse = score["rmse"]
        candidate = (r2, -rmse)
        if candidate > best_score:
            best_score = candidate
            best_name = name
    return best_name, metrics[best_name]


def main() -> None:
    """End-to-end training pipeline."""
    df = generate_synthetic_data(n_rows=2500)
    save_dataset(df)
    perform_eda(df)

    x = df.drop(columns=["Yield"])
    y = df["Yield"]

    categorical_features = ["State", "Crop Type", "Soil Type"]
    numeric_features = ["Rainfall", "Temperature", "Fertilizer Usage", "Area"]

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=350,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    metrics: Dict[str, Dict[str, float]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        metrics[name] = evaluate_model(pipeline, x_test, y_test)
        trained_pipelines[name] = pipeline
        print(f"{name} -> R2: {metrics[name]['r2']:.3f}, MAE: {metrics[name]['mae']:.3f}, RMSE: {metrics[name]['rmse']:.3f}")

    best_model_name, best_metrics = select_best_model(metrics)
    best_pipeline = trained_pipelines[best_model_name]

    best_pipeline.fit(x, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_model_name,
                "metrics": metrics,
                "selected_metrics": best_metrics,
            },
            file,
            indent=2,
        )

    print(f"Best model saved: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
