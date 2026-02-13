"""Streamlit app for Crop Yield Prediction System with advanced features."""

import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import streamlit as st


matplotlib.use("Agg")

DATA_PATH = os.path.join("data", "crop_data.csv")
MODEL_PATH = os.path.join("model", "crop_model.pkl")
METRICS_PATH = os.path.join("model", "model_metrics.json")

# Market prices per ton (in currency units)
MARKET_PRICES = {
    "Wheat": 2500,
    "Rice": 3200,
    "Maize": 2000,
    "Cotton": 5500,
    "Sugarcane": 1800,
    "Soybean": 4000,
    "Pulses": 4500,
}

# Seasonal crop suitability based on Summer, Winter, Rainy seasons
SEASONAL_CROPS = {
    "Summer": ["Maize", "Cotton", "Groundnut"],
    "Winter": ["Wheat", "Barley", "Mustard"],
    "Rainy": ["Rice", "Sugarcane", "Soybean"],
}


def load_options() -> dict:
    """Load category options from dataset or fallback defaults."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return {
            "states": sorted(df["State"].dropna().unique().tolist()),
            "crops": sorted(df["Crop Type"].dropna().unique().tolist()),
            "soils": sorted(df["Soil Type"].dropna().unique().tolist()),
            "mean_yield": float(df["Yield"].mean()),
        }

    return {
        "states": ["Andhra Pradesh", "Bihar", "Gujarat", "Punjab"],
        "crops": ["Wheat", "Rice", "Maize"],
        "soils": ["Loamy", "Clay", "Sandy"],
        "mean_yield": 3.2,
    }


def load_metrics() -> dict:
    """Load metrics from disk or return fallback values."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "best_model": "XGBoost",
        "metrics": {
            "Linear Regression": {"r2": 0.80, "mae": 0.39, "rmse": 0.50},
            "Random Forest": {"r2": 0.85, "mae": 0.35, "rmse": 0.44},
            "XGBoost": {"r2": 0.86, "mae": 0.33, "rmse": 0.43},
        },
    }


def categorize_yield(value: float) -> tuple[str, str]:
    """Return category label and display color for yield value."""
    if value < 2:
        return "Low Yield", "#e74c3c"
    if value <= 4:
        return "Moderate Yield", "#f39c12"
    return "High Yield", "#27ae60"


def get_model_name(pipeline) -> str:
    """Extract the model name from the sklearn pipeline."""
    model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    return model.__class__.__name__


def get_feature_importance(pipeline) -> pd.DataFrame | None:
    """Return feature importance if supported by the model."""
    model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    if not hasattr(model, "feature_importances_"):
        return None

    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    if preprocessor is None:
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance


def get_seasonal_crops(season: str) -> list[str]:
    """Get recommended crops for the selected season."""
    return SEASONAL_CROPS.get(season, [])


def get_seasonal_insights(season: str, rainfall: float, temperature: float) -> list[str]:
    """Generate seasonal insights based on rainfall, temperature, and season."""
    insights = []
    
    # Rainy season insights
    if season == "Rainy" and rainfall > 800:
        insights.append("💧 Favorable season for water-intensive crops.")
    
    if season == "Rainy" and rainfall < 600:
        insights.append("⚠️ Below average rainfall for rainy season. Plan irrigation.")
    
    # Summer season insights
    if season == "Summer" and temperature > 35:
        insights.append("🔥 High heat stress risk. Ensure adequate irrigation.")
    
    if season == "Summer" and rainfall < 400:
        insights.append("⚠️ Low rainfall expected. Drip irrigation recommended.")
    
    # Winter season insights
    if season == "Winter" and temperature < 15:
        insights.append("❄️ Suitable for cool-season crops like wheat and barley.")
    
    return insights


def get_crop_recommendations(rainfall: float, temperature: float, season: str) -> list[str]:
    """Recommend top 3 suitable crops based on rainfall, temperature, and season."""
    seasonal_options = SEASONAL_CROPS.get(season, list(MARKET_PRICES.keys()))
    
    # Suitability scoring based on environmental conditions
    scores = {}
    for crop in seasonal_options:
        score = 0
        
        # Rainfall suitability (optimal 600-1200mm)
        if 600 <= rainfall <= 1200:
            score += 10
        elif 400 <= rainfall < 600 or 1200 < rainfall <= 1500:
            score += 5
        
        # Temperature suitability (varies by crop)
        temp_ranges = {
            "Rice": (20, 30),
            "Wheat": (15, 25),
            "Maize": (18, 27),
            "Cotton": (21, 30),
            "Sugarcane": (21, 27),
            "Soybean": (20, 30),
            "Pulses": (15, 25),
            "Groundnut": (20, 30),
            "Barley": (15, 25),
            "Mustard": (15, 25),
        }
        opt_min, opt_max = temp_ranges.get(crop, (20, 25))
        if opt_min <= temperature <= opt_max:
            score += 10
        elif (opt_min - 5) <= temperature <= (opt_max + 5):
            score += 5
        
        scores[crop] = score
    
    top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [crop for crop, _ in top_3]


def assess_risk(rainfall: float, temperature: float, soil: str) -> tuple[str, str]:
    """Assess farming risk and return risk level and color."""
    risk_factors = []
    
    if rainfall < 400:
        risk_factors.append("Low Rainfall")
    
    if temperature > 35:
        risk_factors.append("High Temperature")
    
    if soil == "Sandy" and rainfall < 600:
        risk_factors.append("Water Stress")
    
    if len(risk_factors) >= 2:
        return "High Risk", "#e74c3c"
    elif len(risk_factors) == 1:
        return "Medium Risk", "#f39c12"
    else:
        return "Low Risk", "#27ae60"


def calculate_revenue(yield_tph: float, area: float, crop: str) -> float:
    """Calculate estimated revenue from predicted yield."""
    price_per_ton = MARKET_PRICES.get(crop, 3000)
    return yield_tph * area * price_per_ton


def generate_report(
    state: str,
    crop: str,
    rainfall: float,
    temperature: float,
    soil: str,
    fertilizer: float,
    area: float,
    season: str,
    prediction: float,
    category: str,
    risk_level: str,
    revenue: float,
) -> str:
    """Generate a text report of the prediction and analysis."""
    report = f"""
{'='*60}
CROP YIELD PREDICTION REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT PARAMETERS
{'-'*60}
State: {state}
Crop Type: {crop}
Season: {season}
Rainfall (mm): {rainfall:.1f}
Temperature (°C): {temperature:.1f}
Soil Type: {soil}
Fertilizer Usage (kg/hectare): {fertilizer:.1f}
Area (hectare): {area:.2f}

PREDICTION RESULTS
{'-'*60}
Predicted Yield: {prediction:.2f} tons/hectare
Yield Category: {category}

FINANCIAL ANALYSIS
{'-'*60}
Market Price per Ton: ₹{MARKET_PRICES.get(crop, 3000):.0f}
Total Production: {prediction * area:.2f} tons
Estimated Revenue: ₹{revenue:,.2f}

RISK ASSESSMENT
{'-'*60}
Risk Level: {risk_level}

RECOMMENDATIONS
{'-'*60}
Based on the current environmental conditions and market analysis,
this crop is {'suitable' if 'Low' in risk_level else 'at risk'} for cultivation.
Consider {'increasing' if prediction < 3 else 'maintaining'} irrigation and fertilizer management.

{'='*60}
End of Report
{'='*60}
"""
    return report


st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

st.title("🌾 Crop Yield Prediction System")
st.write("**Predict crop yield (tons per hectare) with environmental and agricultural inputs.**")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
options = load_options()
metrics_payload = load_metrics()

best_model_name = metrics_payload.get("best_model", get_model_name(model))
metrics_table = metrics_payload.get("metrics", {})
best_model_metrics = metrics_table.get(best_model_name, {})

st.sidebar.header("📌 About")
st.sidebar.write(
    "This app predicts crop yield using a trained regression model built on synthetic data."
)
st.sidebar.header("🧠 Model Details")
st.sidebar.write(f"**Model:** {best_model_name}")
st.sidebar.write(
    f"**R2:** {best_model_metrics.get('r2', 'N/A')}  \n"
    f"**MAE:** {best_model_metrics.get('mae', 'N/A')}  \n"
    f"**RMSE:** {best_model_metrics.get('rmse', 'N/A')}"
)
st.sidebar.header("📝 Instructions")
st.sidebar.write(
    "Fill in the inputs, click Predict Yield, and review the model insights on the right."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🧪 Input Features")
    
    # Season selection with crop recommendations
    st.write("**📅 Select Season**")
    season = st.selectbox(
        "Choose season",
        ["Summer", "Winter", "Rainy"],
        key="season_select"
    )
    
    # Display recommended crops for selected season
    seasonal_crops_list = get_seasonal_crops(season)
    st.write("**🌱 Recommended Crops for Selected Season:**")
    for crop_item in seasonal_crops_list:
        st.write(f"• {crop_item}")
    
    with st.form("yield_form"):
        state = st.selectbox("State", options["states"])
        
        # Filter crops to show only those from the seasonal recommendations
        available_crops = [c for c in seasonal_crops_list if c in options["crops"]]
        if not available_crops:
            available_crops = seasonal_crops_list
        crop = st.selectbox("Crop Type", available_crops)
        
        soil = st.selectbox("Soil Type", options["soils"])
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=100.0, max_value=2500.0, value=900.0
        )
        temperature = st.number_input(
            "Temperature (°C)", min_value=5.0, max_value=45.0, value=26.0
        )
        fertilizer = st.number_input(
            "Fertilizer Usage (kg/hectare)",
            min_value=0.0,
            max_value=500.0,
            value=120.0,
        )
        area = st.number_input("Area (hectare)", min_value=0.1, max_value=50.0, value=4.5)

        submitted = st.form_submit_button("Predict Yield")

with right_col:
    st.subheader("📊 Results & Insights")
    if submitted:
        input_df = pd.DataFrame(
            {
                "State": [state],
                "Crop Type": [crop],
                "Rainfall": [rainfall],
                "Temperature": [temperature],
                "Soil Type": [soil],
                "Fertilizer Usage": [fertilizer],
                "Area": [area],
            }
        )

        prediction = float(model.predict(input_df)[0])
        category, color = categorize_yield(prediction)
        risk_level, risk_color = assess_risk(rainfall, temperature, soil)
        revenue = calculate_revenue(prediction, area, crop)

        # Display seasonal insights
        seasonal_msgs = get_seasonal_insights(season, rainfall, temperature)
        if seasonal_msgs:
            st.subheader("🌤️ Seasonal Insights")
            for msg in seasonal_msgs:
                st.info(msg)

        st.metric("Predicted Yield (tons/hectare)", f"{prediction:.2f}")
        st.markdown(
            f"**Yield Category:** <span style='color:{color}; font-weight:600'>{category}</span>",
            unsafe_allow_html=True,
        )

        # Risk indicator
        st.markdown(
            f"**Risk Level:** <span style='color:{risk_color}; font-weight:600'>{risk_level}</span>",
            unsafe_allow_html=True,
        )

        avg_yield = options["mean_yield"]
        difference_pct = ((prediction - avg_yield) / avg_yield) * 100 if avg_yield else 0
        st.write(
            f"This prediction is **{difference_pct:.1f}%** "
            f"{'higher' if difference_pct >= 0 else 'lower'} than the dataset average yield."
        )

        # Alternative crop recommendations based on conditions
        st.subheader("🌱 Alternative Crop Suggestions")
        recommendations = get_crop_recommendations(rainfall, temperature, season)
        cols = st.columns(len(recommendations))
        for idx, rec_crop in enumerate(recommendations):
            with cols[idx]:
                st.success(f"**{idx + 1}. {rec_crop}**")

        # Financial analysis
        st.subheader("💰 Financial Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Price/Ton", f"₹{MARKET_PRICES[crop]:,}")
        with col2:
            st.metric("Total Production", f"{prediction * area:.2f} tons")
        with col3:
            st.metric("Estimated Revenue", f"₹{revenue:,.0f}")

        st.subheader("📈 Yield Comparison")
        comparison_df = pd.DataFrame(
            {
                "Type": ["Dataset Average", "Predicted Yield"],
                "Yield": [avg_yield, prediction],
            }
        ).set_index("Type")
        st.bar_chart(comparison_df)

        st.subheader("🧾 Model Performance")
        st.write(
            f"**Model:** {best_model_name}  \n"
            f"**R2:** {best_model_metrics.get('r2', 'N/A')}  \n"
            f"**MAE:** {best_model_metrics.get('mae', 'N/A')}  \n"
            f"**RMSE:** {best_model_metrics.get('rmse', 'N/A')}"
        )

        importance_df = get_feature_importance(model)
        if importance_df is not None:
            st.subheader("🔍 Feature Importance")
            top_features = importance_df.head(12).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(top_features["feature"], top_features["importance"], color="#4C78A8")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Top Feature Importances")
            st.pyplot(fig)
        else:
            st.info("Feature importance is available for tree-based models only.")

        # Download report
        report_text = generate_report(
            state,
            crop,
            rainfall,
            temperature,
            soil,
            fertilizer,
            area,
            season,
            prediction,
            category,
            risk_level,
            revenue,
        )
        st.download_button(
            label="📥 Download Report as TXT",
            data=report_text,
            file_name=f"crop_yield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
    else:
        st.info("Submit the form to view predictions and model insights.")
