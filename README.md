# Crop Yield Prediction System

Predict crop yield (tons per hectare) from environmental and agricultural factors using advanced machine learning with seasonal analysis and financial insights.

## Project Structure

```
Crop-Yield-Prediction/
│
├── data/
├── model/
├── notebooks/
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Features

- **Seasonal Prediction**: Summer, Winter, and Rainy season-specific crop recommendations
- **Crop Recommendations**: Intelligent crop suggestions based on environmental conditions
- **Risk Assessment**: Automatic risk evaluation with color-coded indicators
- **Financial Analysis**: Revenue estimation based on market prices
- **Seasonal Insights**: Climate-based recommendations and warnings
- **Download Reports**: Export prediction summaries as text files
- **Feature Importance**: Visual analysis of model feature contributions
- **Multiple Models**: Linear Regression, Random Forest, and XGBoost comparison

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Train the model and generate the dataset:
   ```bash
   python train.py
   ```

2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Outputs

- **Synthetic Dataset**: data/crop_data.csv
- **Best Model**: model/crop_model.pkl
- **Metrics Report**: model/model_metrics.json
- **EDA Plots**: notebooks/eda/

## Screenshots

Add screenshots here:
- `screenshots/prediction_interface.png`
- `screenshots/financial_analysis.png`
- `screenshots/seasonal_insights.png`
- `screenshots/risk_assessment.png`

(Replace with actual images after running the project.)
