# 🥇 Gold Price Forecasting

An end-to-end time series forecasting project comparing 9 models —
from simple baselines to deep learning — on 55 years of gold price data.
Built as a portfolio project to demonstrate rigorous data science methodology,
honest model evaluation, and interactive data storytelling.

**🔗 Live Dashboard:**   
**📊 Dataset:** [Global Gold Price Dataset (Kaggle)](https://www.kaggle.com/datasets/mdmahfuzsumon/global-gold-price-dataset-18332026-monthly)

---

## 📌 Key Finding

LSTM captured the direction of gold's recent surge (2024–2026) better than other models, achieving the lowest MAE ($387). However, all models underestimated the magnitude of the increase.

This highlights a core limitation: models trained only on historical prices cannot capture external drivers such as geopolitical events and macroeconomic shifts.

---

## 🧠 What This Project Demonstrates

- Rigorous **stationarity testing** before any modeling (ADF, KPSS,
  Zivot-Andrews)
- **Statistically justified data decisions** — the pre-1971 gold standard
  period was identified and excluded using the Zivot-Andrews structural
  break test, not assumed
- **Honest model evaluation** — models are compared against baselines,
  not just against each other
- End-to-end pipeline from raw data to deployed interactive dashboard

---

## 📁 Project Structure

````
gold-price-forecasting/
│
├── data/
│   └── README.md               ← instructions to download dataset
│
├── notebooks/
│   ├── 01_eda.ipynb             ← exploratory data analysis
│   ├── 02_stationarity.ipynb   ← ADF, KPSS, Zivot-Andrews tests
│   ├── 03_baseline_models.ipynb
│   ├── 04_statistical_models.ipynb  ← ARIMA, Exponential Smoothing
│   ├── 05_ml_models.ipynb      ← Random Forest, XGBoost
│   ├── 06_deep_learning.ipynb  ← LSTM
│   └── 07_model_comparison.ipynb
│
├── models/                     ← saved trained models
├── app/
│   └── app.py                  ← Streamlit dashboard
├── requirements.txt
└── README.md
````

---

## 📊 Model Results

| Model | Type | MAE (USD) |
|-------|------|-----------|
| 🥇 LSTM | Deep Learning | $387.57 |
| ARIMA(1,1,2) | Statistical | $1,096.23 |
| Exponential Smoothing | Statistical | $1,115.83 |
| Naive | Baseline | $1,129.51 |
| Seasonal Naive | Baseline | $1,157.09 |
| Random Forest | Machine Learning | $1,158.22 |
| XGBoost | Machine Learning | $1,175.68 |
| Window Average | Baseline | $1,184.66 |
| Historical Average | Baseline | $2,488.44 |

> Evaluated on a 24-month held-out test set (March 2024 – February 2026).
> MAE = Mean Absolute Error in USD per troy ounce.

---

## 🔍 Methodology

### Why 1971?
The dataset spans 1833–2026 but pre-1971 data is statistically unusable
for forecasting. Gold was a legally fixed government price for over 130
years. The **Zivot-Andrews structural break test** identified January 1971
as the regime change point — confirmed by the Nixon Shock ending the
Bretton Woods gold standard. All modeling uses post-1971 data only.

### Stationarity
The raw price series is non-stationary (ADF p=0.9991, KPSS p<0.01).
Log returns are stationary (ADF p≈0.0000, KPSS p=0.1000). Statistical
models handle non-stationarity through internal differencing (d=1).
ML and deep learning models use lag features and MinMax scaling.

### Why ML models underperformed baselines
Random Forest and XGBoost cannot extrapolate beyond the price range seen
in training data. With gold surging to $5,000 in the test period —
a level never seen in training — both models anchored predictions around
$2,000. This is a known limitation of tree-based models on non-stationary
time series.

---

## 📈 Dashboard Features

- 📉 Interactive price history (1971–2026) with 12 annotated historical
  events
- 📊 Model comparison chart with MAE for all 9 models
- 🔮 Forecast section — select any model and forecast horizon (1–24 months)
- 📐 Confidence intervals on all forecasts
- ⚠️ Disclaimer — forecasts are not financial advice

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Darri916/gold-price-forecasting
cd gold-price-forecasting
pip install -r requirements.txt
```

Download the dataset from Kaggle and place `monthly.csv` in the `data/`
folder, then:

```bash
streamlit run app/app.py
```

---
## 🧠 Final Insight

This project demonstrates that forecasting performance is not just determined by model complexity, but by the information available in the data.

When key external variables are missing, even advanced models fail to capture real-world behavior.

Understanding these limitations is as important as building the models themselves.

---

## 🚀 Future Improvements

- [ ] Add macroeconomic features (USD index, inflation rate, S&P500)
      to move from univariate to multivariate forecasting
- [ ] Walk-forward validation for more realistic evaluation
- [ ] Shorter forecast horizons with tighter confidence intervals
- [ ] Hyperparameter tuning for LSTM and XGBoost

---

## 🛠️ Tech Stack

`Python` `Streamlit` `TensorFlow/Keras` `scikit-learn` `XGBoost`
`statsmodels` `pmdarima` `Plotly` `pandas` `numpy`
