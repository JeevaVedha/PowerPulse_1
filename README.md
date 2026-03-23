# ⚡ PowerPulse — Household Energy Usage Forecast

> A machine learning project to predict household energy consumption using historical power usage data, featuring time-series feature engineering, multiple regression models, and an interactive Streamlit dashboard.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Business Use Cases](#business-use-cases)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Approach & Methodology](#approach--methodology)
- [Feature Engineering](#feature-engineering)
- [Models Used](#models-used)
- [Results](#results)
- [Dashboard](#dashboard)
- [Tech Stack](#tech-stack)

---

## 📖 Project Overview

Energy management is a critical issue for both households and energy providers. Accurate prediction of energy consumption enables better planning, cost reduction, and resource optimization.

**PowerPulse** builds a machine learning pipeline that:
- Loads and preprocesses real household power consumption data
- Engineers time-series features (lag, rolling, cyclical encoding)
- Trains and compares 3 regression models
- Visualizes predictions through an interactive Streamlit dashboard

---

## 💼 Business Use Cases

| Use Case | Description |
|----------|-------------|
| **Energy Management** | Help households monitor usage, reduce bills, and adopt efficient habits |
| **Demand Forecasting** | Enable energy providers to predict load and optimize pricing |
| **Anomaly Detection** | Identify irregular usage patterns indicating faults or unauthorized usage |
| **Smart Grid Integration** | Power real-time predictive analytics for smart energy systems |
| **Environmental Impact** | Reduce carbon footprints by optimizing consumption |

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) (Dataset ID: 235)
- **Records:** ~2 million rows (per-minute readings)
- **Period:** December 2006 – November 2010
- **Sample Used:** 100,000 rows (random sample, `random_state=42`)

### Columns

| Column | Description |
|--------|-------------|
| `Global_active_power` | Household global active power (kW) — **TARGET** |
| `Global_reactive_power` | Household global reactive power (kW) |
| `Voltage` | Minute-averaged voltage (V) |
| `Global_intensity` | Household global current intensity (A) |
| `Sub_metering_1` | Kitchen energy (Wh) |
| `Sub_metering_2` | Laundry room energy (Wh) |
| `Sub_metering_3` | Water heater & AC energy (Wh) |

---

## 📁 Project Structure

```
PowerPulse_2/
│
├── main.py                    # Entry point — runs full ML pipeline
├── app.py                     # Streamlit dashboard
│
├── src/
│   ├── __init__.py            # Module exports
│   ├── preprocessing.py       # Data loading, cleaning, EDA
│   ├── feature_engineering.py # Time, lag, and aggregate features
│   └── model.py               # Model training, tuning, evaluation, plots
│
├── Data/
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Processed data (if saved)
│
├── Notebooks/                 # Jupyter notebooks for exploration
│
├── eda_distribution.png       # Generated EDA plots
├── eda_hourly_usage.png
├── eda_monthly_usage.png
├── eda_correlation_heatmap.png
├── feature_importance.png
├── model_comparison.png
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PowerPulse.git
cd PowerPulse_2
```

### 2. Install dependencies

```bash
pip install ucimlrepo pandas numpy scikit-learn matplotlib seaborn streamlit plotly
```

Or install all at once:

```bash
pip install ucimlrepo pandas numpy scikit-learn matplotlib seaborn streamlit plotly
```

---

## ▶️ How to Run

### Run the ML pipeline (terminal output + plots)

```bash
cd PowerPulse_2
python main.py
```

### Run the Streamlit dashboard

```bash
cd PowerPulse_2
streamlit run app.py
```

> **Note:** Always run from the `PowerPulse_2/` root folder, not from inside `src/`.

---

## 🔬 Approach & Methodology

### Step 1 — Data Loading & Preprocessing (`src/preprocessing.py`)

- Fetches UCI dataset using `ucimlrepo` (ID: 235)
- Randomly samples 100,000 rows for performance
- Merges `Date` and `Time` into a single `DateTime` column
- Converts all 7 power columns to `float64`
- Handles missing values using **ffill + bfill** (time-series safe)
  - `fillna(0)` is avoided — 0 is a fake value that corrupts lag features
  - Forward fill uses the previous time step's value (realistic for sensor data)

### Step 2 — EDA (`src/preprocessing.py → perform_eda()`)

- Distribution plot of `Global_active_power`
- Average usage by hour of day (identifies peak hours)
- Average usage by month (seasonal patterns)
- Correlation heatmap across all 7 numeric columns

### Step 3 — Feature Engineering (`src/feature_engineering.py`)

See [Feature Engineering](#feature-engineering) section below.

### Step 4 — Model Training & Evaluation (`src/model.py`)

- 80/20 train-test split with `shuffle=False` (preserves time order)
- Trains 3 models: Linear Regression, Random Forest, Gradient Boosting
- Evaluates using MAE, RMSE, R²
- Optionally runs `RandomizedSearchCV` with `TimeSeriesSplit` for RF tuning

---

## 🛠️ Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| `hour` | Time | Hour of day (0–23) |
| `day` | Time | Day of month |
| `month` | Time | Month of year |
| `day_of_week` | Time | Day of week (0=Mon) |
| `hour_sin` / `hour_cos` | Cyclical | Sine/cosine encoding of hour (captures 23→0 continuity) |
| `is_peak_hour` | Binary | 1 if 7–9 AM or 6–10 PM, else 0 |
| `is_weekend` | Binary | 1 if Saturday or Sunday |
| `lag_1` | Lag | Previous minute's power reading |
| `lag_24` | Lag | Same time from 24 steps ago |
| `rolling_mean_24` | Rolling | 24-step rolling average |
| `rolling_std_24` | Rolling | 24-step rolling standard deviation |
| `daily_avg_power` | Aggregate | Average power for that calendar day |

> **Why lag features matter:** Without lag features, the model has no memory of recent history. `lag_1` alone often accounts for 60–70% of feature importance in time-series power prediction.

---

## 🤖 Models Used

### 1. Linear Regression
- Baseline model with `StandardScaler`
- Fast and interpretable
- Best for linear relationships

### 2. Random Forest Regressor
- Ensemble of 100 decision trees
- Handles non-linearity well
- Feature importance built-in
- Optional hyperparameter tuning via `RandomizedSearchCV`

### 3. Gradient Boosting Regressor
- Sequential boosting of weak learners
- `n_estimators=100`, `learning_rate=0.1`
- Best overall accuracy in this project

### Train/Test Split Strategy
```python
train_test_split(X, y, test_size=0.2, shuffle=False)
```
`shuffle=False` is critical for time-series data — shuffling would leak future data into training, making results unrealistically optimistic.

---

## 📈 Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~0.28 | ~0.38 | ~0.88 |
| Random Forest | ~0.09 | ~0.15 | ~0.97 |
| **Gradient Boosting** | **~0.08** | **~0.13** | **~0.98** |

> Actual values will vary slightly based on the random sample drawn from the dataset.

**Best Model: Gradient Boosting** — highest R² score and lowest MAE/RMSE.

---

## 🖥️ Dashboard

The Streamlit dashboard (`app.py`) includes:

- **EDA Section** — Hourly usage chart, monthly usage chart, correlation heatmap
- **Model Selector** — Dropdown to switch between Linear Regression, Random Forest, Gradient Boosting
- **Metrics Cards** — MAE, RMSE, R² displayed side-by-side
- **Model Comparison Table** — All 3 models compared, best values highlighted
- **Actual vs Predicted Chart** — Interactive Plotly chart with hover and zoom
- **Feature Importance Chart** — Top 10 features ranked by importance (for RF and GB)

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, evaluation, preprocessing |
| `ucimlrepo` | Fetching UCI dataset |
| `matplotlib` | Static plots |
| `seaborn` | Correlation heatmap, distribution plots |
| `plotly` | Interactive dashboard charts |
| `streamlit` | Web dashboard |

---

## 📋 Evaluation Metrics

- **MAE (Mean Absolute Error)** — Average magnitude of prediction errors
- **RMSE (Root Mean Squared Error)** — Penalises large errors more than MAE
- **R² Score** — Proportion of variance explained by the model (1.0 = perfect)
- **Feature Importance** — Which features most influence predictions

---

## 🚀 Future Improvements

- Add LSTM / Neural Network model for deeper time-series learning
- Incorporate external weather data as additional features
- Deploy dashboard to Streamlit Cloud
- Add anomaly detection for irregular usage patterns
- Implement hourly/daily forecasting windows

---

## 📚 References

- [UCI Dataset — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)