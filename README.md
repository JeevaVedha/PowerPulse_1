# 🏠 Household Power Consumption Prediction Dashboard

This project builds an interactive Streamlit dashboard to analyze and predict household power consumption using multiple regression models.
It uses the Individual Household Electric Power Consumption Dataset from the UCI Machine Learning Repository (Dataset ID: 235).

## 📘 Overview

The goal of this project is to:

Fetch and preprocess the UCI power consumption dataset.

Sample the data for faster computation.

Perform feature engineering (daily averages, rolling averages, peak hour analysis).

Train and evaluate multiple regression models to predict Global Active Power.

Compare model performance visually using charts and tables in Streamlit.


## 🧰 Technologies Used

Python 3.9+

Streamlit – For building the interactive dashboard

Pandas – For data processing and manipulation

Scikit-learn – For model training and evaluation

Altair – For visualization of model metrics

UCI ML Repository API (ucimlrepo) – To fetch the dataset directly


## 📦 Installation
### 1️⃣ Clone the repository
git clone https://github.com/JeevaVedha/PowerPulse_1.git
cd household-power-consumption-prediction

### 2️⃣ Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

### 3️⃣ Install required dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt yet, create one with:

streamlit
pandas
scikit-learn
altair
ucimlrepo

## ⚙️ How to Run the Dashboard

Run the following command in your terminal:

streamlit run Main.py


The app will automatically open in your browser (default: http://localhost:8501)

If not, copy and paste the URL from the terminal into your web browser.


## 📊 Features and Workflow
### 1. Data Loading

The dataset is automatically fetched from UCI ML Repository using:

from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=235)

### 2. Data Sampling & Preprocessing

Samples 500,000 rows for efficiency.

Converts Date and Time columns into a combined DateTime index.

Handles missing values and converts measurement columns to numeric.

### 3. Feature Engineering

Computes daily averages, weekly rolling averages (SMA).

Extracts peak power consumption hour.

Calculates hourly rolling averages of power consumption.

### 4. Model Training

Four machine learning regression models are trained:

Model	Description
Linear Regression	Simple baseline model
Random Forest Regressor	Ensemble-based non-linear model
Gradient Boosting Regressor	Sequential ensemble model
MLP Regressor	Neural network-based model
### 5. Model Evaluation Metrics

Each model is evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

### 6. Visualization

Bar charts comparing actual vs predicted values for each model.

Altair line chart comparing model performance across MAE, RMSE, and R² metrics.

Interactive Streamlit layout with four parallel columns displaying model outputs.


## 📈 Example Outputs
🔹 Data Overview

Displays dataset shape and first few rows.

🔹 Model Results

Shows predicted vs actual energy consumption comparisons:

Linear Regression: MAE = 0.18, RMSE = 0.22, R² = 0.92
Random Forest: MAE = 0.07, RMSE = 0.11, R² = 0.98
Gradient Boosting: MAE = 0.09, RMSE = 0.12, R² = 0.97
MLP Regressor: MAE = 0.11, RMSE = 0.15, R² = 0.95

🔹 Altair Visualization

Displays performance comparison between models interactively.

## 💡 Future Enhancements

Add real-time energy consumption forecasting using LSTM or Prophet.

Implement interactive filtering by date/time range.

Include energy-saving recommendations based on usage patterns.

Add downloadable model performance reports in PDF/CSV format.