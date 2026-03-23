import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from src import (
    load_and_preprocess_power_data,
    create_time_features,
    create_lag_features,
    train_and_evaluate_models,
    plot_feature_importance
)

st.set_page_config(page_title="PowerPulse Dashboard", layout="wide")
st.title("PowerPulse — Household Energy Consumption Forecast")

# ── Load & Process ────────────────────────────────────────────────
with st.spinner("Loading and processing data..."):
    df_raw = load_and_preprocess_power_data()
    df = create_time_features(df_raw.copy())
    df = create_lag_features(df)

if "DateTime" in df.columns:
    df = df.drop("DateTime", axis=1)

target_column = "Global_active_power"
X = df.drop(target_column, axis=1)
y = df[target_column]

with st.spinner("Training models..."):
    results = train_and_evaluate_models(X, y, tune_hyperparams=False)

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Select Model", list(results.keys()))
show_eda   = st.sidebar.checkbox("Show EDA Section", value=True)

model_data = results[model_name]

# ── EDA Section ───────────────────────────────────────────────────
if show_eda:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hourly Average Usage")
        df_raw['hour'] = pd.to_datetime(df_raw['DateTime']).dt.hour
        hourly = df_raw.groupby('hour')['Global_active_power'].mean().reset_index()
        fig_h = px.bar(hourly, x='hour', y='Global_active_power',
                       labels={'Global_active_power': 'Avg Power (kW)', 'hour': 'Hour of Day'},
                       color='Global_active_power', color_continuous_scale='Blues')
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        st.subheader("Monthly Average Usage")
        df_raw['month'] = pd.to_datetime(df_raw['DateTime']).dt.month
        monthly = df_raw.groupby('month')['Global_active_power'].mean().reset_index()
        fig_m = px.bar(monthly, x='month', y='Global_active_power',
                       labels={'Global_active_power': 'Avg Power (kW)', 'month': 'Month'},
                       color='Global_active_power', color_continuous_scale='Teal')
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Correlation Heatmap")
    import seaborn as sns
    num_cols = ['Global_active_power', 'Global_reactive_power',
                'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_raw[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax)
    st.pyplot(fig_corr)

# ── Model Performance ─────────────────────────────────────────────
st.header(f"Model Performance — {model_name}")

col1, col2, col3 = st.columns(3)
col1.metric("MAE",      f"{model_data['MAE']:.4f}")
col2.metric("RMSE",     f"{model_data['RMSE']:.4f}")
col3.metric("R² Score", f"{model_data['R2']:.4f}")

# ── Model Comparison ──────────────────────────────────────────────
st.subheader("All Models Comparison")
comparison_data = {
    'Model': list(results.keys()),
    'MAE':   [results[m]['MAE']  for m in results],
    'RMSE':  [results[m]['RMSE'] for m in results],
    'R²':    [results[m]['R2']   for m in results],
}
st.dataframe(pd.DataFrame(comparison_data).set_index('Model').style.highlight_min(
    subset=['MAE', 'RMSE'], color='lightgreen'
).highlight_max(subset=['R²'], color='lightgreen'))

# ── Actual vs Predicted ───────────────────────────────────────────
st.subheader("Actual vs Predicted Trend")

fig = go.Figure()
fig.add_trace(go.Scatter(y=model_data["y_test"].values[:500],
                         mode='lines', name='Actual', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(y=model_data["y_pred"][:500],
                         mode='lines', name='Predicted', line=dict(color='coral')))
fig.update_layout(height=450, template="plotly_white", hovermode="x unified",
                  xaxis_title="Time Index", yaxis_title="Global Active Power (kW)")
st.plotly_chart(fig, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────
if model_name in ["Random Forest", "Gradient Boosting"] and "model" in model_data:
    st.subheader("Feature Importance")
    import numpy as np
    model         = model_data["model"]
    feature_names = model_data["feature_names"]
    importances   = model.feature_importances_
    indices       = np.argsort(importances)[::-1][:10]

    fig_fi = px.bar(
        x=[importances[i] for i in indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color=[importances[i] for i in indices],
        color_continuous_scale='Blues'
    )
    fig_fi.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_fi, use_container_width=True)