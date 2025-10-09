from ucimlrepo import fetch_ucirepo
import pandas as pd
import streamlit as st

dataset = fetch_ucirepo(id=235)
df = dataset.data.original  # or .features if .original is not available

#Sampling the data to 500,000 rows for performance
sample_df = df.sample(n=500000, random_state=42)
print("Original Data shape:", len(df))
print("Sample Data shape:", len(sample_df))

# Preprocess Date and Time
sample_df['Date'] = pd.to_datetime(sample_df['Date'], format='%d/%m/%Y')
sample_df['Time'] = pd.to_datetime(sample_df['Time'], format='%H:%M:%S').dt.time
sample_df['DateTime'] = pd.to_datetime(sample_df['Date'].astype(str) + ' ' + sample_df['Time'].astype(str))
sample_df.drop(['Date', 'Time'], axis=1, inplace=True)

# List of columns to process
columns = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

null_percentages = {}
# Process each column
for col in columns:
    # Calculate % of nulls
    null_pct = sample_df[col].isna().sum() / len(sample_df) * 100
    null_percentages[col] = null_pct
    
    # Convert to numeric, coerce errors, fill NA with 0, cast to float64
    sample_df[col] = (
        pd.to_numeric(sample_df[col], errors='coerce')
        .fillna(0)
        .astype('float64')
    )

# If you want, you can show the null percentages
for col, pct in null_percentages.items():
    print(f"{col} null percentage: {pct:.2f}%")
sample_df.set_index('DateTime', inplace=True)

#Feature Columns enhanced

# Daily average & weekly SMA
daily_avg = sample_df['Global_active_power'].resample('D').mean()
daily_avg_sma7 = daily_avg.rolling(window=7).mean()

# Hourly average & peak hour
sample_df['hour'] = sample_df.index.hour
hourly_avg = sample_df.groupby('hour')['Global_active_power'].mean()
peak_hour = hourly_avg.idxmax()
print("Peak consumption hour:", peak_hour)

# Rolling average over past hour
sample_df['GAP_rolling_avg'] = sample_df['Global_active_power'].rolling(window=60).mean()
sample_df['GAP_rolling_avg'] = sample_df['GAP_rolling_avg'].fillna(method='bfill')

Feature_Columns = ['Global_reactive_power', 'Voltage',
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

#--------------------------------------------------------------------------------------
st.set_page_config(page_title="Household Power Consumption Prediction", layout="wide")
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Household Power Consumption Prediction</h1>", unsafe_allow_html=True)

    
st.subheader("Data Overview")
st.write("Original Data shape:", len(df))
st.write("Sample Data shape:", len(sample_df))
st.write("First 5 rows:")
st.write(sample_df.head())

X = sample_df[Feature_Columns]
y = sample_df['Global_active_power']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("Training Linear Regression Model...")
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lr}).head(50))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = root_mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
with col2:
    st.write("Training Random Forest Model...")
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf}).head(50))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = root_mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
with col3:
    st.write("Training Gradient Boosting Model...")
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gb}).head(50))
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = root_mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

with col4:
    st.write("Training MLP Regressor Model...")
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)

    st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_mlp}).head(50))
    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    rmse_mlp = root_mean_squared_error(y_test, y_pred_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)

metrics = {
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "MLP Regressor"],
    "MAE": [mae_lr, mae_rf, mae_gb, mae_mlp],
    "RMSE": [rmse_lr, rmse_rf, rmse_gb, rmse_mlp],
    "R²": [r2_lr, r2_rf, r2_gb, r2_mlp]
}

st.subheader("Model Performance Comparison")
st.table(pd.DataFrame(metrics).set_index('Model'))

 

import altair as alt

results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "MLP Regressor"],
    "MAE": [mae_lr, mae_rf, mae_gb, mae_mlp],
    "RMSE": [rmse_lr, rmse_rf, rmse_gb, rmse_mlp],
    "R²": [r2_lr, r2_rf, r2_gb, r2_mlp]
})

# Melt the DataFrame to long form: each metric becomes a row
df_melted = results_df.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE", "R²"],
    var_name="Metric",
    value_name="Value"
)

# Now plot
chart = (
    alt.Chart(df_melted)
    .mark_line(point=True)
    .encode(
        x=alt.X("Metric:N", sort=["MAE", "RMSE", "R²"]),
        y=alt.Y("Value:Q"),
        color="Model:N",  # if you later have multiple models, this distinguishes them
        tooltip=["Model:N", "Metric:N", "Value:Q"]
    )
    .properties(
        title="Model Performance Metrics",
        width=400,
        height=300
    )
)

st.altair_chart(chart, use_container_width=True)