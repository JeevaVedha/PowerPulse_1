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

# Ensure all feature columns exist in the DataFrame
for col in Feature_Columns:
    if col not in sample_df.columns:
        st.write(f"Warning: {col} not found in DataFrame columns.")

# Check for missing values in feature columns
for col in Feature_Columns:
    count = sample_df[col].isnull().sum()
    st.write(f"{col} - Missing values: {count}")     

# Define features and target
X = sample_df[Feature_Columns]
y = sample_df['Global_active_power']       

st.write("Feature Columns:", Feature_Columns)
st.write("Data Types:", sample_df.dtypes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

models = {
    'Linear Regression',
    'Random Forest',
    'Gradient Boosting',
    'Neural Network'
}

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

for model_name in models.keys():
    if model_name == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)  
        r2 = r2_score(y_test, y_pred)
        st.write(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)   
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)   
        st.write(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    elif model_name == 'Neural Network':
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        st.write(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    models[model_name] = model  # Store the trained model

    