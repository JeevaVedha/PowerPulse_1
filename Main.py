from ucimlrepo import fetch_ucirepo
import pandas as pd
import streamlit as st

dataset = fetch_ucirepo(id=235)
df = dataset.data.original  # or .features if .original is not available

sample_df = df.sample(n=500000, random_state=42)
print("Sample data:")
print(sample_df.head())


print("Original Data shape:", len(df))
print("Sample Data shape:", len(sample_df))

sample_df['Date'] = pd.to_datetime(sample_df['Date'], format='%d/%m/%Y')
sample_df['Time'] = pd.to_datetime(sample_df['Time'], format='%H:%M:%S').dt.time
sample_df['DateTime'] = pd.to_datetime(sample_df['Date'].astype(str) + ' ' + sample_df['Time'].astype(str))
sample_df.drop(['Date', 'Time'], axis=1, inplace=True)

GAP = sample_df['Global_active_power'].isnull().sum() / len(sample_df) * 100
print(f"Global Active Power (GAP) after cleaning: {GAP:.2f}%")
#df['Global_active_power'] = df['Global_active_power'].replace('?', np.nan).astype(float)
sample_df['Global_active_power'] = pd.to_numeric(sample_df['Global_active_power'], errors='coerce').fillna(0).astype('float64')

GRP = sample_df['Global_reactive_power'].isnull().sum() / len(sample_df) * 100
print(f"Global Reactive Power (GRP) as a percentage of total power: {GRP:.2f}%")
#df['Global_reactive_power'] = df['Global_reactive_power'].replace('?', np.nan).astype(float)
sample_df['Global_reactive_power'] = pd.to_numeric(sample_df['Global_reactive_power'], errors='coerce').fillna(0).astype('float64')

V = sample_df['Voltage'].isnull().sum() / len(sample_df) * 100
print(f"Voltage as a percentage of total power: {V:.2f}%")
#df['Voltage'] = df['Voltage'].replace('?', np.nan).astype(float)
sample_df['Voltage'] = pd.to_numeric(sample_df['Voltage'], errors='coerce').fillna(0).astype('float64')

GI = sample_df['Global_intensity'].isnull().sum() / len(sample_df) * 100
print(f"Global Intensity (GI) as a percentage of total power: {GI:.2f}%")
#df['Global_intensity'] = df['Global_intensity'].replace('?', np.nan).astype(float)
sample_df['Global_intensity'] = pd.to_numeric(sample_df['Global_intensity'], errors='coerce').fillna(0).astype('float64')

SM1 = sample_df['Sub_metering_1'].isnull().sum() / len(sample_df) * 100
print(f"Sub Metering 1 (SM1) as a percentage of total power: {SM1:.2f}%")
#df['Sub_metering_1'] = df['Sub_metering_1'].replace('?', np.nan).astype(float)
sample_df['Sub_metering_1'] = pd.to_numeric(sample_df['Sub_metering_1'], errors='coerce').fillna(0).astype('float64')

SM2 = sample_df['Sub_metering_2'].isnull().sum() / len(sample_df) * 100
print(f"Sub Metering 2 (SM2) as a percentage of total power: {SM2:.2f}%")
#df['Sub_metering_2'] = df['Sub_metering_2'].replace('?', np.nan).astype(float)
sample_df['Sub_metering_2'] = pd.to_numeric(sample_df['Sub_metering_2'], errors='coerce').fillna(0).astype('float64')

SM3 = sample_df['Sub_metering_3'].isnull().sum() / len(sample_df) * 100
print(f"Sub Metering 3 (SM3) as a percentage of total power: {SM3:.2f}%")    
#df['Sub_metering_3'] = df['Sub_metering_3'].replace('?', np.nan).astype(float)
sample_df['Sub_metering_3'] = pd.to_numeric(sample_df['Sub_metering_3'], errors='coerce').fillna(0).astype('float64')

sample_df.set_index('DateTime', inplace=True)

# Daily average & weekly SMA
daily_avg = sample_df['Global_active_power'].resample('D').mean()
daily_avg_rolling = daily_avg.rolling(window=7).mean()

# Hourly average & peak hour
sample_df['hour'] = sample_df.index.hour
hourly_avg = sample_df.groupby('hour')['Global_active_power'].mean()
peak_hour = hourly_avg.idxmax()
print("Peak consumption hour:", peak_hour)

Feature_Columns = ['Global_reactive_power', 'Voltage',
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','hour','daily_avg','daily_avg_rolling']

X = sample_df[Feature_Columns]
y = sample_df['Global_active_power']

print(f"Feature Columns:", Feature_Columns)
print(sample_df.dtypes) 

print("First 5 rows:")
print(sample_df.head())
print(sample_df.dtypes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model Training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Random Forest Regressor Model Training
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
print("Linear Regression Model Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

print("Random Forest Model Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, rf_y_pred))       
print("Root Mean Squared Error:", root_mean_squared_error(y_test, rf_y_pred))
print("R^2 Score:", r2_score(y_test, rf_y_pred))

#print(f"Feature Columns:", Feature_Columns)
#print(df.dtypes) 

metrics = {
    "Linear Regression": {
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Root Mean Squared Error": root_mean_squared_error(y_test, y_pred),
        "R^2 Score": r2_score(y_test, y_pred)
    },
    "Random Forest": {
        "Mean Absolute Error": mean_absolute_error(y_test, rf_y_pred),
        "Root Mean Squared Error": root_mean_squared_error(y_test, rf_y_pred),
        "R^2 Score": r2_score(y_test, rf_y_pred)
    }
}

st.title("Power Consumption Prediction Models")

for model, metrics in metrics.items():
    st.subheader(model)
    for metric_name, metric_value in metrics.items():
        st.write(f"{metric_name}: {metric_value}")

st.write("Measurement Metrics:", metrics)