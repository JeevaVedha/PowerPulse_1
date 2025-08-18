from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=235)
df = dataset.data.original  # or .features if .original is not available


#print("\nDataset metadata:")
#print(dataset.metadata)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df.drop(['Date', 'Time'], axis=1, inplace=True)

GAP = df['Global_active_power'].isnull().sum() / len(df) * 100
print(f"Global Active Power (GAP) after cleaning: {GAP:.2f}%")
#df['Global_active_power'] = df['Global_active_power'].replace('?', np.nan).astype(float)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce').fillna(0).astype('float64')

GRP = df['Global_reactive_power'].isnull().sum() / len(df) * 100
print(f"Global Reactive Power (GRP) as a percentage of total power: {GRP:.2f}%")
#df['Global_reactive_power'] = df['Global_reactive_power'].replace('?', np.nan).astype(float)
df['Global_reactive_power'] = pd.to_numeric(df['Global_reactive_power'], errors='coerce').fillna(0).astype('float64')

V = df['Voltage'].isnull().sum() / len(df) * 100
print(f"Voltage as a percentage of total power: {V:.2f}%")
#df['Voltage'] = df['Voltage'].replace('?', np.nan).astype(float)
df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce').fillna(0).astype('float64')

GI = df['Global_intensity'].isnull().sum() / len(df) * 100
print(f"Global Intensity (GI) as a percentage of total power: {GI:.2f}%")
#df['Global_intensity'] = df['Global_intensity'].replace('?', np.nan).astype(float)
df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce').fillna(0).astype('float64')

SM1 = df['Sub_metering_1'].isnull().sum() / len(df) * 100
print(f"Sub Metering 1 (SM1) as a percentage of total power: {SM1:.2f}%")
#df['Sub_metering_1'] = df['Sub_metering_1'].replace('?', np.nan).astype(float)
df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce').fillna(0).astype('float64')

SM2 = df['Sub_metering_2'].isnull().sum() / len(df) * 100
print(f"Sub Metering 2 (SM2) as a percentage of total power: {SM2:.2f}%")
#df['Sub_metering_2'] = df['Sub_metering_2'].replace('?', np.nan).astype(float)
df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce').fillna(0).astype('float64')

SM3 = df['Sub_metering_3'].isnull().sum() / len(df) * 100
print(f"Sub Metering 3 (SM3) as a percentage of total power: {SM3:.2f}%")    
#df['Sub_metering_3'] = df['Sub_metering_3'].replace('?', np.nan).astype(float)
df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce').fillna(0).astype('float64')

df.set_index('DateTime', inplace=True)

# Daily average & weekly SMA
daily_avg = df['Global_active_power'].resample('D').mean()
daily_avg_sma7 = daily_avg.rolling(window=7).mean()

# Hourly average & peak hour
df['hour'] = df.index.hour
hourly_avg = df.groupby('hour')['Global_active_power'].mean()
peak_hour = hourly_avg.idxmax()
print("Peak consumption hour:", peak_hour)

Feature_Columns = ['Global_reactive_power', 'Voltage',
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','hour']

X = df[Feature_Columns]
y = df['Global_active_power']

print(f"Feature Columns:", Feature_Columns)
print(df.dtypes) 

print("First 5 rows:")
print(df.head())
print(df.dtypes)
