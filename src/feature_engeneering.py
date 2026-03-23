import pandas as pd


def create_time_features(df):
    """
    Create time-based features from DateTime column.
    Includes cyclical encoding for hour (sin/cos) for better model learning.
    """

    if 'DateTime' not in df.columns:
        raise ValueError("DateTime column not found in DataFrame")

    if not hasattr(df['DateTime'], "dt"):
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

    # Basic time features
    df['hour']        = df['DateTime'].dt.hour
    df['day']         = df['DateTime'].dt.day
    df['month']       = df['DateTime'].dt.month
    df['day_of_week'] = df['DateTime'].dt.dayofweek

    # Cyclical encoding for hour (captures 23→0 continuity)
    import numpy as np
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Peak hour flag (7-9 AM and 6-10 PM are typical peak hours)
    df['is_peak_hour'] = df['hour'].apply(
        lambda h: 1 if (7 <= h <= 9) or (18 <= h <= 22) else 0
    )

    # Weekend flag
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)

    df.dropna(subset=['DateTime'], inplace=True)
    return df


def create_lag_features(df):
    """
    Create lag and rolling features for time-series modeling.
    Also adds daily average power feature.
    """

    if 'Global_active_power' not in df.columns:
        raise ValueError("Global_active_power column not found")

    # Sort by DateTime (VERY IMPORTANT for time series)
    df = df.sort_values('DateTime')

    # Lag features
    df['lag_1']  = df['Global_active_power'].shift(1)
    df['lag_24'] = df['Global_active_power'].shift(24)

    # Rolling statistics
    df['rolling_mean_24'] = df['Global_active_power'].rolling(window=24).mean()
    df['rolling_std_24']  = df['Global_active_power'].rolling(window=24).std()

    # Daily average power
    df['date_only'] = df['DateTime'].dt.date
    daily_avg = df.groupby('date_only')['Global_active_power'].transform('mean')
    df['daily_avg_power'] = daily_avg
    df.drop('date_only', axis=1, inplace=True)

    # Drop NaN rows created by shift/rolling
    df = df.dropna().reset_index(drop=True)

    return df