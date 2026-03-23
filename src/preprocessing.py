from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_power_data(
    dataset_id: int = 235,
    sample_size: int = 100000,
    random_state: int = 42
):
    """
    Loads and preprocesses the UCI Individual Household Power Consumption dataset.

    Steps:
    1. Fetch dataset from UCI repository
    2. Sample data for performance
    3. Combine Date and Time into DateTime
    4. Convert selected columns to numeric
    5. Handle missing values using ffill + bfill (time-series safe)

    Returns:
        sample_df (pd.DataFrame): Cleaned and processed dataframe
    """

    # Step 1: Load dataset
    dataset = fetch_ucirepo(id=dataset_id)

    try:
        df = dataset.data.original
    except AttributeError:
        df = dataset.data.features

    print("Original Data shape:", df.shape)

    # Step 2: Sampling
    sample_df = df.sample(n=sample_size, random_state=random_state)
    print("Sample Data shape:", sample_df.shape)

    # Step 3: Combine Date and Time
    sample_df['Date'] = pd.to_datetime(sample_df['Date'], format='%d/%m/%Y')
    sample_df['Time'] = pd.to_datetime(sample_df['Time'], format='%H:%M:%S').dt.time
    sample_df['DateTime'] = pd.to_datetime(
        sample_df['Date'].astype(str) + ' ' + sample_df['Time'].astype(str)
    )
    sample_df.drop(['Date', 'Time'], axis=1, inplace=True)
    print("DateTime column created.")

    # Step 4: Columns to process
    columns = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]

    # Step 5: Convert to numeric + time-series safe fill (ffill + bfill)
    for col in columns:
        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')

        null_pct = sample_df[col].isna().sum() / len(sample_df) * 100
        print(f"{col} null %: {null_pct:.2f}%")

        # Sort by DateTime before fill to maintain time order
        sample_df = sample_df.sort_values('DateTime')
        sample_df[col] = (
            sample_df[col]
            .ffill()                        # Forward fill (previous time step)
            .bfill()                        # Backward fill (for leading NaNs)
            .fillna(sample_df[col].median()) # Last safety net
            .astype('float64')
        )

    print("Preprocessing Completed")
    return sample_df


def perform_eda(df):
    """
    Performs Exploratory Data Analysis on the power consumption dataset.
    Generates and saves key visualizations.
    """

    print("\n--- EDA Started ---")
    print("Shape:", df.shape)
    print("\nBasic Stats:")
    print(df.describe())

    # 1. Distribution of Global Active Power
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Global_active_power'], bins=50, kde=True, color='steelblue')
    plt.title('Global Active Power Distribution')
    plt.xlabel('Global Active Power (kW)')
    plt.tight_layout()
    plt.savefig('eda_distribution.png', dpi=100)
    plt.close()
    print("Saved: eda_distribution.png")

    # 2. Hourly Average Usage
    df['hour'] = df['DateTime'].dt.hour
    hourly_avg = df.groupby('hour')['Global_active_power'].mean()

    plt.figure(figsize=(10, 4))
    hourly_avg.plot(kind='bar', color='coral')
    plt.title('Average Energy Usage by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Avg Global Active Power (kW)')
    plt.tight_layout()
    plt.savefig('eda_hourly_usage.png', dpi=100)
    plt.close()
    print("Saved: eda_hourly_usage.png")

    # 3. Monthly Average Usage
    df['month'] = df['DateTime'].dt.month
    monthly_avg = df.groupby('month')['Global_active_power'].mean()

    plt.figure(figsize=(10, 4))
    monthly_avg.plot(kind='bar', color='teal')
    plt.title('Average Energy Usage by Month')
    plt.xlabel('Month')
    plt.ylabel('Avg Global Active Power (kW)')
    plt.tight_layout()
    plt.savefig('eda_monthly_usage.png', dpi=100)
    plt.close()
    print("Saved: eda_monthly_usage.png")

    # 4. Correlation Heatmap
    num_cols = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png', dpi=100)
    plt.close()
    print("Saved: eda_correlation_heatmap.png")

    print("--- EDA Completed ---\n")