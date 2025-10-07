from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=235)
df = dataset.data.original  # or .features if .original is not available

sample_df = df.sample(n=1000000, random_state=42)
print("Sample data:")
print(sample_df.head())


print("Original Data shape:", len(df))
print("Sample Data shape:", len(sample_df))
