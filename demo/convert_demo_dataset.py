import pandas as pd
import os

# Input and output paths
input_parquet_path = '../data/demo_inferred_data.parquet'  # your big file
output_root = '../data/demo_dataset'

# Load the big file
df = pd.read_parquet(input_parquet_path)

# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Group by date
df['date'] = df['timestamp'].dt.date
grouped = df.groupby('date')

# Write each group to its own folder
for date, group in grouped:
    folder_name = f'date={date}'
    folder_path = os.path.join(output_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Drop the 'date' column before saving
    group = group.drop(columns=['date'])

    # Save parquet
    output_path = os.path.join(folder_path, 'predictions.parquet')
    group.to_parquet(output_path, index=False)

print("✅ Done splitting into daily folders.")
