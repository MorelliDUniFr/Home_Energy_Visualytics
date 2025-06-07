import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time as time_module
import os
from config_loader import load_config
from joblib import load

config, config_dir = load_config()

env = config['Settings']['environment']
inference_timestamp = config['Inference']['inference_timestamp']
data_path = config[env]['data_path']
model_file = config['Data']['model_file']
inferred_data_file = config['Data']['inferred_data_file']
infer_data_file = config['Data']['infer_data_file']
appliances_file = config['Data']['appliances_file']
input_scaler_file = config['Data']['input_scaler_file']
target_scalers_file = config['Data']['target_scalers_file']
batch_size = int(config['Inference']['batch_size'])

model_path = os.path.join(data_path, model_file)
inferred_data_path = os.path.join(data_path, inferred_data_file)
infer_data_path = os.path.join(data_path, infer_data_file)
appliances_path = os.path.join(data_path, appliances_file)
input_scaler_path = os.path.join(data_path, input_scaler_file)
target_scalers_path = os.path.join(data_path, target_scalers_file)

input_scaler = load(input_scaler_path)
target_scaler = load(target_scalers_path)

device = torch.device('cpu')

# Read appliance names from the text file
with open(appliances_path, 'r') as f:
    appliances_list = [line.strip() for line in f.readlines()]


def create_day_dataset():
    df = pd.read_parquet(infer_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    first_date = df['timestamp'].dt.date.min()
    first_day_df = df[df['timestamp'].dt.date == first_date]

    timestamps = first_day_df['timestamp'].reset_index(drop=True)  # Keep timestamps for later
    first_day_df = first_day_df.drop(columns=['timestamp'])
    selected_columns = first_day_df.columns[1:10]  # Adjust as needed
    first_day_df = first_day_df[selected_columns]

    X_day = first_day_df.values
    return X_day, timestamps

def create_day_dataset_from_file():
    df = pd.read_parquet(infer_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    timestamps = df['timestamp'].reset_index(drop=True)  # Keep timestamps for later
    df = df.drop(columns=['timestamp'])

    # Apply normalization
    X_day = input_scaler.transform(df)
    return X_day, timestamps

def run_inference(X_day):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    if len(X_day.shape) == 2:
        X_day = np.expand_dims(X_day, axis=1)

    X_day_tensor = torch.tensor(X_day, dtype=torch.float32).to(device)
    day_dataset = TensorDataset(X_day_tensor)
    day_loader = DataLoader(day_dataset, batch_size=batch_size, shuffle=False)

    predictions_all = []
    with torch.no_grad():
        for (batch_X,) in day_loader:
            batch_X = batch_X.to(device)
            batch_predictions = model(batch_X)
            predictions_all.append(batch_predictions.cpu().numpy())

    predictions_np = np.concatenate(predictions_all, axis=0)
    return predictions_np


def melt_dataframe(df):
    # Melt the DataFrame to long format
    df_long = pd.melt(df,
                      id_vars=['timestamp'],   # Columns to keep
                      var_name='appliance',    # New column for appliance names
                      value_name='value')      # New column for values

    # Convert timestamp and extract date, hour, month
    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])
    df_long['date'] = df_long['timestamp'].dt.date
    df_long['minute'] = df_long['timestamp'].dt.minute
    df_long['hour'] = df_long['timestamp'].dt.hour
    df_long['month'] = df_long['timestamp'].dt.to_period('M')

    # Sort by timestamp (and optionally by appliance if you want consistent order)
    df_long = df_long.sort_values(by=['timestamp'])

    return df_long


def append_predictions(timestamps, predictions_np):
    # Combine timestamps and predictions into DataFrame
    pred_df = pd.DataFrame(predictions_np, columns=[f'{appliances_list[i]}' for i in range(predictions_np.shape[1])])
    # Inverse-transform predictions back to original appliance value ranges
    # pred_df[appliances_list] = target_scaler.inverse_transform(pred_df[appliances_list])
    for appliance in appliances_list:
        scaler = target_scaler[appliance]
        pred_df[appliance] = scaler.inverse_transform(pred_df[[appliance]])

    pred_df.insert(0, 'timestamp', timestamps)

    pred_df = melt_dataframe(pred_df)

    if os.path.exists(inferred_data_path):
        complete_df = pd.read_parquet(inferred_data_path)
        combined_df = pd.concat([complete_df, pred_df], ignore_index=True)
    else:
        combined_df = pred_df

    combined_df.to_parquet(inferred_data_path, index=False)
    print(f"[{datetime.now()}] Predictions appended to {inferred_data_path}")


def time_matches(target_time_str):
    now = datetime.now().time()
    target = datetime.strptime(target_time_str, "%H:%M").time()
    return now.hour == target.hour and now.minute == target.minute


if __name__ == "__main__":
    already_run_today = False
    print(f"Inference service started — waiting for {inference_timestamp} each day...")

    while True:
        now = datetime.now()

        if time_matches(inference_timestamp):
            if not already_run_today:
                print(f"[{now}] Time matched — running inference...")
                try:
                    X_day, timestamps = create_day_dataset_from_file()
                    predictions_np = run_inference(X_day)
                    append_predictions(timestamps, predictions_np)
                    already_run_today = True
                except Exception as e:
                    print(f"[{now}] Error during inference: {e}")
        else:
            already_run_today = False  # Reset the flag after target time passes

        time_module.sleep(1)  # Check every 30 seconds
