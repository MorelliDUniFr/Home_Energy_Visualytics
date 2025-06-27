import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time as time_module
import os
from config_loader import load_config
from joblib import load
import json

config, config_dir = load_config()

env = config['Settings']['environment']
inference_timestamp = config['Inference']['inference_timestamp']
data_path = config[env]['data_path']
models_dir = str(config['Data']['models_dir'])
model_file = str(config['Data']['model_file'])
inferred_data_file = str(config['Data']['inferred_data_file'])
infer_data_file = str(config['Data']['daily_data_file'])
column_names_file = str(config['Data']['training_dataset_columns_file'])
scalers_dir = str(config['Data']['scalers_dir'])
input_scaler_file = str(config['Data']['input_scaler_file'])
target_scalers_file = str(config['Data']['target_scalers_file'])
batch_size = int(config['Inference']['batch_size'])

model_path = os.path.join(data_path, models_dir)
inferred_data_path = os.path.join(data_path, inferred_data_file)
infer_data_path = os.path.join(data_path, infer_data_file)
input_scaler_path = os.path.join(data_path, input_scaler_file)
target_scalers_path = os.path.join(data_path, scalers_dir)

input_scaler = load(input_scaler_path)

device = torch.device('cpu')

# Read appliance names
with open(os.path.join(data_path, column_names_file), 'r') as file:
    column_names_json = json.load(file)

# Create the list of appliances dynamically from the model files
appliances_list = [f.replace('.pt', '').rsplit('_', 1)[0].replace('_', ' ').title() for f in os.listdir(model_path)]

def create_day_dataset_from_file():
    df = pd.read_parquet(infer_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    timestamps = df['timestamp'].reset_index(drop=True)  # Keep timestamps for later
    df = df.drop(columns=['timestamp'])

    # Apply normalization
    X_day = input_scaler.transform(df)
    return X_day, timestamps

def run_inference(X_day, appliance):
    appliance_name = appliance.lower().replace(' ', '_')
    model = torch.jit.load(os.path.join(model_path, appliance_name + model_file))
    model.eval()

    if len(X_day.shape) == 2:
        X_day = np.expand_dims(X_day, axis=0)

    X_day_tensor = torch.tensor(X_day, dtype=torch.float32).to(device)
    day_dataset = TensorDataset(X_day_tensor)
    day_loader = DataLoader(day_dataset, batch_size=batch_size, shuffle=False)

    predictions_all = []
    with torch.no_grad():
        for (batch_X,) in day_loader:
            batch_X = batch_X.to(device)
            batch_predictions = model(batch_X)

            # Clamp predictions to be non-negative (power >= 0)
            batch_predictions = torch.clamp(batch_predictions, min=0.0)

            # Convert to NumPy after clamping
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


def compute_other_column(p_df, sm_df):
    # Ensure timestamps are datetime and sorted
    p_df['timestamp'] = pd.to_datetime(p_df['timestamp'])
    sm_df['timestamp'] = pd.to_datetime(sm_df['timestamp'])

    p_df = p_df.sort_values('timestamp').reset_index(drop=True)
    sm_df = sm_df.sort_values('timestamp').reset_index(drop=True)

    # Sum predicted appliance power per timestamp (exclude 'timestamp' column)
    appliance_cols = p_df.columns.difference(['timestamp'])
    p_df['total_pred_power'] = p_df[appliance_cols].sum(axis=1)

    # Sum smart meter phases to get total power per timestamp
    phase_cols = [col for col in sm_df.columns if col.lower() in ['powerl1', 'powerl2', 'powerl3']]
    sm_df['total_sm_power'] = sm_df[phase_cols].sum(axis=1)

    # Merge on timestamp to align rows
    merged = pd.merge(p_df, sm_df[['timestamp', 'total_sm_power']], on='timestamp', how='inner')

    # Compute 'Other' = smart meter total - sum predicted appliances
    merged['Other'] = merged['total_sm_power'] - merged['total_pred_power']

    # Clip negative values to zero
    merged['Other'] = merged['Other'].clip(lower=0)

    # Optional: keep original columns + Other column, drop helper cols
    result = merged.drop(columns=['total_pred_power', 'total_sm_power'])

    return result


def append_predictions(timestamps, predictions_dict):
    """
    timestamps: list of timestamps (len = total timesteps)
    predictions_dict: dict of {appliance_name: np.ndarray of shape (total_timesteps,)}
                      or (1, seq_len, 1) / (batch, seq_len, 1)
    """
    pred_df = pd.DataFrame({'timestamp': timestamps})

    for appliance, pred in predictions_dict.items():
        appliance_name = appliance.lower().replace(' ', '_')

        # Remove batch dimension if necessary
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]  # shape: (seq_len, 1)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred[:, 0]  # shape: (seq_len,)
        elif pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[2])[:, 0]  # flatten and squeeze

        # Inverse scale
        target_scaler = load(os.path.join(target_scalers_path, appliance_name + target_scalers_file))
        pred_reshaped = pred.reshape(-1, 1)
        pred_inverse = target_scaler.inverse_transform(pred_reshaped).flatten()

        pred_df[appliance] = pred_inverse

    # Compute 'Other' column
    sm_df = pd.read_parquet(infer_data_path)
    pred_df = compute_other_column(pred_df, sm_df)

    # Melt and save
    pred_df = melt_dataframe(pred_df)

    # Load previous data if exists
    if os.path.exists(inferred_data_path):
        try:
            existing_df = pd.read_parquet(inferred_data_path)
            pred_df = pd.concat([existing_df, pred_df], ignore_index=True)
        except Exception as e:
            print(f"Error loading existing file: {e}")
            # You can choose to fail or continue with only new predictions

    pred_df.to_parquet(inferred_data_path, index=False)
    print(f"[{datetime.now()}] Predictions for appliances {list(predictions_dict.keys())} appended to {inferred_data_path}")


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
                    predictions = {}
                    for appliance in appliances_list:
                        predictions_np = run_inference(X_day, appliance=appliance)
                        predictions[appliance] = predictions_np

                    append_predictions(timestamps, predictions)

                    already_run_today = True
                except Exception as e:
                    print(f"[{now}] Error during inference: {e}")
        else:
            already_run_today = False  # Reset the flag after target time passes

        time_module.sleep(30)
