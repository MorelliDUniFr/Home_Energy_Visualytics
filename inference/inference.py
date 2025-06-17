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
models_dir = config['Data']['models_dir']
model_file = config['Data']['model_file']
inferred_data_file = config['Data']['inferred_data_file']
infer_data_file = config['Data']['infer_data_file']
column_names_file = config['Data']['training_dataset_columns_file']
scalers_dir = config['Data']['scalers_dir']
input_scaler_file = config['Data']['input_scaler_file']
target_scalers_file = config['Data']['target_scalers_file']
batch_size = int(config['Inference']['batch_size'])

model_path = os.path.join(data_path, models_dir)
inferred_data_path = os.path.join(data_path, inferred_data_file)
infer_data_path = os.path.join(data_path, infer_data_file)
input_scaler_path = os.path.join(data_path, input_scaler_file)
target_scalers_path = os.path.join(data_path, scalers_dir)

input_scaler = load(input_scaler_path)

device = torch.device('cpu')

# Read appliance names from the text file
with open(os.path.join(data_path, column_names_file), 'r') as file:
    column_names_json = json.load(file)

appliances_list = column_names_json['appliances']

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


def append_predictions(timestamps, predictions_dict):
    """
    timestamps: list of timestamps (len = total timesteps)
    predictions_dict: dict of {appliance_name: np.ndarray of shape (total_timesteps,)}
                      or (1, seq_len, 1) / (batch, seq_len, 1)
    """
    pred_df = pd.DataFrame({'timestamp': timestamps})

    for appliance, pred in predictions_dict.items():
        print(f"Processing {appliance} - shape before reshape:", pred.shape)
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

    # Melt and save
    pred_df = melt_dataframe(pred_df)
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
