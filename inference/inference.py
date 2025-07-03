import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time as time_module
import os
from config_loader import load_config, logger
from joblib import load
import json
import pyarrow as pa
import pyarrow.parquet as pq

# Load configuration
config, config_dir = load_config()

env = config['Settings']['environment']
inference_timestamp = config['Inference']['inference_timestamp']
data_path = config[env]['data_path']
models_dir = str(config['Data']['models_dir'])
model_file = str(config['Data']['model_file'])
inferred_data_dir = 'inferred_data'  # new folder inside data_path for inference output
# infer_data_file = str(config['Data']['daily_data_file'])
infer_data_file = 'mqtt_data_daily.parquet'  # file to read for inference, can be changed to any other file
column_names_file = str(config['Data']['training_dataset_columns_file'])
scalers_dir = str(config['Data']['scalers_dir'])
input_scaler_file = str(config['Data']['input_scaler_file'])
target_scalers_file = str(config['Data']['target_scalers_file'])
batch_size = int(config['Inference']['batch_size'])

model_path = os.path.join(data_path, models_dir)
infer_data_path = os.path.join(data_path, infer_data_file)
input_scaler_path = os.path.join(data_path, input_scaler_file)
target_scalers_path = os.path.join(data_path, scalers_dir)
inferred_data_path = os.path.join(data_path, inferred_data_dir)

input_scaler = load(input_scaler_path)
device = torch.device('cpu')

# Read appliance names from JSON file (optional, you can get from model filenames too)
with open(os.path.join(data_path, column_names_file), 'r') as file:
    column_names_json = json.load(file)

# Create list of appliances from model filenames (removing suffix, underscores -> spaces)
appliances_list = [f.replace('.pt', '').rsplit('_', 1)[0].replace('_', ' ').title() for f in os.listdir(model_path)]


def prepare_inference_input(filename):
    df = pd.read_parquet(filename)

    drop_cols = ['SMid', 'Pi', 'Po', 'P1o', 'P2o', 'P3o', 'Ei', 'Ei1', 'Ei2', 'Eo', 'Eo1', 'Eo2']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    rename_map = {
        'P1i': 'powerl1', 'P2i': 'powerl2', 'P3i': 'powerl3',
        'I1': 'currentl1', 'I2': 'currentl2', 'I3': 'currentl3',
        'V1': 'voltagel1', 'V2': 'voltagel2', 'V3': 'voltagel3',
    }
    df = df.rename(columns=rename_map)

    ts = df.pop('timestamp').reset_index(drop=True)

    df = df[list(rename_map.values())]

    X = input_scaler.transform(df)
    return X, ts


def run_inference(day_input, app):
    appliance_name = app.lower().replace(' ', '_')
    model = torch.jit.load(os.path.join(model_path, appliance_name + model_file))
    model.eval()

    day_input = np.expand_dims(day_input, axis=0)
    day_input_tensor = torch.tensor(day_input, dtype=torch.float32).to(device)
    day_dataset = TensorDataset(day_input_tensor)
    day_loader = DataLoader(day_dataset, batch_size=batch_size, shuffle=False)

    predictions_all = []
    with torch.no_grad():
        for (batch_X,) in day_loader:
            batch_X = batch_X.to(device)
            batch_predictions = model(batch_X)

            batch_predictions = torch.clamp(batch_predictions, min=0.0)
            predictions_all.append(batch_predictions.cpu().numpy())

    return np.concatenate(predictions_all, axis=0)


def melt_dataframe(df):
    df_long = pd.melt(df,
                      id_vars=['timestamp'],
                      var_name='appliance',
                      value_name='value')
    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])
    df_long['date'] = df_long['timestamp'].dt.date
    df_long['minute'] = df_long['timestamp'].dt.minute
    df_long['hour'] = df_long['timestamp'].dt.hour
    df_long['month'] = df_long['timestamp'].dt.to_period('M')
    return df_long.sort_values(by=['timestamp'])


def compute_other_column(p_df, sm_df):
    p_df['timestamp'] = pd.to_datetime(p_df['timestamp'])
    sm_df['timestamp'] = pd.to_datetime(sm_df['timestamp'])

    p_df = p_df.sort_values('timestamp').reset_index(drop=True)
    sm_df = sm_df.sort_values('timestamp').reset_index(drop=True)

    appliance_cols = p_df.columns.difference(['timestamp'])
    p_df['total_pred_power'] = p_df[appliance_cols].sum(axis=1)

    phase_cols = [col for col in sm_df.columns if col.lower() in ['powerl1', 'powerl2', 'powerl3']]
    sm_df['total_sm_power'] = sm_df[phase_cols].sum(axis=1)

    merged = pd.merge(p_df, sm_df[['timestamp', 'total_sm_power']], on='timestamp', how='inner')

    merged['Other'] = (merged['total_sm_power'] - merged['total_pred_power']).clip(lower=0)

    result = merged.drop(columns=['total_pred_power', 'total_sm_power'])
    return result


def append_predictions(ts, predictions_dict):
    pred_df = pd.DataFrame({'timestamp': ts})

    for app, pred in predictions_dict.items():
        appliance_name = app.lower().replace(' ', '_')

        pred = np.squeeze(pred)
        target_scaler = load(os.path.join(target_scalers_path, appliance_name + target_scalers_file))
        pred_reshaped = pred.reshape(-1, 1)
        pred_inverse = target_scaler.inverse_transform(pred_reshaped).flatten()

        pred_df[app] = pred_inverse

    sm_df = pd.read_parquet(infer_data_path)
    pred_df = compute_other_column(pred_df, sm_df)
    pred_df = melt_dataframe(pred_df)

    # Ensure 'date' column is datetime.date (not string)
    pred_df['date'] = pd.to_datetime(pred_df['date']).dt.date

    # Generate folder name from the first date (as string)
    first_date = pred_df['date'].iloc[0]
    partition_folder = os.path.join(inferred_data_path, f'date={first_date.strftime("%Y-%m-%d")}')
    os.makedirs(partition_folder, exist_ok=True)
    file_path = os.path.join(partition_folder, 'predictions.parquet')

    # Merge with existing data if exists
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_parquet(file_path)
            pred_df = pd.concat([existing_df, pred_df], ignore_index=True)
        except Exception as ex:
            logger.error(f"Error loading existing inference file: {ex}")

    # Save data
    pred_df.to_parquet(file_path, index=False)
    logger.info(f"[{datetime.now()}] Predictions appended to {file_path}")



def time_matches(target_time_str):
    return datetime.now().strftime("%H:%M") == target_time_str


if __name__ == "__main__":
    already_run_today = False
    logger.info(f"Inference service started — waiting for {inference_timestamp} each day...")

    while True:
        now = datetime.now()
        if time_matches(inference_timestamp):
            if not already_run_today:
                logger.info(f"[{now}] Time matched — running inference...")
                try:
                    input_data, timestamps = prepare_inference_input(infer_data_path)
                    predictions = {}
                    for appliance in appliances_list:
                        predictions_np = run_inference(input_data, app=appliance)
                        predictions[appliance] = predictions_np

                    append_predictions(timestamps, predictions)
                    already_run_today = True
                except Exception as e:
                    logger.error(f"[{now}] Error during inference: {e}")
        else:
            already_run_today = False

        time_module.sleep(1)
