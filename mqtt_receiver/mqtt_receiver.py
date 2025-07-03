import json
import paho.mqtt.client as mqtt
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timedelta
import threading
from config_loader import load_config, logger
import shutil  # for file copy

config, config_dir = load_config()

env = config['Settings']['environment']
data_path = config[env]['data_path']
broker = config['MQTT']['broker']
port = int(config['MQTT']['port'])
topic = config['MQTT']['topic']
transfer_timestamp = config['MQTT']['transfer_timestamp']
daily_data_file = config['Data']['daily_data_file']  # e.g. 'daily.parquet'
whole_data_dir = config['Data']['whole_data_dir']  # change to directory path for whole dataset partitions

# Calculate reset time (one minute after transfer time)
transfer_time = datetime.strptime(transfer_timestamp, "%H:%M")
reset_time = transfer_time + timedelta(minutes=1)
RESET_TIMESTAMP = reset_time.strftime("%H:%M")

os.makedirs(data_path, exist_ok=True)
os.makedirs(os.path.join(data_path, whole_data_dir), exist_ok=True)

columns_to_drop = ['SMid', 'Po', 'P1o', 'P2o', 'P3o', 'Ei', 'Ei1', 'Ei2', 'Eo', 'Eo1', 'Eo2']

buffer = []
already_copied_today = False


def write_daily_file(df):
    """Overwrite daily file with the given DataFrame."""
    daily_path = os.path.join(data_path, daily_data_file)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, daily_path)
    logger.info(f"Daily file written with {len(df)} rows.")


def flush_buffer_to_daily_file():
    global buffer
    if not buffer:
        return
    batch_df = pd.concat(buffer, ignore_index=True)
    batch_df_filtered = batch_df.drop(columns=columns_to_drop, errors='ignore')
    write_daily_file(batch_df_filtered)
    buffer.clear()


def copy_daily_to_partitioned_whole():
    """
    Copy daily file to whole dataset directory with date partitioning,
    using the date from the data timestamps to name the partition folder.
    """
    daily_path = os.path.join(data_path, daily_data_file)
    if not os.path.exists(daily_path):
        logger.info("Daily file does not exist; skipping whole data copy.")
        return

    try:
        df = pd.read_parquet(daily_path)
        if 'timestamp' not in df.columns:
            logger.error("No 'timestamp' column found in daily data file.")
            return

        # Extract the date from the first timestamp (assumes all data is from one day)
        data_date = pd.to_datetime(df['timestamp'].iloc[0]).date()
        date_str = data_date.strftime("%Y-%m-%d")

        partition_folder = os.path.join(data_path, whole_data_dir, f"date={date_str}")
        os.makedirs(partition_folder, exist_ok=True)

        target_path = os.path.join(partition_folder, daily_data_file)

        shutil.copy2(daily_path, target_path)
        logger.info(f"Copied daily file to whole dataset partition: {target_path}")

    except Exception as e:
        logger.error(f"Error copying daily file to whole dataset: {e}")


def check_and_manage_daily_files():
    global already_copied_today
    logger.info("Starting daily file management thread...")
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M")

        if current_time == transfer_timestamp and not already_copied_today:
            logger.info(f"{transfer_timestamp} reached — flushing buffer and copying daily file.")
            flush_buffer_to_daily_file()
            copy_daily_to_partitioned_whole()
            already_copied_today = True

        if current_time == RESET_TIMESTAMP:
            already_copied_today = False

        time.sleep(1)


def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected with result code {rc}")
    client.subscribe(topic)
    logger.info(f"Subscribed to topic: {topic}")


def on_message(client, userdata, msg):
    global buffer
    try:
        nested_data = json.loads(msg.payload.decode())
        new_data = pd.json_normalize(nested_data)
        new_data.columns = new_data.columns.str.replace('z.', '', regex=False)

        if 'Time' in new_data.columns:
            new_data.rename(columns={'Time': 'timestamp'}, inplace=True)

        buffer.append(new_data)

        # Flush buffer if it reaches 60 entries (about 10 minutes if messages every 10s)
        if len(buffer) >= 60:
            flush_buffer_to_daily_file()

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")


# --- Main setup ---
logger.info("Initializing MQTT client...")
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port, 0)
logger.info(f"Connected to broker {broker} on port {port}")

client.loop_start()

logger.info(f"Transfer timestamp: {transfer_timestamp}")
threading.Thread(target=check_and_manage_daily_files, daemon=True).start()

while True:
    time.sleep(1)
