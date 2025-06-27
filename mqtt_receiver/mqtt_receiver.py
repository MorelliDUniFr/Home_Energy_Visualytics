import json
import paho.mqtt.client as mqtt
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timedelta
import threading
from config_loader import load_config

config, config_dir = load_config()

# Determine environment
env = config['Settings']['environment']
data_path = config[env]['data_path']
broker = config['MQTT']['broker']
port = int(config['MQTT']['port'])
topic = config['MQTT']['topic']
transfer_timestamp = config['MQTT']['transfer_timestamp']
daily_data_file = config['Data']['daily_data_file']
whole_data_file = config['Data']['whole_data_file']

# Calculate reset time (one minute after transfer time)
transfer_time = datetime.strptime(transfer_timestamp, "%H:%M")
reset_time = transfer_time + timedelta(minutes=1)
RESET_TIMESTAMP = reset_time.strftime("%H:%M")

# Ensure data directory exists
os.makedirs(data_path, exist_ok=True)

# Columns info
columns_to_drop = ['SMid', 'Po', 'P1o', 'P2o', 'P3o', 'Ei', 'Ei1', 'Ei2', 'Eo', 'Eo1', 'Eo2']

buffer = []  # Buffer for incoming data batches
already_written_today = False  # Flag for midnight whole file append


def append_to_parquet(df, parquet_path):
    """Append df to parquet_path by reading existing file and concatenating."""
    if os.path.exists(parquet_path):
        existing_table = pq.read_table(parquet_path)
        new_table = pa.Table.from_pandas(df=df)
        combined_table = pa.concat_tables([existing_table, new_table])
    else:
        combined_table = pa.Table.from_pandas(df=df)
    pq.write_table(combined_table, parquet_path)


def flush_buffer_to_daily_file():
    global buffer
    if not buffer:
        return
    batch_df = pd.concat(buffer, ignore_index=True)
    # Drop columns not needed in daily file
    batch_df_filtered = batch_df.drop(columns=columns_to_drop, errors='ignore')
    append_to_parquet(batch_df_filtered, os.path.join(data_path, daily_data_file))
    buffer.clear()
    print(f"Flushed {len(batch_df)} rows from buffer to daily file.")


def append_daily_to_whole():
    """Append entire daily parquet file to whole parquet file at midnight."""
    daily_path = os.path.join(data_path, daily_data_file)
    whole_path = os.path.join(data_path, whole_data_file)

    if not os.path.exists(daily_path):
        print("Daily file does not exist; nothing to append.")
        return

    try:
        daily_table = pq.read_table(daily_path)

        if os.path.exists(whole_path):
            whole_table = pq.read_table(whole_path)
            combined_table = pa.concat_tables([whole_table, daily_table])
        else:
            combined_table = daily_table

        pq.write_table(combined_table, whole_path)
        print("Appended daily file to whole file successfully.")

    except Exception as e:
        print(f"Error appending daily file to whole file: {e}")


def check_and_write_daily_data():
    global already_written_today
    print("Starting time-checking thread...")
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M")

        if current_time == transfer_timestamp and not already_written_today:
            print(f"{transfer_timestamp} reached — appending daily file to whole file.")
            flush_buffer_to_daily_file()  # flush any remaining data before appending
            append_daily_to_whole()
            already_written_today = True

        if current_time == RESET_TIMESTAMP:
            already_written_today = False

        time.sleep(1)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic)
    print(f"Subscribed to topic: {topic}")


def on_message(client, userdata, msg):
    global buffer
    try:
        nested_data = json.loads(msg.payload.decode())
        new_data = pd.json_normalize(nested_data)
        new_data.columns = new_data.columns.str.replace('z.', '', regex=False)

        if 'Time' in new_data.columns:
            new_data.rename(columns={'Time': 'timestamp'}, inplace=True)

        buffer.append(new_data)

        if len(buffer) >= 60:
            flush_buffer_to_daily_file()

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


# Initialize MQTT client
print("Initializing MQTT client...")
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port, 0)
print(f"Connected to broker {broker} on port {port}")

client.loop_start()

print("Transfer timestamp:", transfer_timestamp)
threading.Thread(target=check_and_write_daily_data, daemon=True).start()

while True:
    time.sleep(1)
