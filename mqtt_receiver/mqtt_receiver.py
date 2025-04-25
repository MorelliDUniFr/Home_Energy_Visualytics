import json
import paho.mqtt.client as mqtt
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timedelta
import threading

# MQTT Broker Settings
BROKER = "192.168.1.189"  # Use your broker's address
PORT = 1883  # Default MQTT port
TOPIC = "tele/gPlugDI_1E533C/SENSOR"

TRANSFER_TIMESTAMP = "16:40:00"
# Convert the string timestamp to a datetime object
transfer_time = datetime.strptime(TRANSFER_TIMESTAMP, "%H:%M:%S")
# Add 2 seconds to the transfer timestamp
reset_time = transfer_time + timedelta(seconds=2)
# Format the reset timestamp back to the string format
RESET_TIMESTAMP = reset_time.strftime("%H:%M:%S")

DATA_DIR = "/app/data"
DAY_PARQUET_FILE = os.path.join(DATA_DIR, "mqtt_data_day.parquet")
WHOLE_PARQUET_FILE = os.path.join(DATA_DIR, "mqtt_data_whole.parquet")

# Ensure the directory exists
if not os.path.exists(DATA_DIR):
    print("Creating data directory...")
    os.makedirs(DATA_DIR)

# Prepare an empty DataFrame for the Parquet file (initially empty)
columns_to_keep = ["timestamp", "Pi", "P1i", "P2i", "P3i",
           "V1", "V2", "V3", "I1", "I2", "I3"]
columns_to_drop = ['SMid', 'Po', 'P1o', 'P2o', 'P3o', 'Ei', 'Ei1', 'Ei2', 'Eo', 'Eo1', 'Eo2']
columns = ["timestamp", 'SMid', 'Pi', 'Po', 'P1i', 'P1o', 'P2i', 'P2o', 'P3i', 'P3o',
           'V1', 'V2', 'V3', 'I1', 'I2', 'I3', 'Ei', 'Ei1', 'Ei2', 'Eo', 'Eo1', 'Eo2']
raw_df = pd.DataFrame(columns=columns)

def append_to_parquet(df, parquet_path):
    print("Appending data to parquet file...")
    if os.path.exists(parquet_path):
        # Read the existing Parquet data
        existing_table = pq.read_table(parquet_path)
        # Convert DataFrame to Apache Arrow Table
        new_table = pa.Table.from_pandas(df=df)
        # Combine old and new data
        combined_table = pa.concat_tables([existing_table, new_table])
    else:
        # Convert DataFrame to Apache Arrow Table
        combined_table = pa.Table.from_pandas(df=df)

    # Write the combined table back to Parquet
    pq.write_table(combined_table, parquet_path)


def write_new_parquet_file(df, parquet_path):
    # Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(df=df)

    table = table.to_pandas()
    # Drop unnecessary columns
    table = table.drop(columns=columns_to_drop,
                             errors='ignore')  # 'ignore' prevents errors if column does not exist
    table_day = pa.Table.from_pandas(table)

    # Write the combined table back to Parquet
    pq.write_table(table_day, parquet_path)


# Function to write data to Parquet file asynchronously
def write_data_to_parquet():
    global raw_df
    try:
        if raw_df.empty:
            print("No new data to write.")
            return

        # Append to the whole data file
        append_to_parquet(raw_df, WHOLE_PARQUET_FILE)

        # Write the data to the daily Parquet file
        write_new_parquet_file(raw_df, DAY_PARQUET_FILE)

        # Clear the DataFrame after writing
        raw_df = pd.DataFrame(columns=columns)

        print("Data successfully written to Parquet.")

    except Exception as e:
        print(f"Error while writing data: {e}")


# Function to check if it's midnight and trigger the writing process
def check_and_write_daily_data():
    global raw_df
    print("Starting the time checking thread...")  # Debugging log
    already_written_today = False

    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # Change the time to match when you want to trigger the action
        if current_time == TRANSFER_TIMESTAMP and not already_written_today and not raw_df.empty:  # Ensure time format matches the expected one
            print("Writing data for yesterday...")
            # Run the write function in a separate thread to avoid blocking the MQTT loop
            threading.Thread(target=write_data_to_parquet).start()
            already_written_today = True

        # Reset the flag after midnight
        if current_time == RESET_TIMESTAMP:
            already_written_today = False

        time.sleep(1)  # Check every second


# Callback when connected to the broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # Subscribe to the topic
    client.subscribe(TOPIC)
    print("Subscribed to topic:", TOPIC)


def on_message(client, userdata, msg):
    global raw_df

    try:
        # Load the JSON into a Python dictionary
        nested_data = json.loads(msg.payload.decode())

        # Normalize the nested JSON structure
        new_data = pd.json_normalize(nested_data)

        # Remove 'z.' prefix from column names
        new_data.columns = new_data.columns.str.replace('z.', '', regex=False)

        # Rename the "Time" column to "timestamp"
        if 'Time' in new_data.columns:
            new_data.rename(columns={'Time': 'timestamp'}, inplace=True)

        # Concatenate the new data with the existing data
        raw_df = pd.concat([raw_df, new_data], ignore_index=True)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Initialize MQTT Client
print("Initializing MQTT Client...")
client = mqtt.Client()
print("Setting up callbacks...")
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker and subscribe to the topic
client.connect(BROKER, PORT, 0)
print(f"Connected to broker {BROKER} on port {PORT}")

# Start the MQTT loop in a separate thread to allow the main program to continue
client.loop_start()

# Start the midnight check in a separate thread
threading.Thread(target=check_and_write_daily_data, daemon=True).start()

# Run the main thread (blocking the program)
while True:
    time.sleep(1)  # Main thread keeps running
