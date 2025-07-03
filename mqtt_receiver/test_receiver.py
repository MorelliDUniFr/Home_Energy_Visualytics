import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
from datetime import datetime, timedelta
import logging

# --- Mock logger (replace with your logger in real use) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (adjust paths as needed) ---
data_path = "../data"
daily_data_file = "daily.parquet"
whole_data_path = os.path.join(data_path, "whole_dataset")

os.makedirs(data_path, exist_ok=True)
os.makedirs(whole_data_path, exist_ok=True)

buffer = []  # global buffer

# --- Flush buffer to daily parquet file ---
def flush_buffer_to_daily_file():
    global buffer
    if not buffer:
        logger.info("Buffer empty, nothing to flush.")
        return
    batch_df = pd.concat(buffer, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(batch_df), os.path.join(data_path, daily_data_file))
    buffer.clear()
    logger.info(f"Flushed {len(batch_df)} rows from buffer to daily file.")

# --- Copy daily file to partitioned whole dataset (simulate partition by copying file) ---
def copy_daily_to_partitioned_whole(date):
    src = os.path.join(data_path, daily_data_file)
    if not os.path.exists(src):
        logger.error("Daily file does not exist, cannot copy.")
        return
    # Format partition folder using the passed date argument
    partition_folder = os.path.join(whole_data_path, date.strftime("date=%Y-%m-%d"))
    os.makedirs(partition_folder, exist_ok=True)
    dest = os.path.join(partition_folder, daily_data_file)
    shutil.copy2(src, dest)
    logger.info(f"Copied daily file to partitioned dataset at {dest}")


if __name__ == "__main__":
    # Starting datetime (e.g., midnight of the given day)
    day = datetime(2025, 7, 1)  # Replace with your desired date and time at midnight

    # Generate 8640 timestamps spaced by 10 seconds (8640 * 10s = 1 day)
    timestamps = [day + timedelta(seconds=10 * i) for i in range(8640)]

    dummy_data = pd.DataFrame({
        'timestamp': timestamps,
        'Power': range(100, 100 + 8640),  # Sample power values (can adjust as needed)
        'OtherColumn': range(1, 8641)  # Another sample column
    })

    # Add dummy data to buffer
    buffer.append(dummy_data)

    # Flush and copy
    flush_buffer_to_daily_file()
    copy_daily_to_partitioned_whole(day)
