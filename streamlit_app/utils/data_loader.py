import streamlit as st
import pyarrow.dataset as ds
from datetime import date, datetime, timedelta
import pandas as pd
import os
import re
from .filters import filter_appliances_with_nonzero_sum, time_filter

@st.cache_data(show_spinner=False)
def load_inferred_data_partitioned(dataset_path: str, start_date: str, end_date: str, fingerprint: float) -> pd.DataFrame:
    """
    Load parquet dataset filtered by date partitions between start_date and end_date (inclusive).
    The cache is invalidated whenever the fingerprint changes.
    """
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    date_col = ds.field("date")
    date_filter = (date_col >= start.strftime("%Y-%m-%d")) & (date_col <= end.strftime("%Y-%m-%d"))

    scanner = dataset.scanner(filter=date_filter)
    table = scanner.to_table()
    df = table.to_pandas()

    return df


def load_and_filter_data_by_time(dataset_path: str, period: str, suffix: str):
    """
    period: 'Day', 'Week', 'Month', 'Year'
    suffix: string like '_1', '_2' to access session_state keys
    """
    # Get selected values from session_state with fallback
    if period == "Day":
        selected_date = st.session_state.get(f"selected_date{suffix}", date.today() - timedelta(days=1))
        start_date = end_date = selected_date
    elif period == "Week":
        selected_date = st.session_state.get(f"selected_date{suffix}", date.today() - timedelta(weeks=1))
        start_date = selected_date
        end_date = start_date + timedelta(days=6)
    elif period == "Month":
        selected_date = st.session_state.get(f"selected_date{suffix}", date.today() - timedelta(days=30))
        start_date = selected_date
        end_date = start_date + timedelta(days=29)
    elif period == "Year":
        selected_year = st.session_state.get(f"selected_year{suffix}", date.today().year - 1)
        start_date = date(selected_year, 1, 1)
        end_date = date(selected_year, 12, 31)
    else:
        # Default fallback, load yesterday
        start_date = end_date = date.today() - timedelta(days=1)

    # Format dates as strings
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    fingerprint = get_dataset_fingerprint(dataset_path)

    df = load_inferred_data_partitioned(dataset_path, start_str, end_str, fingerprint)
    df_filtered = filter_appliances_with_nonzero_sum(df)

    return start_date, df_filtered


@st.cache_data
def filter_data_by_time(f_data: pd.DataFrame, t_filter: str, selected_day: str) -> pd.DataFrame:
    """Filters data based on selected time range and removes appliances with zero totals."""
    filtered_data = f_data[
        f_data['date'].between(selected_day, selected_day + timedelta(days=time_filter[t_filter]["timedelta_days"]))
    ]

    # Filter out appliances with total zero
    filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

    return filtered_data


def get_earliest_date(dataset_path: str):
    # Scan dataset folder names like 'date=YYYY-MM-DD'
    dates = []
    for foldername in os.listdir(dataset_path):
        match = re.match(r"date=(\d{4})-(\d{2})-(\d{2})", foldername)
        if match:
            y, m, d = map(int, match.groups())
            dates.append(date(y, m, d))
    if dates:
        return min(dates)
    else:
        return None  # or some default date


def load_data_by_date_range(dataset_path: str, start_date: date, end_date: date):
    """
    Load data from the partitioned dataset between start_date and end_date (inclusive),
    and filter out appliances with zero total consumption.

    Parameters:
    - dataset_path: path to the partitioned dataset
    - start_date: datetime.date object representing the start date
    - end_date: datetime.date object representing the end date

    Returns:
    - filtered DataFrame with data between the dates and appliances with non-zero consumption
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    fingerprint = get_dataset_fingerprint(dataset_path)
    df = load_inferred_data_partitioned(dataset_path, start_str, end_str, fingerprint)
    df_filtered = filter_appliances_with_nonzero_sum(df)

    return df_filtered


def get_dataset_fingerprint(dataset_path: str) -> float:
    """
    Returns the latest modification time of any file or folder inside the dataset directory.
    """
    latest_mtime = 0
    for root, dirs, files in os.walk(dataset_path):
        for name in dirs + files:
            try:
                full_path = os.path.join(root, name)
                mtime = os.path.getmtime(full_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
            except FileNotFoundError:
                pass  # Might happen during folder writes
    return latest_mtime
