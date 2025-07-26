import streamlit as st
import pyarrow.dataset as ds
from datetime import date, datetime, timedelta
import pandas as pd
import os
import re

# Local utility imports
from .filters import filter_appliances_with_nonzero_sum, time_filter


@st.cache_data(show_spinner=False)
def load_inferred_data_partitioned(dataset_path: str, start_date: str, end_date: str, fingerprint: float) -> pd.DataFrame:
    """
    Loads data from a Hive-partitioned Parquet dataset between start_date and end_date.
    Uses caching with a custom fingerprint to avoid redundant I/O.

    Args:
        dataset_path (str): Path to the dataset directory.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        fingerprint (float): A value to bust cache when dataset is updated.

    Returns:
        pd.DataFrame: Filtered dataset for the given date range.
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
    Loads and filters data for a given time period using session_state for inputs.

    Args:
        dataset_path (str): Path to dataset.
        period (str): One of 'Day', 'Week', 'Month', 'Year'.
        suffix (str): Identifier for distinguishing state keys (e.g., '_1').

    Returns:
        Tuple[date, pd.DataFrame]: Start date and filtered DataFrame.
    """
    # Select start and end dates based on period
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
        # Fallback to yesterday if unknown period
        start_date = end_date = date.today() - timedelta(days=1)

    # Format dates
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Cache key fingerprint based on latest dataset change
    fingerprint = get_dataset_fingerprint(dataset_path)

    # Load and filter
    df = load_inferred_data_partitioned(dataset_path, start_str, end_str, fingerprint)
    df_filtered = filter_appliances_with_nonzero_sum(df)

    return start_date, df_filtered


@st.cache_data
def filter_data_by_time(f_data: pd.DataFrame, t_filter: str, selected_day: str) -> pd.DataFrame:
    """
    Filters data within a specific time window based on the selected filter type (e.g. Day, Week).

    Args:
        f_data (pd.DataFrame): Full dataset.
        t_filter (str): One of the keys from `time_filter` dict (e.g. 'Day').
        selected_day (str): Reference start day in datetime.date format.

    Returns:
        pd.DataFrame: Filtered DataFrame limited to the selected date range and active appliances.
    """
    filtered_data = f_data[
        f_data['date'].between(selected_day, selected_day + timedelta(days=time_filter[t_filter]["timedelta_days"]))
    ]
    return filter_appliances_with_nonzero_sum(filtered_data)


def get_earliest_date(dataset_path: str):
    """
    Scans the dataset partition folder names and returns the earliest available date.

    Args:
        dataset_path (str): Root directory containing 'date=YYYY-MM-DD' subfolders.

    Returns:
        datetime.date or None: Earliest date found or None if no valid folders.
    """
    dates = []
    for foldername in os.listdir(dataset_path):
        match = re.match(r"date=(\d{4})-(\d{2})-(\d{2})", foldername)
        if match:
            y, m, d = map(int, match.groups())
            dates.append(date(y, m, d))
    return min(dates) if dates else None


def load_data_by_date_range(dataset_path: str, start_date: date, end_date: date):
    """
    Loads data between two dates and filters out inactive appliances.

    Args:
        dataset_path (str): Path to partitioned dataset.
        start_date (date): Start date (inclusive).
        end_date (date): End date (inclusive).

    Returns:
        pd.DataFrame: Filtered dataset within the specified range.
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    fingerprint = get_dataset_fingerprint(dataset_path)
    df = load_inferred_data_partitioned(dataset_path, start_str, end_str, fingerprint)
    return filter_appliances_with_nonzero_sum(df)


def get_dataset_fingerprint(dataset_path: str) -> float:
    """
    Returns the latest modification time found within the dataset directory tree.

    Args:
        dataset_path (str): Path to the root of the dataset directory.

    Returns:
        float: Most recent modification timestamp.
    """
    latest_mtime = 0
    for root, dirs, files in os.walk(dataset_path):
        for name in dirs + files:
            try:
                full_path = os.path.join(root, name)
                mtime = os.path.getmtime(full_path)
                latest_mtime = max(latest_mtime, mtime)
            except FileNotFoundError:
                pass  # Can occur if a file is removed during walk
    return latest_mtime
