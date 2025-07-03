import os
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date, timedelta, datetime
from config_loader import load_config
import pyarrow.dataset as ds
import re

config, config_dir = load_config()

env = config['Settings']['environment']
data_path = str(config[env]['data_path'])
appliances_colors_file = config['Data']['appliances_colors_file']
# inferred_data_file = config['Data']['inferred_data_file']
# inferred_data_file = 'demo_inferred_data_2024.parquet'
inferred_data_file = 'real_inferred_whole.parquet'
models_dir = str(config['Data']['models_dir'])
scalers_dir = str(config['Data']['scalers_dir'])
model_file = str(config['Data']['model_file'])
target_scalers_file = str(config['Data']['target_scalers_file'])
inferred_dataset_path = '../data/inferred_data'

# Define a color palette
color_palette = px.colors.qualitative.Pastel

# ---------------------------
# Constants
# ---------------------------
DATE_FORMAT = "DD.MM.YYYY"

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def load_inferred_data_partitioned(dataset_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load parquet dataset filtered by date partitions between start_date and end_date (inclusive).
    Dates are strings in 'YYYY-MM-DD' format.
    """
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    # Create filter expressions on partition column 'date'
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Build filter: date >= start AND date <= end
    date_col = ds.field("date")
    date_filter = (date_col >= start) & (date_col <= end)

    table = dataset.to_table(filter=date_filter)
    df = table.to_pandas()
    return df


def get_date_ranges():
    """Returns a dictionary with common date ranges."""
    today = date.today()

    return {
        "today": today,
        "yesterday": today - timedelta(days=1),
        "another_yesterday": today - timedelta(days=2),
        "last_week": today - timedelta(weeks=1),
        "last_month": today - timedelta(days=30),
        "last_year": date(today.year - 1, today.month, 1),
    }

def safe_load_json(filename):
    """Safely loads a JSON file into a dictionary."""
    print("Loading data from JSON file...")
    if os.path.exists(filename):
        return pd.read_json(filename, typ="series").to_dict()
    else:
        st.error(f"Missing file: {filename}")
        return {}

# ---------------------------
# Load Data
# ---------------------------
date_ranges = get_date_ranges()

def format_appliance_name(filename: str) -> str:
    base = filename.split('.')[0].rsplit('_', 1)[0]
    parts = base.split('_')
    return ' '.join([
        word.upper() if len(word) <= 2 and word.islower() else word.title()
        for word in parts
    ])


def read_appliances_from_files():
    model_files = os.listdir(os.path.join(data_path, models_dir))
    scaler_files = os.listdir(os.path.join(data_path, scalers_dir))

    appliances_names = [format_appliance_name(f) for f in model_files]

    appliances_names.sort(key=lambda x: (x == 'Other', x))

    return appliances_names


def generate_appliance_colors(names):
    colors = {}
    # Assign colors using the original order
    for i, app in enumerate(names):
        if app == 'Other':
            colors[app] = 'rgb(153, 153, 153)'
        else:
            colors[app] = color_palette[i % len(color_palette)]

    return colors


if os.path.exists(os.path.join(data_path, appliances_colors_file)):
    appliance_colors = safe_load_json(os.path.join(data_path, appliances_colors_file))
else:
    appliance_names = read_appliances_from_files()
    appliance_colors = generate_appliance_colors(appliance_names)
    # Save the colors to a JSON file
    with open(os.path.join(data_path, appliances_colors_file), 'w') as f:
        pd.Series(appliance_colors).to_json(f, indent=4)

parquet_file = os.path.join(data_path, inferred_data_file)
mtime = os.path.getmtime(parquet_file)
# data = safe_load_parquet(parquet_file, mtime=mtime)

# Sort appliances
appliance_order = list(appliance_colors.keys())


# ---------------------------
# Filters
# ---------------------------
time_filter = {
    "Day": {
        "tick_format": "%H:%M",
        "chart_x_value": "timestamp",
        "input_string": 'select_day',
        "max_value": date_ranges["yesterday"],
        "value": date_ranges["yesterday"],
        "timedelta_days": 0,
        "group_by_column": ["hour", "minute"],
        "time_period": "minute",
        "date_format": "%d %B %Y",
        "floor_period": "min",
    },
    "Week": {
        "tick_format": "%d %b %H:%M",
        "chart_x_value": "date",
        "input_string": 'select_starting_day',
        "max_value": date_ranges["last_week"],
        "value": date_ranges["last_week"],
        "timedelta_days": 6,
        "group_by_column": "date",
        "time_period": None,
        "date_format": "%d %B %Y",
        "floor_period": "h",
    },
    "Month": {
        "tick_format": "%d %b",
        "chart_x_value": "date",
        "input_string": 'select_starting_day',
        "max_value": date_ranges["last_month"],
        "value": date_ranges["last_month"],
        "timedelta_days": 29,
        "group_by_column": "date",
        "time_period": None,
        "date_format": "%B %Y",
        "floor_period": "D",
    },
    "Year": {
        "tick_format": "%d %b %Y",
        "chart_x_value": "date",
        "input_string": 'select_starting_month',
        "max_value": date_ranges["last_year"],
        "value": date_ranges["last_year"],
        "timedelta_days": 364,
        "group_by_column": "month",
        "time_period": "month",
        "date_format": "%Y",
        "floor_period": "D",
    },
}

# ---------------------------
# Data Processing Functions
# ---------------------------
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

    # Load filtered data from partitioned parquet dataset
    df = load_inferred_data_partitioned(dataset_path, start_str, end_str)

    # Now filter out appliances with zero sum as before
    df_filtered = filter_appliances_with_nonzero_sum(df)

    return start_date, df_filtered


@st.cache_data
def filter_data_by_time(f_data: pd.DataFrame, t_filter: str, selected_day: date) -> pd.DataFrame:
    """Filters data based on selected time range and removes appliances with zero total."""
    filtered_data = f_data[
        f_data['date'].between(selected_day, selected_day + timedelta(days=time_filter[t_filter]["timedelta_days"]))
    ]

    # Filter out appliances with total zero
    filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

    return filtered_data

def aggregate_data(f_data: pd.DataFrame, group_by_columns: list, date_column: str, time_period: str = None) -> pd.DataFrame:
    """Aggregates data by summing values and formatting time-based columns."""
    f_data = f_data.copy()  # Avoid modifying the original DataFrame

    # Ensure the date column is in datetime format
    f_data[date_column] = pd.to_datetime(f_data[date_column], errors='coerce')
    f_data.dropna(subset=[date_column], inplace=True)

    # Flatten group_by_columns
    group_by_columns = [item for sublist in group_by_columns for item in (sublist if isinstance(sublist, list) else [sublist])]

    # Group and reset index
    f_data = f_data.groupby(group_by_columns, as_index=False)['value'].mean()

    # Convert time-related columns if necessary
    if time_period == 'minute' and 'minute' in f_data.columns:
        f_data["timestamp"] = pd.to_datetime(f_data['minute'].astype(str).str.zfill(2), format='%M')
        f_data.drop(columns=['minute'], inplace=True)
    elif time_period == 'hour' and 'hour' in f_data.columns:
        f_data['timestamp'] = pd.to_datetime(f_data["hour"].astype(str).str.zfill(2), format='%H')
        f_data.drop(columns=["hour"], inplace=True)
    elif time_period == "month" and "month" in f_data.columns:
        f_data["date"] = f_data["month"].dt.to_timestamp()
        f_data.drop(columns=["month"], inplace=True)

    return f_data


# ---------------------------
# Utility Functions
# ---------------------------
@st.cache_data
def get_available_years(f_data=None, dataset_path=None):
    print(dataset_path)
    """
    Returns available years in sorted order.
    If dataset_path is given, extract years from parquet partitions.
    f_data argument is kept for compatibility but ignored if dataset_path is provided.
    """
    if dataset_path:
        dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
        years = set()

        for fragment in dataset.get_fragments():
            expr = str(fragment.partition_expression)
            year_match = re.search(r"year=([0-9]{4})", expr)
            if year_match:
                years.add(int(year_match.group(1)))

        return sorted(years)
    else:
        # fallback: use f_data dataframe
        return sorted(f_data["month"].dt.year.unique())

@st.cache_data
def get_available_months_for_year(f_data=None, selected_year=None, dataset_path=None):
    """
    Returns available months for a given year in sorted order.
    If dataset_path is given, extract months from parquet partitions.
    f_data argument is kept for compatibility but ignored if dataset_path is provided.
    """
    if dataset_path and selected_year:
        dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
        months = set()

        for fragment in dataset.get_fragments():
            expr = str(fragment.partition_expression)
            year_match = re.search(r"year=([0-9]{4})", expr)
            month_match = re.search(r"month=([0-9]{1,2})", expr)

            if year_match and month_match:
                year = int(year_match.group(1))
                month = int(month_match.group(1))
                if year == selected_year:
                    months.add(month)

        return sorted(months)
    else:
        # fallback: use f_data dataframe
        filtered_data = f_data[f_data["month"].dt.year == selected_year]
        # return month names sorted by month number
        return sorted(
            filtered_data["month"].dt.strftime("%B").unique(),
            key=lambda x: pd.to_datetime(x, format="%B").month
        )


def find_sel_indexes(available_years, selected_year, dataset_path):
    """Finds the indexes of the selected year and month using partition metadata."""

    if selected_year is None:
        index_sel_year = len(available_years) - 1
    else:
        index_sel_year = available_years.index(selected_year)

    sel_year = available_years[index_sel_year]
    available_months = get_available_months_for_year(dataset_path=dataset_path, selected_year=sel_year)

    if len(available_months) == 1:
        if len(available_years) == 1:
            index_sel_month = 0
        else:
            index_sel_year = len(available_years) - 2
            # Update sel_year to new year index
            sel_year = available_years[index_sel_year]
            available_months = get_available_months_for_year(dataset_path=dataset_path, selected_year=sel_year)
            index_sel_month = len(available_months) - 1
    else:
        index_sel_month = len(available_months) - 2

    return index_sel_year, index_sel_month


def rgb_to_rgba(rgb_string, alpha=1.0):
    """
    Converts an 'rgb(r, g, b)' string to an 'rgba(r, g, b, alpha)' string.
    """
    # Extract numbers from the rgb string
    values = rgb_string.strip()[4:-1]  # Removes 'rgb(' and ')'
    r, g, b = values.split(',')
    return f'rgba({r.strip()}, {g.strip()}, {b.strip()}, {alpha})'


def filter_appliances_with_nonzero_sum(f_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out appliances with a total value sum of zero.
    """
    appliance_sums = f_data.groupby('appliance')['value'].sum()
    active_appliances = appliance_sums[appliance_sums != 0].index.tolist()
    return f_data[f_data['appliance'].isin(active_appliances)]


def format_value(value, unit):
    prefixes = ['', 'k', 'M', 'G']

    scale = 1000.0
    abs_value = abs(value)

    for i in range(len(prefixes) - 1, -1, -1):
        threshold = scale ** i
        if abs_value >= threshold:
            scaled_value = value / threshold
            return f"{scaled_value:.2f} {prefixes[i]}{unit}"
    return f"{value:.2f} {prefixes[0]}{unit}"  # for very small values
