from datetime import date, timedelta
import pandas as pd
import calendar
from dateutil.relativedelta import relativedelta


# Generate commonly used relative date values
def get_date_ranges():
    """
    Returns a dictionary of reference dates such as yesterday, last week,
    last month, and last year. Handles calendar edge cases (e.g. Feb 29).
    """
    today = date.today()

    # Common relative dates
    yesterday = today - timedelta(days=1)
    another_yesterday = today - timedelta(days=2)
    last_week = today - timedelta(weeks=1)

    # Handle month transitions safely (e.g., from March 31 to Feb 28)
    last_month = today - relativedelta(months=1)
    last_month_day = min(today.day, calendar.monthrange(last_month.year, last_month.month)[1])
    last_month_date = date(last_month.year, last_month.month, last_month_day)

    # Handle year transitions safely
    last_year = today - relativedelta(years=1)
    last_year_day = min(today.day, calendar.monthrange(last_year.year, last_year.month)[1])
    last_year_date = date(last_year.year, last_year.month, last_year_day)

    return {
        "today": today,
        "yesterday": yesterday,
        "another_yesterday": another_yesterday,
        "last_week": last_week,
        "last_month": last_month_date,
        "last_year": last_year_date,
    }


# Predefined configuration for different time filters used in the app
date_ranges = get_date_ranges()

time_filter = {
    "Day": {
        "tick_format": "%H:%M",  # x-axis format
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


# Filter out appliances that have zero consumption over the period
def filter_appliances_with_nonzero_sum(f_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out appliances whose total 'value' across all rows is zero.

    Args:
        f_data (pd.DataFrame): Input data with 'appliance' and 'value' columns.

    Returns:
        pd.DataFrame: Filtered data with only active appliances.
    """
    appliance_sums = f_data.groupby('appliance')['value'].sum()
    active_appliances = appliance_sums[appliance_sums != 0].index.tolist()
    return f_data[f_data['appliance'].isin(active_appliances)]


# Aggregate the data over time and appliance groups
def aggregate_data(f_data: pd.DataFrame, group_by_columns: list, date_column: str,
                   time_period: str = None) -> pd.DataFrame:
    """
    Aggregates input data by averaging values over specified groupings.
    Handles conversion of time columns (minute/hour/month) to appropriate datetime formats.

    Args:
        f_data (pd.DataFrame): Input DataFrame containing 'value' and time columns.
        group_by_columns (list): Columns to group by, e.g., ['hour', 'minute'] or 'date'.
        date_column (str): Column name containing timestamp/date information.
        time_period (str): One of 'minute', 'hour', or 'month'. Used for formatting.

    Returns:
        pd.DataFrame: Aggregated and formatted data.
    """
    f_data = f_data.copy()  # Work on a copy to avoid modifying the original

    # Ensure datetime format and drop invalid entries
    f_data[date_column] = pd.to_datetime(f_data[date_column], errors='coerce')
    f_data.dropna(subset=[date_column], inplace=True)

    # Flatten nested groupings if needed
    group_by_columns = [item for sublist in group_by_columns for item in
                        (sublist if isinstance(sublist, list) else [sublist])]

    # Group data and compute the mean of values
    f_data = f_data.groupby(group_by_columns, as_index=False)['value'].mean()

    # Convert grouped time columns to usable datetime columns
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
