import streamlit as st
import os
import re
import calendar

# Cache the function to avoid reprocessing folder names on reruns
@st.cache_data
def get_available_years(dataset_path: str):
    """
    Extracts all unique years from the dataset directory names.
    Expects folder names in the format 'date=YYYY-MM-DD'.
    """
    years = set()
    for foldername in os.listdir(dataset_path):
        match = re.match(r"date=(\d{4})-(\d{2})-(\d{2})", foldername)
        if match:
            year = int(match.group(1))
            years.add(year)
    return sorted(years)

# Cache the function to avoid redundant I/O
@st.cache_data
def get_available_months_for_year(dataset_path: str, selected_year: int):
    """
    Returns a list of month names available for the given year,
    based on folder names in the dataset directory.
    """
    months = set()
    for foldername in os.listdir(dataset_path):
        match = re.match(r"date=(\d{4})-(\d{2})-(\d{2})", foldername)
        if match and int(match.group(1)) == selected_year:
            month_num = int(match.group(2))
            month_name = calendar.month_name[month_num]  # e.g. 3 -> 'March'
            months.add(month_name)
    # Ensure correct month order (Jan -> Dec)
    return sorted(months, key=lambda m: list(calendar.month_name).index(m))


def find_sel_indexes(available_years, selected_year, dataset_path):
    """
    Determines default index positions for selected year and month.

    - If no year is selected, defaults to the most recent year.
    - If only one month is available for that year and there's more than one year,
      the logic steps back to the previous year to avoid displaying a single-month view.
    - Returns a tuple: (year_index, month_index)
    """
    if selected_year is None:
        index_sel_year = len(available_years) - 1  # Default to latest year
    else:
        index_sel_year = available_years.index(selected_year)

    sel_year = available_years[index_sel_year]
    available_months = get_available_months_for_year(dataset_path=dataset_path, selected_year=sel_year)

    # Handle the edge case: only one month available
    if len(available_months) == 1:
        if len(available_years) == 1:
            index_sel_month = 0
        else:
            index_sel_year = len(available_years) - 2  # Step back one year
            sel_year = available_years[index_sel_year]
            available_months = get_available_months_for_year(dataset_path=dataset_path, selected_year=sel_year)
            index_sel_month = len(available_months) - 1
    else:
        # Default to the second-to-last month for context
        index_sel_month = len(available_months) - 2

    return index_sel_year, index_sel_month
