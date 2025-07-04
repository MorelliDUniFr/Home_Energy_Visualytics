import streamlit as st
import os
import re
import calendar


@st.cache_data
def get_available_years(dataset_path: str):
    years = set()
    for foldername in os.listdir(dataset_path):
        match = re.match(r"date=(\d{4})-(\d{2})-(\d{2})", foldername)
        if match:
            year = int(match.group(1))
            years.add(year)
    return sorted(years)


@st.cache_data
def get_available_months_for_year(dataset_path: str, selected_year: int):
    months = set()
    for foldername in os.listdir(dataset_path):
        match = re.match(r"date=(\d{4})-(\d{2})-(\d{2})", foldername)
        if match and int(match.group(1)) == selected_year:
            month_num = int(match.group(2))
            month_name = calendar.month_name[month_num]  # e.g. 3 -> 'March'
            months.add(month_name)
    # Sort by month number to keep correct order
    return sorted(months, key=lambda m: list(calendar.month_name).index(m))


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