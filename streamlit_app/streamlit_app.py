import streamlit as st
from translations import t, languages

# --- Set default session state values ---
default_state = {
    "lang": "en",
    "time_period": "Day",
    "selected_date_1": None,
    "selected_month_1": None,
    "selected_year_1": None,
    "selected_date_2": None,
    "selected_month_2": None,
    "selected_year_2": None,
    "v_bar_chart": 'actual',
    "chart_type": 'bar_chart',
    "line_type": 'individual',
    "selected_columns": [],
    "selected_appliances": [],
}


for key, value in default_state.items():
    st.session_state.setdefault(key, value)

# --- Define pages ---
pg = st.navigation([
    st.Page('01_time_period_consumption.py', title=t('page_1_sidebar_title')),
])

# --- Page config ---
st.set_page_config(
    page_title=(t('page_title')),
    layout='wide',
)



# --- Run selected page ---
pg.run()
