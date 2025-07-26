import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title= 'Dashboard',
    layout='wide',
)

from utils.session_state_utils import store_value, load_value
from utils.translations import t, languages

# --- Set default session state values ---
default_state = {
    "selected_columns": [],
    "selected_appliances": [],
    "consumption_time_period_1": 0,
    "consumption_time_period_2": 0,
}

load_value("selected_tariff", "simple")
load_value("lang", "en")

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Define app_pages ---
pg = st.navigation([
    st.Page('app_pages/01_time_period_consumption.py', title=t('page_1_sidebar_title')),
    st.Page('app_pages/02_compare_with_previous_period.py', title=t('page_2_sidebar_title')),
    st.Page('app_pages/03_analyze_data.py', title=t('page_3_sidebar_title')),
])



tariff_labels = {
    "simple": "Simple",
    "duo": "Duo ",
    "chrono": "Chrono"
}

# --- Sidebar language selector ---
with st.sidebar:
    selected_tariff = st.pills(
        label=t("tariff"),
        options=["simple", "duo", "chrono"],
        selection_mode="single",
        format_func=lambda code: tariff_labels[code],
        key="_selected_tariff",
        on_change=store_value,
        args=("selected_tariff",),
    )

    selected_lang = st.selectbox(
        label="🌐 " + t("select_language"),
        options=list(languages.keys()),
        format_func=lambda code: languages[code],
        key="_lang",
        on_change=store_value,
        args=("lang",),
    )

# --- Run selected page ---
pg.run()
