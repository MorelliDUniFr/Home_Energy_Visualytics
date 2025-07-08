import streamlit as st
from utils.translations import t, languages


# --- Set default session state values ---
default_state = {
    "lang": "en",
    "selected_columns": [],
    "selected_appliances": [],
    "consumption_time_period_1": 0,
    "consumption_time_period_2": 0,
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Define app_pages ---
pg = st.navigation([
    st.Page('app_pages/01_time_period_consumption.py', title=t('page_1_sidebar_title')),
    st.Page('app_pages/02_compare_with_previous_period.py', title=t('page_2_sidebar_title')),
    st.Page('app_pages/03_analyze_data.py', title=t('page_3_sidebar_title')),
])

# --- Page config ---
st.set_page_config(
    page_title=(t('page_title')),
    layout='wide',
)

# --- Sidebar language selector ---
with st.sidebar:
    selected_lang = st.selectbox(
        label="🌐 " + t("select_language"),
        options=list(languages.keys()),
        format_func=lambda code: languages[code],
        index=list(languages).index(st.session_state.lang),
    )

    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

# --- Run selected page ---
pg.run()
