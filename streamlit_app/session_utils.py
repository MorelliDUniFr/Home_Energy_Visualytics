import streamlit as st

def store_value(key: str):
    """Copy the widget temp key (_key) to the persistent session key (key)."""
    st.session_state[key] = st.session_state[f"_{key}"]

def load_value(key: str, default=None):
    """Load persistent session value into the widget temp key (_key)."""
    if key in st.session_state:
        st.session_state[f"_{key}"] = st.session_state[key]
    elif default is not None:
        st.session_state[f"_{key}"] = default
        st.session_state[key] = default
