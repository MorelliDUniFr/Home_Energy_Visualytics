import streamlit as st


# Save the temporary widget value (_key) to the persistent session state key
def store_value(key: str):
    """
    Copy the widget's temporary session key (_key) to the main persistent session key.

    Useful for syncing user input from UI widgets into a stable session value.
    """
    st.session_state[key] = st.session_state[f"_{key}"]


# Load a value from persistent session state into the temporary widget key (_key)
def load_value(key: str, default=None):
    """
    Load the persistent session value into the widget's temporary session key (_key).

    If the key doesn't exist yet and a default is provided, both keys are initialized with that default.
    """
    if key in st.session_state:
        st.session_state[f"_{key}"] = st.session_state[key]
    elif default is not None:
        st.session_state[f"_{key}"] = default
        st.session_state[key] = default
