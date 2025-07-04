import os
import pandas as pd
import streamlit as st


def safe_load_json(filename):
    """Safely loads a JSON file into a dictionary."""
    print("Loading data from JSON file...")
    if os.path.exists(filename):
        return pd.read_json(filename, typ="series").to_dict()
    else:
        st.error(f"Missing file: {filename}")
        return {}