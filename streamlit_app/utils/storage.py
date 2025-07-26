import os
import pandas as pd
import streamlit as st


# Utility function to safely load a JSON file into a dictionary
def safe_load_json(filename):
    """
    Safely loads a JSON file into a dictionary using pandas.

    If the file does not exist, shows an error message in Streamlit
    and returns an empty dictionary.
    """
    print("Loading data from JSON file...")  # Console log for debugging
    if os.path.exists(filename):
        # Read JSON as a pandas Series and convert it to a plain dictionary
        return pd.read_json(filename, typ="series").to_dict()
    else:
        # Display error in Streamlit UI and return empty dictionary
        st.error(f"Missing file: {filename}")
        return {}
