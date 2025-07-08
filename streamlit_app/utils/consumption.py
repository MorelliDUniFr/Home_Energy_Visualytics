import streamlit as st
from utils.session_state_utils import load_value


def _compute_total_wh(data):
    """
    Helper function: computes total energy in Wh from raw data (in 10-second intervals).
    """
    df = data.copy()
    df = df.groupby("appliance", as_index=False)["value"].sum()
    df["value"] *= 10 / 3600  # Convert to Wh
    return df["value"].sum()


def compute_total_consumption(data, suffix='_1'):
    """
    Computes total Wh and stores it in session state with dynamic suffix.
    """
    key = f'consumption_time_period{suffix}'
    load_value(key, 0.0)
    st.session_state[key] = _compute_total_wh(data)


def compute_total_consumptions(data1, data2):
    """
    Convenience function to compute and store both totals.
    """
    compute_total_consumption(data1, suffix='_1')
    compute_total_consumption(data2, suffix='_2')
