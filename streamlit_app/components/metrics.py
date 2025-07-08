import streamlit as st
from utils.translations import t
from utils.session_state_utils import load_value
from utils.formatting import format_value
from utils.config_utils import consumption_cost


def _calculate_cost(wh: float) -> float:
    """Converts Wh to CHF based on configured cost per kWh."""
    return (wh / 1000) * (consumption_cost / 100)


def _get_consumption_and_cost(key_suffix: str):
    """Loads consumption and computes its cost."""
    key = f'consumption_time_period{key_suffix}'
    load_value(key, 0)
    consumption = st.session_state[key]
    cost = _calculate_cost(consumption)
    return consumption, cost


def display_consumption_metrics(cols, key_suffix: str = '_1', is_comparison: bool = False):
    consumption, cost = _get_consumption_and_cost(key_suffix)

    with cols[0]:
        st.metric(label=t('total_consumption'), value=f"{format_value(consumption, 'Wh')}")

    with cols[1]:
        if is_comparison:
            # Determine which other time period to compare to
            other_suffix = '_2' if key_suffix == '_1' else '_1'
            other_consumption, other_cost = _get_consumption_and_cost(other_suffix)

            if other_cost != 0:
                delta = ((cost - other_cost) / other_cost) * 100
                delta_str = f"{delta:.2f}%"
            else:
                delta_str = None  # Avoid dividing by 0

            st.metric(label=t('total_cost'), value=f"{cost:.2f} CHF", delta=delta_str, delta_color='inverse')
        else:
            st.metric(label=t('total_cost'), value=f"{cost:.2f} CHF")
