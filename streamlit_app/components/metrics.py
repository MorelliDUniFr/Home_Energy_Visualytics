import streamlit as st
from utils.translations import t
from utils.session_state_utils import load_value
from utils.formatting import format_value
from utils.consumption import compute_cost_from_consumption


def _get_consumption_and_cost(key_suffix: str, tariff_name: str):
    """Loads raw power data and computes total consumption and cost based on tariff."""
    key = f'raw_power_data{key_suffix}'  # assuming raw power data saved here as a DataFrame
    load_value(key, None)
    df = st.session_state.get(key)
    if df is None or df.empty:
        return 0.0, 0.0

    # Compute total consumption in Wh for display (optional)
    total_wh = df['value'].sum() * 10 / 3600

    # Compute cost using your detailed tariff function
    cost = compute_cost_from_consumption(df, tariff_name)

    return total_wh, cost


def display_consumption_metrics(cols, live_value, key_suffix: str = '_1', is_comparison: bool = False):
    load_value(f'consumption_time_period{key_suffix}', 0.0)
    load_value(f'cost_time_period{key_suffix}', 0.0)

    consumption = st.session_state.get(f'consumption_time_period{key_suffix}', 0.0)
    cost = st.session_state.get(f'cost_time_period{key_suffix}', 0.0)

    # Defaults for deltas
    consumption_delta_str = None
    cost_delta_str = None

    if is_comparison:
        other_suffix = '_2' if key_suffix == '_1' else '_1'
        load_value(f'consumption_time_period{other_suffix}', 0.0)
        load_value(f'cost_time_period{other_suffix}', 0.0)

        other_consumption = st.session_state.get(f'consumption_time_period{other_suffix}', 0.0)
        other_cost = st.session_state.get(f'cost_time_period{other_suffix}', 0.0)

        if other_consumption != 0:
            consumption_delta = ((consumption - other_consumption) / other_consumption) * 100
            consumption_delta_str = f"{consumption_delta:.2f}%"

        if other_cost != 0:
            cost_delta = ((cost - other_cost) / other_cost) * 100
            cost_delta_str = f"{cost_delta:.2f}%"

    with cols[0]:
        st.metric(
            label=t('total_consumption'),
            value=f"{format_value(consumption, 'Wh')}",
            delta=consumption_delta_str,
            delta_color="inverse"
        )

    with cols[1]:
        st.metric(
            label=t('total_cost'),
            value=f"{cost:.2f} CHF",
            delta=cost_delta_str,
            delta_color='inverse'
        )

    with cols[2]:
        st.metric(
            label=t('total_live_power'),
            value=f"{format_value(live_value, 'W')}",
            delta=None  # No delta for live value
        )


