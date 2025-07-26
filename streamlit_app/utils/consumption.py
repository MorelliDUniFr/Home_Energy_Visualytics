import streamlit as st
from utils.session_state_utils import load_value
import pandas as pd

# Define energy pricing schemes for each tariff
TARIFFS = {
    'simple': {
        'fixed_fee_annual': 720.0,
        'rates_ct_kwh': {'all': 8.73 + 1.86},  # Energy + grid fee in cts
        'timezones': [('00:00', '24:00', 'all')]
    },
    'duo': {
        'fixed_fee_annual': 518.3,
        'rates_ct_kwh': {
            'peak': 4.41 + 1.86,
            'offpeak': 1.96 + 1.86
        },
        'timezones': [
            # Weekday peak
            ('07:00', '23:00', 'peak'),
            # Saturday peak
            ('07:00', '13:30', 'peak'),
            # Offpeak
            ('00:00', '07:00', 'offpeak'),
            ('23:00', '24:00', 'offpeak'),
            ('13:30', '24:00', 'offpeak')
        ]
    },
    'chrono': {
        'fixed_fee_annual': 906.0,
        'rates_ct_kwh': {
            'peak': 3.08 + 1.91,
            'base': 1.44 + 1.91
        },
        'timezones': [
            # Peak hours (weekday)
            ('09:00', '12:00', 'peak'),
            ('17:00', '19:00', 'peak'),
            # Base hours (rest of the day and weekend)
            ('00:00', '09:00', 'base'),
            ('12:00', '17:00', 'base'),
            ('19:00', '24:00', 'base'),
        ]
    }
}


def get_tariff_period(timestamp, tariff_name):
    """
    Determine which tariff period (peak/base/offpeak/all) applies for a given timestamp.

    Args:
        timestamp (pd.Timestamp): The time of the reading.
        tariff_name (str): The name of the tariff plan.

    Returns:
        str: One of 'peak', 'offpeak', 'base', or 'all'.
    """
    time = timestamp.time()
    day = timestamp.dayofweek  # Monday=0, Sunday=6

    if tariff_name == 'simple':
        return 'all'

    if tariff_name == 'duo':
        if day <= 4:  # Monday to Friday
            if pd.to_datetime('07:00').time() <= time < pd.to_datetime('23:00').time():
                return 'peak'
            return 'offpeak'
        elif day == 5:  # Saturday
            if pd.to_datetime('07:00').time() <= time < pd.to_datetime('13:30').time():
                return 'peak'
            return 'offpeak'
        else:  # Sunday
            return 'offpeak'

    if tariff_name == 'chrono':
        if day >= 5:  # Weekend
            return 'base'
        if (pd.to_datetime('09:00').time() <= time < pd.to_datetime('12:00').time()) or \
                (pd.to_datetime('17:00').time() <= time < pd.to_datetime('19:00').time()):
            return 'peak'
        return 'base'


def compute_cost_from_consumption(df, tariff_name):
    """
    Calculate the energy cost in CHF for the given DataFrame using the selected tariff.

    Args:
        df (pd.DataFrame): Must contain 'timestamp' (datetime) and 'value' (power in Watts).
        tariff_name (str): The name of the tariff to use.

    Returns:
        float: Total cost in CHF.
    """
    if df.empty:
        return 0.0

    df = df.copy()
    # Convert 10-second interval power readings to energy in Wh
    df['energy_wh'] = df['value'] * 10 / 3600
    df.set_index('timestamp', inplace=True)

    # Aggregate to hourly energy usage
    hourly = df['energy_wh'].resample('1h').sum().reset_index()

    # Determine tariff period for each hour
    hourly['period'] = hourly['timestamp'].apply(lambda ts: get_tariff_period(ts, tariff_name))

    tariff = TARIFFS[tariff_name]
    period_totals = hourly.groupby('period')['energy_wh'].sum()

    total_cost = 0.0
    for period, wh in period_totals.items():
        kwh = wh / 1000
        # Get rate per period or fallback to 'all'
        rate_ct = tariff['rates_ct_kwh'].get(period, tariff['rates_ct_kwh'].get('all', 0))
        total_cost += kwh * rate_ct / 100  # Convert cts to CHF

    # Add pro-rated fixed annual fee
    days = (hourly['timestamp'].dt.date.max() - hourly['timestamp'].dt.date.min()).days + 1
    total_cost += tariff['fixed_fee_annual'] * days / 365

    return total_cost


def _compute_total_wh(data):
    """
    Convert raw power readings into total energy (Wh) for all appliances combined.

    Args:
        data (pd.DataFrame): Data with 'value' in Watts.

    Returns:
        float: Total energy in Wh.
    """
    df = data.copy()
    grouped = df.groupby("appliance", as_index=False)["value"].sum()
    grouped["value"] *= 10 / 3600  # 10-second intervals to Wh
    return grouped["value"].sum()


def compute_total_consumption(data, suffix='_1'):
    """
    Compute and store total energy consumption (Wh) in Streamlit session state.

    Args:
        data (pd.DataFrame): Raw consumption data.
        suffix (str): Key suffix to store/retrieve state separately (e.g., '_1', '_2').
    """
    key = f'consumption_time_period{suffix}'
    load_value(key, 0.0)
    st.session_state[key] = _compute_total_wh(data)


def compute_total_consumption_and_cost(data, tariff_name='simple', suffix='_1'):
    """
    Compute and store both consumption (Wh) and cost (CHF) for a dataset.

    Args:
        data (pd.DataFrame): Raw consumption data.
        tariff_name (str): Selected tariff name.
        suffix (str): Key suffix for session state.
    """
    key_c = f'consumption_time_period{suffix}'
    key_cost = f'cost_time_period{suffix}'

    if data is None or data.empty:
        st.session_state[key_c] = 0.0
        st.session_state[key_cost] = 0.0
        return

    st.session_state[key_c] = _compute_total_wh(data)
    st.session_state[key_cost] = compute_cost_from_consumption(data, tariff_name)


def compute_total_consumptions(data1, data2):
    """
    Compute and store consumption and cost for two datasets using the selected tariff.

    Args:
        data1 (pd.DataFrame): First time period dataset.
        data2 (pd.DataFrame): Second time period dataset.
    """
    tariff = st.session_state.get('selected_tariff', 'simple')
    compute_total_consumption_and_cost(data1, tariff, suffix='_1')
    compute_total_consumption_and_cost(data2, tariff, suffix='_2')
