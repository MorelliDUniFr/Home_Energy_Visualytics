import streamlit as st
from utils.session_state_utils import load_value
import pandas as pd

TARIFFS = {
    'simple': {
        'fixed_fee_annual': 720.0,
        'rates_ct_kwh': {'all': 8.73 + 1.86},
        'timezones': [('00:00', '24:00', 'all')]
    },
    'duo': {
        'fixed_fee_annual': 518.3,
        'rates_ct_kwh': {
            'peak': 4.41 + 1.86,
            'offpeak': 1.96 + 1.86
        },
        'timezones': [
            ('07:00', '23:00', 'peak'),
            ('07:00', '13:30', 'peak'),
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
            ('09:00', '12:00', 'peak'),
            ('17:00', '19:00', 'peak'),
            ('00:00', '09:00', 'base'),
            ('12:00', '17:00', 'base'),
            ('19:00', '24:00', 'base'),
        ]
    }
}


def get_tariff_period(timestamp, tariff_name):
    """
    Determine the tariff period label ('peak', 'offpeak', 'base', or 'all') for a given timestamp and tariff.
    Assumes Swiss local time and applies the rules defined in TARIFFS.
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
        if day == 5:  # Saturday
            if pd.to_datetime('07:00').time() <= time < pd.to_datetime('13:30').time():
                return 'peak'
            return 'offpeak'
        return 'offpeak'  # Sunday

    if tariff_name == 'chrono':
        if day >= 5:  # Weekend (Saturday or Sunday)
            return 'base'
        if (pd.to_datetime('09:00').time() <= time < pd.to_datetime('12:00').time()) or \
           (pd.to_datetime('17:00').time() <= time < pd.to_datetime('19:00').time()):
            return 'peak'
        return 'base'


def compute_cost_from_consumption(df, tariff_name):
    """
    Compute the total cost in CHF of the energy consumption DataFrame `df` using the specified tariff.
    The DataFrame is expected to have columns 'timestamp' (datetime) and 'value' (power in Watts).
    """
    if df.empty:
        return 0.0

    df = df.copy()
    df['energy_wh'] = df['value'] * 10 / 3600  # Convert power (W) over 10s intervals to Wh
    df.set_index('timestamp', inplace=True)

    hourly = df['energy_wh'].resample('1h').sum().reset_index()
    hourly['period'] = hourly['timestamp'].apply(lambda ts: get_tariff_period(ts, tariff_name))

    tariff = TARIFFS[tariff_name]
    period_totals = hourly.groupby('period')['energy_wh'].sum()

    total_cost = 0.0
    for period, wh in period_totals.items():
        kwh = wh / 1000
        rate_ct = tariff['rates_ct_kwh'].get(period, tariff['rates_ct_kwh'].get('all', 0))
        total_cost += kwh * rate_ct / 100

    days = (hourly['timestamp'].dt.date.max() - hourly['timestamp'].dt.date.min()).days + 1
    total_cost += tariff['fixed_fee_annual'] * days / 365

    return total_cost


def _compute_total_wh(data):
    """
    Compute total energy in Wh from raw data with 10-second intervals.
    Aggregates by appliance and sums all values.
    """
    df = data.copy()
    grouped = df.groupby("appliance", as_index=False)["value"].sum()
    grouped["value"] *= 10 / 3600  # Convert to Wh
    return grouped["value"].sum()


def compute_total_consumption(data, suffix='_1'):
    """
    Compute total consumption in Wh and store it in Streamlit session state with the given suffix.
    """
    key = f'consumption_time_period{suffix}'
    load_value(key, 0.0)
    st.session_state[key] = _compute_total_wh(data)


def compute_total_consumption_and_cost(data, tariff_name='simple', suffix='_1'):
    """
    Compute total consumption and cost and store both in session state with the given suffix.
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
    Compute and store consumption and cost for two datasets using the selected tariff in session state.
    """
    tariff = st.session_state.get('selected_tariff', 'simple')
    compute_total_consumption_and_cost(data1, tariff, suffix='_1')
    compute_total_consumption_and_cost(data2, tariff, suffix='_2')
