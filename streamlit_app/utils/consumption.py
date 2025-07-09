import streamlit as st
from utils.session_state_utils import load_value
import pandas as pd

TARIFFS = {
    'simple': {
        'fixed_fee_annual': 720.0,  # CHF
        'rates_ct_kwh': { 'all': 8.73 + 1.86 },  # 10.59 ct/kWh total
        'timezones': [('00:00', '24:00', 'all')]
    },
    'duo': {
        'fixed_fee_annual': 518.3,
        'rates_ct_kwh': {
            'peak': 4.41 + 1.86,     # 6.27 ct/kWh total
            'offpeak': 1.96 + 1.86   # 3.82 ct/kWh total
        },
        'timezones': [
            ('07:00', '23:00', 'peak'),  # Mon-Fri peak hours
            ('07:00', '13:30', 'peak'),  # Sat peak hours
            ('00:00', '07:00', 'offpeak'),
            ('23:00', '24:00', 'offpeak'),
            ('13:30', '24:00', 'offpeak')  # Sat afternoon and all Sunday off-peak
        ]
    },
    'chrono': {
        'fixed_fee_annual': 906.0,
        'rates_ct_kwh': {
            'peak': 3.08 + 1.91,  # 4.99 ct/kWh total
            'base': 1.44 + 1.91   # 3.35 ct/kWh total
        },
        'timezones': [
            ('09:00', '12:00', 'peak'),  # Mon-Fri morning peak
            ('17:00', '19:00', 'peak'),  # Mon-Fri evening peak
            ('00:00', '09:00', 'base'),
            ('12:00', '17:00', 'base'),
            ('19:00', '24:00', 'base'),
        ]
    }
}

def get_tariff_period(timestamp, tariff_name):
    """
    Given a timestamp and tariff name, return the period label ('peak', 'offpeak', 'base', 'all').
    Assumes Swiss local time. Handles only weekdays/weekends per your tariff rules.
    """
    tariff = TARIFFS[tariff_name]
    time = timestamp.time()
    day = timestamp.dayofweek  # Monday=0, Sunday=6

    # For simple tariff, always 'all'
    if tariff_name == 'simple':
        return 'all'

    # For duo tariff, peak hours differ for weekdays and Saturday
    if tariff_name == 'duo':
        # Weekdays Mon-Fri
        if day <= 4:
            if time >= pd.to_datetime('07:00').time() and time < pd.to_datetime('23:00').time():
                return 'peak'
            else:
                return 'offpeak'
        # Saturday
        elif day == 5:
            if time >= pd.to_datetime('07:00').time() and time < pd.to_datetime('13:30').time():
                return 'peak'
            else:
                return 'offpeak'
        # Sunday
        else:
            return 'offpeak'

    # For chrono tariff
    if tariff_name == 'chrono':
        if day >= 5:  # Sat or Sun
            # All base on weekends
            return 'base'
        else:
            # Mon-Fri
            if (time >= pd.to_datetime('09:00').time() and time < pd.to_datetime('12:00').time()) or \
                    (time >= pd.to_datetime('17:00').time() and time < pd.to_datetime('19:00').time()):
                return 'peak'
            else:
                return 'base'


def compute_cost_from_consumption(df, tariff_name):
    if df.empty:
        return 0.0

    df = df.copy()
    df['energy_wh'] = df['value'] * 10 / 3600
    df.set_index('timestamp', inplace=True)

    # Resample to hourly consumption sums
    df_hourly = df['energy_wh'].resample('1h').sum().reset_index()

    # Vectorized tariff period assignment
    df_hourly['period'] = df_hourly['timestamp'].apply(lambda ts: get_tariff_period(ts, tariff_name))

    # Sum energy by tariff period
    period_consumption = df_hourly.groupby('period')['energy_wh'].sum()

    tariff = TARIFFS[tariff_name]

    total_cost_chf = 0.0
    for period, wh in period_consumption.items():
        kwh = wh / 1000
        rate_ct = tariff['rates_ct_kwh'].get(period, tariff['rates_ct_kwh'].get('all', 0))
        cost = kwh * rate_ct / 100
        total_cost_chf += cost

    # Prorate fixed fee
    start_date = df_hourly['timestamp'].dt.date.min()
    end_date = df_hourly['timestamp'].dt.date.max()
    days = (end_date - start_date).days + 1
    fixed_fee_prorated = tariff['fixed_fee_annual'] * days / 365
    total_cost_chf += fixed_fee_prorated

    return total_cost_chf


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


def compute_total_consumption_and_cost(data, tariff_name='simple', suffix='_1'):
    if data is None or data.empty:
        st.session_state[f'consumption_time_period{suffix}'] = 0.0
        st.session_state[f'cost_time_period{suffix}'] = 0.0
        return

    # Use your existing total Wh computation
    total_wh = _compute_total_wh(data)

    # Use your existing cost computation
    cost = compute_cost_from_consumption(data, tariff_name)

    # Store both values in session state
    st.session_state[f'consumption_time_period{suffix}'] = total_wh
    st.session_state[f'cost_time_period{suffix}'] = cost


def compute_total_consumptions(data1, data2):
    """
    Convenience function to compute and store both totals.
    """
    compute_total_consumption_and_cost(data1, tariff_name=st.session_state.selected_tariff, suffix='_1')
    compute_total_consumption_and_cost(data2, tariff_name=st.session_state.selected_tariff, suffix='_2')
