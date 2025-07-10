from utils.translations import t, translate_appliance_name
import plotly.express as px
from babel.dates import format_date
from datetime import datetime
import pandas as pd
from utils.filters import time_filter
from utils.formatting import format_value
from utils.appliances import get_ordered_appliance_list
import streamlit as st
from utils.session_state_utils import store_value, load_value


def plot_vertical_bar_chart(f_data_1, f_data_2, colors, time_period):
    """
    Plots a grouped bar chart comparing appliance usage between two datasets (dates),
    with an option to view in actual Wh or as relative percentages.
    """
    # Convert 'date' column to datetime.date
    f_data_1['date'] = pd.to_datetime(f_data_1['date']).dt.date
    f_data_2['date'] = pd.to_datetime(f_data_2['date']).dt.date

    # Extract and format the date for both datasets
    date_1 = f_data_1['date'].iloc[0].strftime(time_filter[time_period]["date_format"])
    date_2 = f_data_2['date'].iloc[0].strftime(time_filter[time_period]["date_format"])

    if time_period == "Day":
        date_1 = format_date(datetime.strptime(date_1, "%d %B %Y").date(), format="d MMMM yyyy", locale=st.session_state.lang)
        date_2 = format_date(datetime.strptime(date_2, "%d %B %Y").date(), format="d MMMM yyyy", locale=st.session_state.lang)
    elif time_period == "Month":
        date_1 = format_date(datetime.strptime(date_1, "%B %Y").date(), format="MMMM yyyy", locale=st.session_state.lang)
        date_2 = format_date(datetime.strptime(date_2, "%B %Y").date(), format="MMMM yyyy", locale=st.session_state.lang)
    elif time_period == "Year":
        date_1 = format_date(datetime.strptime(date_1, "%Y").date(), format="yyyy", locale=st.session_state.lang)
        date_2 = format_date(datetime.strptime(date_2, "%Y").date(), format="yyyy", locale=st.session_state.lang)

    # Warn the user if the dates selected are the same
    if date_1 == date_2:
        st.warning(t('warning_same_dates'), icon='ℹ️')
        st.stop()

    # Add the formatted date as a new column to both datasets
    f_data_1 = f_data_1.copy()
    f_data_2 = f_data_2.copy()
    f_data_1[time_period], f_data_2[time_period] = date_1, date_2

    # Combine datasets
    combined_data = pd.concat([f_data_1, f_data_2])

    options = {
        "actual": t('actual_consumption') + " (Wh)",
        "relative": t('relative_consumption') + " (%)"
    }

    load_value('v_bar_chart', default='actual')

    def get_label(option_key):
        return options[option_key]

    st.radio(
        label=f"{t('display_mode')}:",
        options=list(options.keys()),
        format_func=get_label,
        horizontal=True,
        key='_v_bar_chart',
        on_change=store_value,
        args=('v_bar_chart',),
    )

    # Group and calculate Wh
    sample_interval = 10  # seconds
    grouped = combined_data.groupby(['appliance', time_period], observed=False)['value'].sum().reset_index()

    grouped['value'] *= sample_interval / 3_600  # to Wh
    grouped["formatted_value"] = grouped["value"].apply(format_value, unit="Wh")

    if st.session_state.v_bar_chart == "relative":
        # Normalize each appliance's value relative to its maximum across both days
        grouped['percentage'] = grouped.groupby('appliance', observed=False)['value'].transform(
            lambda x: 100 * x / x.max())
        y_column = 'percentage'
        y_label = 'Energy Consumption (%)'
        hover_text = '%{customdata[0]}<extra></extra>'
        grouped["custom"] = grouped["value"]  # Actual Wh values
    else:
        y_column = 'value'
        y_label = f'{t('energy_consumption')} (Wh)'
        hover_text = '%{customdata[0]}<extra></extra>'

    grouped[time_period] = pd.Categorical(grouped[time_period], [date_1, date_2], ordered=True)
    grouped.sort_values(["appliance", time_period], inplace=True)
    filtered_appliance_order = [translate_appliance_name(app) for app in get_ordered_appliance_list() if
                                app in pd.unique(pd.concat([f_data_1["appliance"], f_data_2["appliance"]]))]

    grouped["translated_appliance"] = grouped["appliance"].apply(translate_appliance_name)

    # Plot
    fig = px.bar(
        grouped,
        x="translated_appliance",
        y=y_column,
        color=time_period,
        barmode="group",
        title=f"{t('appliance_usage_period')} {date_1} {t('and')} {date_2}",
        labels={"appliance": "Appliance", y_column: y_label},
        color_discrete_sequence=[colors[app] for app in colors],
        custom_data=["formatted_value"],
    )

    # Force correct x-axis order
    fig.update_xaxes(categoryorder='array', categoryarray=filtered_appliance_order)

    # Custom bar colors
    for i, dt in enumerate(fig.data):
        original_appliances = grouped[grouped[time_period] == dt.name]['appliance']
        dt.marker.color = [colors[app] for app in original_appliances]

    fig.update_layout(
        xaxis_title=t('appliances'),
        yaxis_title=y_label,
        font=dict(family="Arial", size=12),
        bargap=0.2,
        bargroupgap=0.2,
        showlegend=False,
        dragmode=False,  # Disables box/lasso/zoom selection
        xaxis_fixedrange=True,  # Prevents zooming/panning on X
        yaxis_fixedrange=True,  # Prevents zooming/panning on Y
    )

    fig.update_traces(hovertemplate=hover_text)

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)
