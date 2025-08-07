import streamlit as st
import pandas as pd
import plotly.express as px
from babel.dates import format_date
from datetime import datetime
from utils.translations import t, translate_appliance_name
from utils.filters import time_filter
from utils.formatting import format_value
from utils.appliances import get_ordered_appliance_list
from utils.session_state_utils import store_value, load_value


def plot_vertical_bar_chart(f_data_1, f_data_2, colors, time_period):
    """
    Plot a grouped vertical bar chart comparing appliance usage between two datasets.

    Args:
        f_data_1 (pd.DataFrame): DataFrame for first period with columns including 'date', 'appliance', 'value'
        f_data_2 (pd.DataFrame): DataFrame for second period with same schema
        colors (dict): appliance name -> color mapping (e.g. {'Washing Machine': 'rgb(123,45,67)', ...})
        time_period (str): one of 'Day', 'Week', 'Month', 'Year' to format dates and group data accordingly

    Functionality:
        - Formats and compares two datasets on the x-axis by appliance.
        - Allows toggling between absolute consumption (Wh) and relative percentage.
        - Warns if the selected periods are the same.
        - Uses localized date formatting based on the current Streamlit language.
    """
    # Ensure 'date' columns are datetime.date objects for formatting
    f_data_1['date'] = pd.to_datetime(f_data_1['date']).dt.date
    f_data_2['date'] = pd.to_datetime(f_data_2['date']).dt.date

    # Format the first date of each dataset according to the time_period
    date_format_str = time_filter[time_period]["date_format"]

    date_1 = f_data_1['date'].iloc[0].strftime(date_format_str)
    date_2 = f_data_2['date'].iloc[0].strftime(date_format_str)

    # Further localize date strings using babel, depending on time_period
    lang = st.session_state.lang
    if time_period == "Day":
        date_1 = format_date(datetime.strptime(date_1, "%d %B %Y").date(), format="d MMMM yyyy", locale=lang)
        date_2 = format_date(datetime.strptime(date_2, "%d %B %Y").date(), format="d MMMM yyyy", locale=lang)
    elif time_period == "Month":
        date_1 = format_date(datetime.strptime(date_1, "%B %Y").date(), format="MMMM yyyy", locale=lang)
        date_2 = format_date(datetime.strptime(date_2, "%B %Y").date(), format="MMMM yyyy", locale=lang)
    elif time_period == "Year":
        date_1 = format_date(datetime.strptime(date_1, "%Y").date(), format="yyyy", locale=lang)
        date_2 = format_date(datetime.strptime(date_2, "%Y").date(), format="yyyy", locale=lang)

    # If the two selected dates are identical, warn user and stop further plotting
    if date_1 == date_2:
        st.warning(t('warning_same_dates'), icon='ℹ️')
        st.stop()

    # Add the formatted date string as a new column for grouping by time_period label
    f_data_1 = f_data_1.copy()
    f_data_2 = f_data_2.copy()
    f_data_1[time_period] = date_1
    f_data_2[time_period] = date_2

    # Combine the two datasets into one DataFrame for plotting
    combined_data = pd.concat([f_data_1, f_data_2])

    # Radio buttons for choosing between absolute consumption and relative percentages
    options = {
        "actual": t('actual_consumption') + " (Wh)",
        "relative": t('relative_consumption') + " (%)"
    }

    # Load previous selection from session state or default to 'actual'
    load_value('v_bar_chart', default='actual')

    st.radio(
        label=f"{t('display_mode')}:",
        options=list(options.keys()),
        format_func=lambda key: options[key],
        horizontal=True,
        key='_v_bar_chart',
        on_change=store_value,
        args=('v_bar_chart',),
    )

    # Group by appliance and time_period and sum the raw 'value' (assumed to be power readings)
    sample_interval_seconds = 10
    grouped = combined_data.groupby(['appliance', time_period], observed=False)['value'].sum().reset_index()

    # Convert summed values from power (W sampled every 10s) to energy in Wh
    grouped['value'] = grouped['value'] * sample_interval_seconds / 3600

    # Format values for tooltip display
    grouped["formatted_value"] = grouped["value"].apply(format_value, unit="Wh")

    if st.session_state.v_bar_chart == "relative":
        # Pivot data to have appliances as rows and time periods as columns with 'value'
        pivot = grouped.pivot(index='appliance', columns=time_period, values='value')

        # Replace zeros in date_1 to avoid division by zero — you could also keep zeros if you want 0% in that case
        pivot[date_1] = pivot[date_1].replace(0, float('nan'))

        # Calculate percentage relative to date_1 baseline
        pivot_percentage = pd.DataFrame({
            date_1: 100,  # baseline 100%
            date_2: (pivot[date_2] / pivot[date_1]) * 100
        })

        # Melt to long format for merging
        percentage_long = pivot_percentage.reset_index().melt(id_vars='appliance', var_name=time_period,
                                                              value_name='percentage')

        # Merge with original grouped data
        grouped = pd.merge(grouped, percentage_long, on=['appliance', time_period], how='left')

        # Set plotting column to 'percentage'
        y_column = 'percentage'
        y_label = t('energy_consumption') + ' (%)'
        hover_text = '%{customdata[0]}<extra></extra>'
        grouped["custom"] = grouped["value"]  # show actual Wh in hover tooltip
    else:
        y_column = 'value'
        y_label = t('energy_consumption') + ' (Wh)'
        hover_text = '%{customdata[0]}<extra></extra>'

    # Fix x-axis order by date labels
    grouped[time_period] = pd.Categorical(grouped[time_period], [date_1, date_2], ordered=True)
    grouped.sort_values(["appliance", time_period], inplace=True)

    # Filter appliances present in either dataset and translate appliance names for display
    appliances_in_data = pd.unique(pd.concat([f_data_1["appliance"], f_data_2["appliance"]]))
    filtered_appliance_order = [
        translate_appliance_name(app) for app in get_ordered_appliance_list()
        if app in appliances_in_data
    ]

    grouped["translated_appliance"] = grouped["appliance"].apply(translate_appliance_name)

    # Create the bar chart with plotly express
    fig = px.bar(
        grouped,
        x="translated_appliance",
        y=y_column,
        color=time_period,
        barmode="group",
        title=f"{t('appliance_usage_period')} {date_1} {t('and')} {date_2}",
        labels={"appliance": t('appliances'), y_column: y_label},
        color_discrete_sequence=[colors[app] for app in colors],
        custom_data=["formatted_value"],
    )

    # Enforce x-axis category order
    fig.update_xaxes(categoryorder='array', categoryarray=filtered_appliance_order)

    # Set bar colors correctly for each time period
    for trace in fig.data:
        # trace.name corresponds to the time_period label, e.g. date_1 or date_2
        appliances_for_trace = grouped[grouped[time_period] == trace.name]['appliance']
        trace.marker.color = [colors[app] for app in appliances_for_trace]

    # Layout tweaks to disable zoom, legend, and improve appearance
    fig.update_layout(
        xaxis_title=t('appliances'),
        yaxis_title=y_label,
        font=dict(family="Arial", size=12),
        bargap=0.2,
        bargroupgap=0.2,
        showlegend=False,
        dragmode="zoom",
        xaxis_fixedrange=False,
        yaxis_fixedrange=False,
    )

    # Tooltip formatting
    fig.update_traces(hovertemplate=hover_text)

    # Render chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
