import streamlit as st
import plotly.express as px
import pandas as pd
from utils.translations import t, translate_appliance_name
from utils.formatting import format_value
from utils.appliances import get_appliance_colors
from utils.filters import time_filter
from utils.appliances import rgb_to_rgba, get_ordered_appliance_list
from utils.session_state_utils import load_value, store_value


def plot_line_chart(f_data, t_filter):
    required_columns = ['appliance', 'value', 'timestamp']
    # Check if all required columns exist in the data
    if not all(col in f_data.columns for col in required_columns):
        st.warning(f'Data must contain the following columns: {', '.join(required_columns)}.', icon='ℹ️')

    # Convert timestamp column to datetime objects
    f_data['timestamp'] = pd.to_datetime(f_data['timestamp'])

    # Convert power (W) sampled every 10 seconds to energy in Wh
    f_data['energy_Wh'] = f_data['value'] * (10 / 3600)

    # Determine aggregation period (e.g., 15min, 1h) based on selected time filter
    floor_period = time_filter[t_filter]['floor_period']

    # Round down timestamps to the nearest aggregation interval
    f_data['agg_time'] = f_data['timestamp'].dt.floor(floor_period)

    # Aggregate energy consumption per appliance per aggregation time
    f_data = (
        f_data
        .groupby(['agg_time', 'appliance'], as_index=False)
        .agg({'energy_Wh': 'sum'})
    )

    # Rename aggregated column for consistency
    f_data = f_data.rename(columns={'energy_Wh': 'value'})

    # Format the values for display in tooltips or labels
    f_data['formatted_value'] = f_data['value'].apply(format_value, args=('Wh',))

    # Translate appliance names for UI display
    f_data['translated_appliance'] = f_data['appliance'].apply(translate_appliance_name)

    # Determine line chart style based on user selection (cumulative or individual lines)
    if st.session_state.line_type == 'cumulative':
        # Reverse legend order so last appliance is on top visually
        legend_order = get_ordered_appliance_list()[::-1]
        legend_traceorder = "reversed"
        stackgroup="one"  # Enable stacking of areas (cumulative)
        fill="tonexty"    # Fill between lines cumulatively
    elif st.session_state.line_type == 'individual':
        legend_order = get_ordered_appliance_list()
        legend_traceorder = "normal"
        stackgroup=None   # No stacking; separate lines
        fill="tozeroy"   # Fill area from line to zero baseline

    # Create area plot with Plotly Express
    fig = px.area(
        data_frame=f_data,
        x="agg_time",
        y="value",
        color="translated_appliance",
        labels={
            "agg_time": "Time",
            "value": "Usage",
            "translated_appliance": t("appliances")
        },
        category_orders={"translated_appliance": [translate_appliance_name(a) for a in legend_order]},
        color_discrete_map={translate_appliance_name(a): get_appliance_colors()[a] for a in get_appliance_colors()},
        custom_data=["appliance", "formatted_value"]  # for hover info
    )

    # Common layout settings for the plot
    common_layout = dict(
        template="plotly_dark",  # Dark theme
        legend=dict(orientation="h", x=0, y=1.1, xanchor="left", yanchor="bottom"),  # Horizontal legend above plot
        hovermode="x unified",  # Unified hover label on x-axis
        margin=dict(l=40, r=40, t=40, b=40),  # Margins around plot
    )

    # Update figure layout with axis and legend settings
    fig.update_layout(
        **common_layout,
        xaxis=dict(
            title=t('time'),
            tickformat=time_filter[t_filter]["tick_format"],  # Format ticks according to filter
            showgrid=False,
            showline=True,
            tickangle=0
        ),
        yaxis=dict(
            title=f"{t('energy')} [Wh]",
            showgrid=True,
            gridcolor="gray",
            gridwidth=1,
            griddash="dot",  # Dotted grid lines
            tickformat="~s",  # SI unit formatting for ticks
            autorange=True,
        ),
        legend_traceorder=legend_traceorder,  # Legend order based on line_type
    )

    fill_alpha = 0.5  # Transparency for area fill

    # Update each trace's color and fill color with alpha transparency
    for trace in fig.data:
        # Find original appliance name from translated name
        original_name = next((a for a in get_appliance_colors() if translate_appliance_name(a) == trace.name), trace.name)
        color_rgb = get_appliance_colors().get(original_name)
        # Set fill and line colors with transparency
        trace.update(
            fillcolor=rgb_to_rgba(color_rgb, alpha=fill_alpha),
            line=dict(color=color_rgb)
        )

    # Customize hover tooltip and stacking/fill style for traces
    fig.update_traces(
        hovertemplate='%{customdata[1]}',  # Show formatted energy value in hover
        stackgroup=stackgroup,  # Apply stacking if cumulative
        line_shape='spline',    # Smooth curve lines
        fill=fill,              # Fill style depending on stacking
    )

    # Enable range slider on x-axis for easier navigation
    fig.update_xaxes(rangeslider_visible=True)

    # Set y-axis to linear scale and automatic range
    fig.update_yaxes(type='linear', autorange=True)

    # Render Plotly figure in Streamlit, fitting container width
    st.plotly_chart(fig, use_container_width=True)
