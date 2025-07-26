import streamlit as st
import plotly.express as px
from utils.translations import t, translate_appliance_name
from utils.appliances import get_ordered_appliance_list
from utils.formatting import format_value


def plot_pie_chart(f_data, column, chart_key, colors):
    """
    Plot a pie chart of appliance energy consumption distribution.

    Args:
        f_data (pd.DataFrame): DataFrame with columns ['appliance', 'value', 'date'] and raw power values.
        column (st.delta_generator): Streamlit container (e.g., st.container() or st.sidebar) to place the chart in.
        chart_key (str): Unique Streamlit key for the plotly chart widget.
        colors (dict): Mapping appliance name -> color string (e.g. 'Washing Machine' -> 'rgb(...)').

    Returns:
        plotly.graph_objects.Figure or None: The pie chart figure, or None if no valid data.
    """

    # Check for empty DataFrame early
    if f_data.empty:
        with column:
            st.warning(t('warning_message'), icon='ℹ️')
        return None

    required_columns = ['appliance', 'value', 'date']
    if not all(col in f_data.columns for col in required_columns):
        with column:
            st.warning(f"Data must contain columns: {', '.join(required_columns)}.", icon='ℹ️')
        return None

    # Filter appliances present in the data according to the preferred order
    unique_appliances = f_data['appliance'].unique()
    filtered_order = [app for app in get_ordered_appliance_list() if app in unique_appliances]

    sample_interval_sec = 10  # seconds between samples

    # Aggregate total power values per appliance, reindex to ensure consistent order and fill missing with 0
    appliances_consumption = (
        f_data.groupby('appliance', observed=False)['value'].sum()
        .reindex(filtered_order, fill_value=0)
    )

    # Convert power readings to Wh (assuming 'value' is in watts over 10 seconds intervals)
    appliances_consumption = appliances_consumption * sample_interval_sec / 3600

    if appliances_consumption.empty:
        with column:
            st.warning("No appliance consumption data available.", icon='ℹ️')
        return None

    # Translate appliance names for display
    translated_names = [translate_appliance_name(name) for name in appliances_consumption.index]

    # Create the pie chart with color mapping from appliance names to colors
    fig_pie = px.pie(
        names=translated_names,
        values=appliances_consumption.values,
        title=t('appliance_distribution'),
        color=translated_names,
        color_discrete_map={translate_appliance_name(name): colors[name] for name in appliances_consumption.index},
        category_orders={"names": translated_names},
        hole=0.25,
    )

    # Customize hover and text formatting
    fig_pie.update_traces(
        textinfo='percent',
        textfont=dict(size=14, weight='bold'),
        hovertemplate='%{label}: %{customdata}',
        customdata=[format_value(val, 'Wh') for val in appliances_consumption.values],
        texttemplate='%{percent:.1%}',
    )

    with column:
        st.plotly_chart(fig_pie, use_container_width=True, key=chart_key)

    return fig_pie
