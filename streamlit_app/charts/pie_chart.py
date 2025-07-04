import streamlit as st
import plotly.express as px
from utils.translations import t, translate_appliance_name
from utils.appliances import appliance_order
from utils.formatting import format_value


def plot_pie_chart(f_data, column, chart_key, colors):
    if f_data.empty:
        with column:
            st.warning(body=t('warning_message'), icon='ℹ️')
        return None

    required_columns = ['appliance', 'value', 'date']
    if not all(col in f_data.columns for col in required_columns):
        with column:
            st.warning(f"Data must contain the following columns: {', '.join(required_columns)}.")
        return None

    # Get unique appliances from the filtered data
    unique_appliances = f_data['appliance'].unique()

    # Create a filtered appliance order that keeps only the appliances present in the data
    filtered_appliance_order = [appliance for appliance in appliance_order if appliance in unique_appliances]

    sample_interval = 10  # seconds

    # Now group and reindex using this filtered appliance order
    appliances_consumption = (
        f_data.groupby('appliance', observed=False)['value'].sum()
        .reindex(filtered_appliance_order, fill_value=0)
    )

    appliances_consumption *= sample_interval / 3_600

    if appliances_consumption.empty:
        with column:
            st.warning(f"No appliance consumption data available.")
        return None

    translated_names = [translate_appliance_name(name) for name in appliances_consumption.index]

    fig_pie = px.pie(
        names=translated_names,
        values=appliances_consumption.values,
        title=f"{t('appliance_distribution')} ({t('total_consumption')}: {format_value(appliances_consumption.sum(), 'Wh')})",
        color=translated_names,
        color_discrete_map={translate_appliance_name(name): colors[name] for name in appliances_consumption.index},
        category_orders={"names": translated_names},
        hole=0.25,
    )

    # Update hover template to show formatted consumption
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
