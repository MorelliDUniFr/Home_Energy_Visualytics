from utils import *
import streamlit as st
import plotly.express as px
import pandas as pd
from translations import t, translate_appliance_name

def create_line_chart(f_data, t_filter):
    required_columns = ['appliance', 'value', 'timestamp']
    if not all(col in f_data.columns for col in required_columns):
        st.warning(f'Data must contain the following columns: {', '.join(required_columns)}.')
        return None

    f_data['timestamp'] = pd.to_datetime(f_data['timestamp'])

    f_data['energy_Wh'] = f_data['value'] * (10 / 3600) # added
    floor_period = time_filter[t_filter]['floor_period']
    f_data['agg_time'] = f_data['timestamp'].dt.floor(floor_period)

    f_data = (
        f_data
        .groupby(['agg_time', 'appliance'], as_index=False)
        .agg({'energy_Wh': 'sum'})
    )
    f_data = f_data.rename(columns={'energy_Wh': 'value'}) # added

    f_data['formatted_value'] = f_data['value'].apply(format_value, args=('Wh',))

    f_data['translated_appliance'] = f_data['appliance'].apply(translate_appliance_name)

    view_mode_options = {
        'individual': t('individual'),
        'cumulative': t('cumulative')
    }

    # Ensure default is set
    st.session_state.line_type = st.session_state.line_type or 'individual'

    st.radio(
        f"{t('display_mode')}:",
        options=list(view_mode_options.keys()),  # internal keys: ['individual', 'cumulative']
        format_func=lambda x: view_mode_options[x],  # show translated label
        horizontal=True,
        key='line_type'  # stored directly in session_state
    )

    if st.session_state.line_type == 'cumulative':
        legend_order = appliance_order[::-1]
        legend_traceorder = "reversed"
        stackgroup="one"  # Enable cumulative stacking
        fill="tonexty"
    elif st.session_state.line_type == 'individual':
        legend_order = appliance_order
        legend_traceorder = "normal"
        stackgroup=None  # No cumulative stacking
        fill="tozeroy"

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
        color_discrete_map={translate_appliance_name(a): appliance_colors[a] for a in appliance_colors},
        custom_data=["appliance", "formatted_value"]
    )

    common_layout = dict(
        template="plotly_dark",
        legend=dict(orientation="h", x=0, y=1.1, xanchor="left", yanchor="bottom"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    fig.update_layout(
        **common_layout,
        xaxis=dict(
            title=t('time'),
            tickformat=time_filter[t_filter]["tick_format"],
            showgrid=False,
            showline=True,
            tickangle=0
        ),
        yaxis=dict(
            title=f"{t('energy')} [Wh]",
            showgrid=True,
            gridcolor="gray",
            gridwidth=1,
            griddash="dot",
            tickformat="~s",
            autorange=True,
        ),
        legend_traceorder=legend_traceorder,
    )

    fill_alpha = 0.5
    for trace in fig.data:
        original_name = next((a for a in appliance_colors if translate_appliance_name(a) == trace.name), trace.name)
        color_rgb = appliance_colors.get(original_name)
        trace.update(
            fillcolor=rgb_to_rgba(color_rgb, alpha=fill_alpha),
            line=dict(color=color_rgb)
        )

    fig.update_traces(
        hovertemplate='%{customdata[1]}',
        stackgroup=stackgroup,  # Enable cumulative stacking
        line_shape='spline',
        fill=fill,
    )

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(type='linear', autorange=True)

    return fig
