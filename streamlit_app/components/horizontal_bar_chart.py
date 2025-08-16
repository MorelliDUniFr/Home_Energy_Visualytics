import streamlit as st
import plotly.express as px
from utils.translations import t, translate_appliance_name
from utils.formatting import format_value


def plot_horizontal_bar_chart(data, colors, chart_key):
    """
    Horizontal bar chart showing % contribution per appliance.
    Hover displays actual energy usage.
    """

    # Warn and exit early if no data
    if data.empty:
        st.warning(t('warning_message'), icon='ℹ️')
        return

    df = data.copy()

    # Aggregate power values per appliance (sum of 'value' column)
    totals = df.groupby("appliance", as_index=False)["value"].sum()

    # Convert total power from 10-second samples (W) to Wh
    totals["value"] *= 10 / 3600

    # Calculate total energy consumed to get percentages
    total = totals["value"].sum()

    # Calculate % contribution of each appliance
    totals["percentage"] = 100 * totals["value"] / total

    # Create labels for bar text (e.g. "12.3%")
    totals["label"] = totals["percentage"].apply(lambda x: f"{x:.1f}%")

    # Format actual energy values for hover text (e.g. "123 Wh")
    totals["formatted_value"] = totals["value"].apply(format_value, args=("Wh",))

    # Sort appliances ascending by percentage for better horizontal bar display
    totals = totals.sort_values("percentage", ascending=True)

    # Translate appliance names for display
    totals["translated_appliance"] = totals["appliance"].apply(translate_appliance_name)

    # Create Plotly horizontal bar chart
    fig = px.bar(
        totals,
        x="percentage",
        y="translated_appliance",
        orientation="h",
        text="label",  # show percentage on bars
        color="translated_appliance",
        color_discrete_map={translate_appliance_name(name): colors[name] for name in totals["appliance"]},  # map colors
        labels={"percentage": "Percentage", "translated_appliance": t('appliances')},
        title=f"{t('appliance_distribution')}",
    )

    # Add vertical dotted lines every 10% as visual guides
    for x in range(10, 100, 10):
        fig.add_vline(
            x=x,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            opacity=0.2
        )

    # Customize hover tooltip for each appliance showing actual energy usage
    for trace in fig.data:
        orig_name = totals.loc[totals["translated_appliance"] == trace.name, "appliance"].values[0]
        value = totals.loc[totals["translated_appliance"] == trace.name, "formatted_value"].values[0]
        trace.customdata = [[value]]
        trace.hovertemplate = f"{translate_appliance_name(orig_name)}: {value}<extra></extra>"

    # Show percentage labels outside bars for clarity
    fig.update_traces(textposition="outside")

    # Layout settings for axis, grid, margin, and disabling interactions like zoom
    fig.update_layout(
        xaxis=dict(
            title="",
            tickformat=".0f",
            ticksuffix="%",
            showgrid=True,
            gridcolor="gray",
            griddash="dot",
        ),
        yaxis=dict(categoryorder="total ascending"),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        dragmode="zoom",
        xaxis_fixedrange=False,
        yaxis_fixedrange=False,
    )

    # Render the figure in Streamlit with the provided key
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
