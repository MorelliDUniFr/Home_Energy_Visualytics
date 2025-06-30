from utils import *
import streamlit as st
import plotly.express as px
from translations import t, translate_appliance_name

def plot_horizontal_bar_chart(data, colors):
    """
    Horizontal bar chart showing % contribution per appliance.
    Hover displays actual energy usage.
    """
    df = data.copy()

    # Aggregate and convert to Wh
    totals = df.groupby("appliance", as_index=False)["value"].sum()
    totals["value"] *= 10 / 3600

    # Compute percentage
    total = totals["value"].sum()
    totals["percentage"] = 100 * totals["value"] / total
    totals["label"] = totals["percentage"].apply(lambda x: f"{x:.1f}%")
    totals["formatted_value"] = totals["value"].apply(format_value, args=("Wh",))

    # Sort for visual clarity
    totals = totals.sort_values("percentage", ascending=True)

    totals["translated_appliance"] = totals["appliance"].apply(translate_appliance_name)

    # Plot with color per appliance
    fig = px.bar(
        totals,
        x="percentage",
        y="translated_appliance",
        orientation="h",
        text="label",
        color="translated_appliance",
        color_discrete_map={translate_appliance_name(name): colors[name] for name in totals["appliance"]},
        labels={"percentage": "Percentage", "translated_appliance": t('appliances')},
        title=f"{t('appliance_distribution')} ({t('total_consumption')}: {format_value(total, 'Wh')})",
    )

    # Add subtle vertical guide lines every 10%
    for x in range(10, 100, 10):
        fig.add_vline(
            x=x,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            opacity=0.2
        )

    # Manually assign customdata and hovertemplate for each appliance
    for trace in fig.data:
        orig_name = totals.loc[totals["translated_appliance"] == trace.name, "appliance"].values[0]
        value = totals.loc[totals["translated_appliance"] == trace.name, "formatted_value"].values[0]
        trace.customdata = [[value]]
        trace.hovertemplate = f"{translate_appliance_name(orig_name)}: {value}<extra></extra>"

    fig.update_traces(textposition="outside")

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
        dragmode=False,  # Disables box/lasso/zoom selection
        xaxis_fixedrange=True,  # Prevents zooming/panning on X
        yaxis_fixedrange=True,  # Prevents zooming/panning on Y
    )

    st.plotly_chart(fig, use_container_width=True)
