from components.pie_chart import plot_pie_chart
from components.line_chart import plot_line_chart
from components.horizontal_bar_chart import plot_horizontal_bar_chart
from utils.translations import t
from babel.dates import format_date
from datetime import datetime
import streamlit as st
from utils.session_state_utils import load_value, store_value
from utils.config_utils import inferred_dataset_path, DATE_FORMAT, consumption_cost
from utils.partition_utils import get_available_years, get_available_months_for_year
from utils.data_loader import get_earliest_date, load_and_filter_data_by_time
from utils.filters import time_filter
from utils.appliances import appliance_colors
from utils.annotations import load_annotations, save_annotations, add_annotation, get_grouped_annotations, delete_annotation
from utils.formatting import format_value
from components.metrics import display_consumption_metrics
from utils.consumption import compute_total_consumption
from utils.data_loader import get_dataset_fingerprint
from utils.data_loader import load_inferred_data_partitioned
from utils.filters import filter_appliances_with_nonzero_sum
import pandas as pd
from components.annotations import plot_annotations

st.title(body=t('page_1_title'), anchor=False)

c1, c2, c3, _, text_column1, text_column2, _ = st.columns([1.33, 1.33, 1.33, 1.5, 1.5, 1.5, 1.5])
text_column = [text_column1, text_column2]

with st.container():
    with c1:
        load_value("time_period", default='Day')
        translated_labels = {key: t(key.lower()) for key in time_filter}
        label_options = list(translated_labels.keys())
        st.selectbox(
            t('select_time_period'),
            options=label_options,
            format_func=lambda x: translated_labels[x],
            key='_time_period',
            on_change=store_value,
            args=('time_period',)
        )

        # Use partition-aware available years:
        available_years = get_available_years(dataset_path=inferred_dataset_path)

    # === Day or Week Selection ===
    if st.session_state.time_period in ['Day', 'Week']:
        with c2:
            earliest_date = get_earliest_date(inferred_dataset_path)
            max_date = time_filter[st.session_state.time_period]['max_value']

            load_value('selected_date_1', default=max_date)

            if max_date < earliest_date:
                max_date = time_filter['Day']['max_value']

            st.date_input(
                t(time_filter[st.session_state.time_period]['input_string']),
                min_value=earliest_date,
                max_value=max_date,
                format=DATE_FORMAT,
                key='_selected_date_1',
                on_change=store_value,
                args=('selected_date_1',)
            )

            if st.session_state.time_period == 'Day':
                end_date = st.session_state.selected_date_1
            else:
                end_date = st.session_state.selected_date_1 + pd.Timedelta(days=6)

            fingerprint = get_dataset_fingerprint(inferred_dataset_path)
            filtered_data = load_inferred_data_partitioned(
                inferred_dataset_path,
                st.session_state.selected_date_1.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                fingerprint=fingerprint,
            )
            filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

        if '_selected_date_1' in st.session_state:
            st.session_state.selected_date_1 = st.session_state['_selected_date_1']
            st.session_state.selected_year_1 = st.session_state.selected_date_1.year
            st.session_state.selected_month_1 = st.session_state.selected_date_1.strftime('%B')

    # === Year Selection ===
    elif st.session_state.time_period == 'Year':
        with c2:
            load_value('selected_year_1',
                       default=available_years[0] if len(available_years) == 1 else available_years[-2])

            st.selectbox(
                t('select_year'),
                available_years,
                key='_selected_year_1',
                on_change=store_value,
                args=('selected_year_1',)
            )

            selected_date = pd.to_datetime(f"{st.session_state.selected_year_1}-01-01").date()
            end_date = pd.to_datetime(f"{st.session_state.selected_year_1}-12-31").date()

            filtered_data = load_inferred_data_partitioned(
                inferred_dataset_path,
                selected_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                fingerprint=get_dataset_fingerprint(inferred_dataset_path)
            )
            filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

    # === Month Selection ===
    elif st.session_state.time_period == 'Month':
        load_value('selected_year_1',
                   default=available_years[0] if len(available_years) == 1 else available_years[-2])

        with c3:
            st.selectbox(
                t('select_year'),
                available_years,
                key='_selected_year_1',
                on_change=store_value,
                args=('selected_year_1',)
            )

        # Use partition-aware months retrieval:
        available_months = get_available_months_for_year(
            dataset_path=inferred_dataset_path,
            selected_year=st.session_state.selected_year_1
        )

        # Map months to localized labels:
        month_label_map = {
            m: format_date(datetime.strptime(m, '%B'), format='LLLL', locale=st.session_state.lang)
            for m in available_months
        }

        load_value('selected_month_1',
                   default=available_months[0 if len(available_months) == 1 else len(available_months) - 2])

        with c2:
            st.selectbox(
                t("select_month"),
                options=available_months,
                format_func=lambda x: month_label_map[x],
                key='_selected_month_1',
                on_change=store_value,
                args=('selected_month_1',)
            )

            selected_date = pd.to_datetime(f"{st.session_state.selected_year_1}-{st.session_state.selected_month_1}-01").date()
            end_date = selected_date + pd.offsets.MonthEnd(0)
            filtered_data = load_inferred_data_partitioned(
                inferred_dataset_path,
                selected_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                fingerprint=get_dataset_fingerprint(inferred_dataset_path)
            )
            filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

if filtered_data.empty:
    st.warning(body=t('warning_message'), icon='ℹ️')
    st.stop()

c11, c12 = st.columns([4, 6], border=True)

with c11:
    chart_options = {
        "bar_chart": t('bar_chart'),
        "pie_chart": t('pie_chart')
    }

    load_value('chart_type', default='bar_chart')

    def get_label(chart_key):
        return chart_options[chart_key]

    st.radio(
        label=f'{t("display_mode")}:',
        options=list(chart_options.keys()),
        format_func=get_label,
        horizontal=True,
        key='_chart_type',
        on_change=store_value,
        args=('chart_type',)
    )

    if st.session_state.chart_type == 'bar_chart':
        plot_horizontal_bar_chart(filtered_data, colors=appliance_colors, chart_key='horizontal_bar_chart1')
    else:
        plot_pie_chart(filtered_data, c11, 'pie_chart', colors=appliance_colors)

with c12:
    view_mode_options = {
        'individual': t('individual'),
        'cumulative': t('cumulative')
    }

    load_value('line_type', 'individual')

    st.radio(
        f"{t('display_mode')}:",
        options=list(view_mode_options.keys()),  # internal keys: ['individual', 'cumulative']
        format_func=lambda x: view_mode_options[x],  # show translated label
        horizontal=True,
        key='_line_type',
        on_change=store_value,
        args=('line_type',)
    )

    plot_line_chart(filtered_data, t_filter=st.session_state.time_period)

st.divider()

plot_annotations()

compute_total_consumption(filtered_data)
display_consumption_metrics(cols=text_column, key_suffix='_1')