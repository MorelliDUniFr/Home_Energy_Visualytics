import streamlit as st
from components.pie_chart import plot_pie_chart
from components.vertical_bar_chart import plot_vertical_bar_chart
from utils.translations import t
from babel.dates import format_date
from datetime import datetime
from utils.session_state_utils import load_value, store_value
from utils.filters import time_filter, date_ranges, filter_appliances_with_nonzero_sum
from utils.data_loader import get_earliest_date, load_inferred_data_partitioned, get_dataset_fingerprint
from utils.partition_utils import get_available_years, get_available_months_for_year, find_sel_indexes
import pandas as pd
from utils.appliances import get_appliance_colors
from utils.config_utils import inferred_dataset_path, DATE_FORMAT
from components.horizontal_bar_chart import plot_horizontal_bar_chart
from components.metrics import display_consumption_metrics
from utils.consumption import compute_total_consumptions
from utils.appliances import get_appliance_colors


def select_and_filter_data(side_suffix: str, container_col):
    selected_date_key = f'selected_date{side_suffix}'
    selected_year_key = f'selected_year{side_suffix}'
    selected_month_key = f'selected_month{side_suffix}'

    earliest_date = get_earliest_date(inferred_dataset_path)
    default_dates = {
        '_1': earliest_date,
        '_2': date_ranges.get('yesterday', earliest_date),
    }

    with container_col:
        if st.session_state.time_period == "Day":
            col_selector, _, col_metric1, col_metric2 = st.columns([0.8, 0.1, 1.05, 1.05], gap="small")
            col_metrics = [col_metric1, col_metric2]

            with col_selector:
                temp_key = f"_{selected_date_key}"
                if selected_date_key in st.session_state:
                    st.session_state[temp_key] = st.session_state[selected_date_key]
                else:
                    st.session_state[temp_key] = default_dates[side_suffix]

                selected_date = st.date_input(
                    t('select_day'),
                    min_value=earliest_date,
                    max_value=date_ranges['yesterday'],
                    format=DATE_FORMAT,
                    key=temp_key,
                    on_change=lambda k=selected_date_key, tk=temp_key: st.session_state.update(
                        {k: st.session_state[tk]})
                )

                fingerprint = get_dataset_fingerprint(inferred_dataset_path)
                filtered_data = load_inferred_data_partitioned(
                    inferred_dataset_path,
                    selected_date.strftime("%Y-%m-%d"),
                    selected_date.strftime("%Y-%m-%d"),
                    fingerprint=fingerprint,
                )
                filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

        elif st.session_state.time_period == "Month":
            available_years = get_available_years(inferred_dataset_path)
            index_sel_year, index_sel_month = find_sel_indexes(available_years, st.session_state.get(selected_year_key), inferred_dataset_path)

            col_month, col_year, col_metric1, col_metric2 = st.columns([0.8, 0.8, 1.2, 1.2], gap="small")
            col_metrics = [col_metric1, col_metric2]

            with col_year:
                if st.session_state.get(selected_year_key) is None:
                    st.session_state[selected_year_key] = available_years[index_sel_year]

                selected_year = st.selectbox(
                    t('select_year'),
                    available_years,
                    key=f"select_box_year{side_suffix}"
                )
                st.session_state[selected_year_key] = selected_year

                available_months = get_available_months_for_year(inferred_dataset_path, selected_year)
                index_sel_month = 0 if len(available_months) == 1 else len(available_months) - 2

            with col_month:
                translated_month_map = {
                    format_date(datetime.strptime(m, '%B'), format='LLLL', locale=st.session_state.lang): m
                    for m in available_months
                }

                temp_month_key = f"_select_box_month{side_suffix}"
                persistent_month_key = selected_month_key

                # Load persistent value into temp widget key
                if persistent_month_key in st.session_state:
                    # Find translated label corresponding to the persistent value
                    persistent_value = st.session_state[persistent_month_key]
                    for label, month_val in translated_month_map.items():
                        if month_val == persistent_value:
                            st.session_state[temp_month_key] = label
                            break
                    else:
                        # fallback to default index_sel_month if no match
                        st.session_state[temp_month_key] = list(translated_month_map.keys())[index_sel_month]
                else:
                    # Initialize if missing
                    st.session_state[persistent_month_key] = list(translated_month_map.values())[index_sel_month]
                    st.session_state[temp_month_key] = list(translated_month_map.keys())[index_sel_month]

                def on_month_change(persistent_key=persistent_month_key, temp_key=temp_month_key):
                    st.session_state[persistent_key] = translated_month_map[st.session_state[temp_key]]

                st.selectbox(
                    t('select_month'),
                    options=list(translated_month_map.keys()),
                    key=temp_month_key,
                    on_change=on_month_change
                )

                selected_month = st.session_state[persistent_month_key]

            selected_date = pd.to_datetime(f"{selected_year}-{selected_month}-01").date()
            end_date = selected_date + pd.offsets.MonthEnd(0)
            filtered_data = load_inferred_data_partitioned(
                inferred_dataset_path,
                selected_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                fingerprint=get_dataset_fingerprint(inferred_dataset_path)
            )
            filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

        elif st.session_state.time_period == "Year":
            available_years = get_available_years(inferred_dataset_path)
            if len(available_years) == 1:
                st.warning(t("not_enough_data_to_compare"), icon='ℹ️')
                st.stop()
            else:
                col_selector, _, col_metric1, col_metric2 = st.columns([0.8, 0.1, 1.05, 1.05], gap="small")
                col_metrics = [col_metric1, col_metric2]
                with col_selector:
                    load_value(selected_year_key, default=available_years[-1])
                    st.selectbox(
                        t('select_year'),
                        available_years,
                        value=st.session_state[f"_{selected_year_key}"],
                        key=f"_{selected_year_key}",
                        on_change=store_value,
                        args=(selected_year_key,)
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

        return selected_date, filtered_data, col_metrics

st.title(body=t('page_2_title'), anchor=False)

earliest_date = get_earliest_date(inferred_dataset_path)
if earliest_date is None:
    st.info(t('user_too_early'))
    st.stop()

c1, _ = st.columns([1.33, 8.66])


with st.container():
    with c1:
        load_value("time_period", default='Day')
        if st.session_state.time_period == 'Week':
            st.session_state['_time_period'] = 'Day'
            st.session_state['time_period'] = 'Day'

        excluded_option = "Week"
        filtered_time_options = [key for key in time_filter.keys() if key != excluded_option]

        translated_time_options = {
            key: t(key.lower()) for key in filtered_time_options
        }

        selected_label = st.selectbox(
            t("select_time_period"),
            options=filtered_time_options,
            format_func=lambda x: translated_time_options[x],
            key='_time_period',
            on_change=store_value,
            args=('time_period',),
        )

    if st.session_state.time_period == "Year":
        available_years = get_available_years(inferred_dataset_path)
        if len(available_years) <= 1:
            st.warning(t("not_enough_years_to_compare"), icon='ℹ️')
            st.stop()
    if st.session_state.time_period == "Month":
        [available_year] = get_available_years(inferred_dataset_path)
        available_months = get_available_months_for_year(inferred_dataset_path, available_year)
        if len(available_months) <= 1:
            st.warning(t("not_enough_months_to_compare"), icon='ℹ️')
            st.stop()
    if st.session_state.time_period == "Day" or st.session_state.time_period == "Week":
        earliest_date = get_earliest_date(inferred_dataset_path)
        max_date = time_filter[st.session_state.time_period]['max_value']
        if max_date == earliest_date:
            st.warning(t("not_enough_days_to_compare"), icon='ℹ️')
            st.stop()

    c11, c12 = st.columns([1, 1], border=True)

    filtered_date_1, filtered_data_1, col_metrics1 = select_and_filter_data('_1', c11)
    load_value('chart_type', default='bar_chart')
    if st.session_state.chart_type == 'bar_chart':
        with c11:
            plot_horizontal_bar_chart(filtered_data_1, colors=get_appliance_colors(), chart_key='horizontal_bar_chart1')
    else:
        plot_pie_chart(filtered_data_1, c11, 'pie_chart1', colors=get_appliance_colors())

    filtered_date_2, filtered_data_2, col_metrics2 = select_and_filter_data('_2', c12)

    compute_total_consumptions(filtered_data_1, filtered_data_2)
    display_consumption_metrics(col_metrics1, key_suffix='_1', is_comparison=True)
    display_consumption_metrics(col_metrics2, key_suffix='_2', is_comparison=True)

    if st.session_state.chart_type == 'bar_chart':
        with c12:
            plot_horizontal_bar_chart(filtered_data_2, colors=get_appliance_colors(), chart_key='horizontal_bar_chart2')
    else:
        plot_pie_chart(filtered_data_2, c12, 'pie_chart2', colors=get_appliance_colors())

    with st.container(border=True):
        if filtered_data_1.empty or filtered_data_2.empty:
            st.stop()
        else:
            plot_vertical_bar_chart(
                filtered_data_1,
                filtered_data_2,
                colors=get_appliance_colors(),
                time_period=st.session_state.time_period
            )
