import streamlit as st
from charts.pie_chart import plot_pie_chart
from charts.percentage_bar_chart import plot_percentage_bar_chart
from utils.translations import t
from babel.dates import format_date
from datetime import datetime, timedelta
from utils.session_state_utils import load_value, store_value
from utils.filters import time_filter, date_ranges, filter_appliances_with_nonzero_sum
from utils.data_loader import get_earliest_date, load_inferred_data_partitioned, get_dataset_fingerprint
from utils.partition_utils import get_available_years, get_available_months_for_year, find_sel_indexes
import pandas as pd
from utils.appliances import appliance_colors
from utils.config_utils import inferred_dataset_path, DATE_FORMAT

st.title(t('page_2_title'))

c1, _ = st.columns([2, 8])
c11, c12 = st.columns([1, 1], border=True)

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
            args=('time_period',)
        )

    if '_time_period' in st.session_state:
        st.session_state['time_period'] = st.session_state['_time_period']

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
            with st.columns([1, 1, 1], gap="small")[1]:
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

            _, col_month, col_year, _ = st.columns([1, 1, 1, 1], gap="small")

            with col_year:
                if st.session_state.get(selected_year_key) is None:
                    st.session_state[selected_year_key] = available_years[index_sel_year]

                selected_year = st.selectbox(
                    t('select_year'),
                    available_years,
                    index=available_years.index(st.session_state[selected_year_key]),
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
                    index=list(translated_month_map.keys()).index(st.session_state[temp_month_key]),
                    key=temp_month_key,
                    on_change=on_month_change
                )

                selected_month = st.session_state[persistent_month_key]

            selected_date = pd.to_datetime(f"{selected_year}-{selected_month}-01").date()
            end_date = selected_date + timedelta(days=29)
            filtered_data = load_inferred_data_partitioned(
                inferred_dataset_path,
                selected_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

        elif st.session_state.time_period == "Year":
            available_years = get_available_years(inferred_dataset_path)
            if len(available_years) == 1:
                st.warning(t("not_enough_data_to_compare"))
                st.stop()
            else:
                with st.columns([1, 1, 1], gap="small")[1]:
                    load_value(selected_year_key, default=available_years[-1])
                    st.selectbox(
                        t('select_year'),
                        available_years,
                        value=st.session_state[f"_{selected_year_key}"],
                        key=f"_{selected_year_key}",
                        on_change=store_value,
                        args=(selected_year_key,)
                    )
                    st.session_state[selected_year_key] = st.session_state[f"_{selected_year_key}"]
                    selected_year = st.session_state[selected_year_key]

                    selected_date = pd.to_datetime(f"{selected_year}-01-01").date()
                    end_date = pd.to_datetime(f"{selected_year}-12-31").date()

                    filtered_data = load_inferred_data_partitioned(
                        inferred_dataset_path,
                        selected_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    filtered_data = filter_appliances_with_nonzero_sum(filtered_data)

        return selected_date, filtered_data

filtered_date_1, filtered_data_1 = select_and_filter_data('_1', c11)
plot_pie_chart(filtered_data_1, c11, "pie-chart1", colors=appliance_colors)

filtered_date_2, filtered_data_2 = select_and_filter_data('_2', c12)
plot_pie_chart(filtered_data_2, c12, "pie-chart2", colors=appliance_colors)

with st.container(border=True):
    if filtered_data_1.empty or filtered_data_2.empty:
        st.stop()
    else:
        plot_percentage_bar_chart(
            filtered_data_1,
            filtered_data_2,
            colors=appliance_colors,
            time_period=st.session_state.time_period
        )

