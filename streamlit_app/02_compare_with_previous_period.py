from utils import *
import streamlit as st
from pie_chart import plot_pie_chart
from percentage_bar_chart import plot_percentage_bar_chart
from translations import t
from babel.dates import format_date
from datetime import datetime
from session_utils import store_value, load_value

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

        # Selectbox showing translated labels, storing canonical internal keys
        selected_label = st.selectbox(
            t("select_time_period"),
            options=filtered_time_options,  # internal keys like "Day", "Month"
            format_func=lambda x: translated_time_options[x],  # show translated label
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

    default_dates = {
        '_1': date_ranges.get('another_yesterday', data['date'].min()),
        '_2': date_ranges.get('yesterday', data['date'].min()),
    }

    with container_col:
        if st.session_state.time_period == "Day":
            with st.columns([1, 1, 1], gap="small")[1]:
                # Load the persistent value into the widget temporary key (_selected_date_key)
                temp_key = f"_{selected_date_key}"
                if selected_date_key in st.session_state:
                    st.session_state[temp_key] = st.session_state[selected_date_key]
                else:
                    st.session_state[temp_key] = default_dates[side_suffix]

                # Create the date input widget with the temp key and on_change handler to store value persistently
                selected_date = st.date_input(
                    t('select_day'),
                    min_value=data['date'].min(),
                    max_value=date_ranges['yesterday'],
                    value=st.session_state[temp_key],
                    format=DATE_FORMAT,
                    key=temp_key,
                    on_change=lambda k=selected_date_key, tk=temp_key: st.session_state.update(
                        {k: st.session_state[tk]})
                )

                # Now you can use the persistent selected date key safely
                filtered_data = filter_data_by_time(data, "Day", st.session_state[selected_date_key])

        elif st.session_state.time_period == "Month":
            available_years = get_available_years(data)
            index_sel_year, index_sel_month = find_sel_indexes(available_years, st.session_state.get(selected_year_key))

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

                available_months = get_available_months_for_year(data, selected_year)
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
                    # When widget changes, sync persistent key with selected month value
                    selected_label = st.session_state[temp_key]
                    st.session_state[persistent_key] = translated_month_map[selected_label]

                # Create the selectbox widget with the temp key and on_change callback
                st.selectbox(
                    t('select_month'),
                    options=list(translated_month_map.keys()),
                    index=list(translated_month_map.keys()).index(st.session_state[temp_month_key]),
                    key=temp_month_key,
                    on_change=on_month_change
                )

                # selected_month is now the persistent month value
                selected_month = st.session_state[persistent_month_key]

            selected_date = pd.to_datetime(f"{selected_year}-{selected_month}-01").date()
            filtered_data = filter_data_by_time(data, "Month", selected_date)


        elif st.session_state.time_period == "Year":
            available_years = get_available_years(data)
            if len(available_years) == 1:
                st.warning(t("not_enough_data_to_compare"))
                st.stop()
            else:
                with st.columns([1, 1, 1], gap="small")[1]:
                    # Load persistent value to temporary widget key, or set default
                    load_value(selected_year_key, default=available_years[-1])

                    st.selectbox(
                        t('select_year'),
                        available_years,
                        value=st.session_state[f"_{selected_year_key}"],  # set selected value directly
                        key=f"_{selected_year_key}",
                        on_change=store_value,
                        args=(selected_year_key,),
                    )

                    # Sync back to persistent key
                    st.session_state[selected_year_key] = st.session_state[f"_{selected_year_key}"]

                    selected_date = pd.to_datetime(f"{st.session_state[selected_year_key]}-01-01").date()
                    filtered_data = filter_data_by_time(data, "Year", selected_date)

        return selected_date, filtered_data

filtered_date_1, filtered_data_1 = select_and_filter_data('_1', c11)
plot_pie_chart(filtered_data_1, c11, "pie-chart1", colors=appliance_colors)

filtered_date_2, filtered_data_2 = select_and_filter_data('_2', c12)
plot_pie_chart(filtered_data_2, c12, "pie-chart2", colors=appliance_colors)

with st.container(border=True):
    if filtered_data_1.empty or filtered_data_2.empty:
        st.stop()
    else:
        plot_percentage_bar_chart(filtered_data_1, filtered_data_2, colors=appliance_colors, time_period=st.session_state.time_period)
