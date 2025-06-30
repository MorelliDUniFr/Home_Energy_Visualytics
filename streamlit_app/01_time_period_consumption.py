from utils import *
from pie_chart import plot_pie_chart
from line_chart import create_line_chart
from horizontal_bar_chart import plot_horizontal_bar_chart
from translations import t
from babel.dates import format_date
from datetime import datetime

st.title(t('page_1_title'))

c1, c2, c3, _ = st.columns([1.33, 1.33, 1.33, 6])

with st.container():
    with c1:
        with c1:
            # Translated time period options
            translated_labels = {t(key.lower()): key for key in time_filter}
            current_label = t(st.session_state.time_period.lower())
            selected_label = st.selectbox(t('select_time_period'), options=list(translated_labels.keys()),
                                          index=list(translated_labels).index(current_label))
            st.session_state.time_period = translated_labels[selected_label]

        time_period = st.session_state.time_period
        available_years = get_available_years(data)

    # === Day/Week Selection ===
    if time_period in ['Day', 'Week']:
        with c2:
            st.session_state.selected_date_1 = st.session_state.selected_date_1 or time_filter[time_period]['value']
            selected_date = st.date_input(
                t(time_filter[time_period]['input_string']),
                min_value=data['date'].min(),
                max_value=time_filter[time_period]['max_value'],
                value=st.session_state.selected_date_1,
                format=DATE_FORMAT
            )
            st.session_state.selected_date_1 = selected_date
            st.session_state.selected_year_1 = selected_date.year
            st.session_state.selected_month_1 = selected_date.strftime('%B')

    # === Year Selection ===
    elif time_period == 'Year':
        with c2:
            index = 0 if len(available_years) == 1 else len(available_years) - 2
            st.session_state.selected_year_1 = st.session_state.selected_year_1 or available_years[index]
            selected_year = st.selectbox(t('select_year'), available_years, index=available_years.index(st.session_state.selected_year_1))
            st.session_state.selected_year_1 = selected_year
            selected_date = pd.to_datetime(f'{selected_year}-01-01').date()

    # === Month Selection ===
    elif time_period == 'Month':
        index = 0 if len(available_years) == 1 else len(available_years) - 2
        st.session_state.selected_year_1 = st.session_state.selected_year_1 or available_years[index]

        with c3:
            selected_year = st.selectbox(
                t('select_year'),
                available_years,
                index=available_years.index(st.session_state.selected_year_1)
            )
            st.session_state.selected_year_1 = selected_year

        available_months = get_available_months_for_year(data, selected_year)
        index_sel_month = 0 if len(available_months) == 1 else len(available_months) - 2

        month_label_map = {
            format_date(datetime.strptime(m, '%B'), format='LLLL', locale=st.session_state.lang): m
            for m in available_months
        }

        st.session_state.selected_month_1 = st.session_state.selected_month_1 or list(month_label_map.values())[
            index_sel_month]

        with c2:
            selected_label = st.selectbox(
                t("select_month"),
                list(month_label_map.keys()),
                index=list(month_label_map.values()).index(st.session_state.selected_month_1)
            )
            selected_month = month_label_map[selected_label]
            st.session_state.selected_month_1 = selected_month

        selected_date = pd.to_datetime(f'{selected_year}-{selected_month}-01').date()

    # Filter data once after time selection
    filtered_data = filter_data_by_time(data, time_period, selected_date)

if filtered_data.empty:
    st.warning(body=t('warning_message'), icon='ℹ️')
    st.stop()

c11, c12 = st.columns([4, 6], border=True)
with c11:
    chart_options = {
        "bar_chart": t('bar_chart'),
        "pie_chart": t('pie_chart')
    }
    translated_labels = list(chart_options.values())

    # Initialize if missing or invalid
    if 'chart_type' not in st.session_state or st.session_state.chart_type not in chart_options:
        st.session_state.chart_type = 'bar_chart'

    default_index = translated_labels.index(chart_options[st.session_state.chart_type])

    selected_label = st.radio(f'{t("display_mode")}:', translated_labels, horizontal=True, index=default_index)

    # Map back from translated label to internal key
    for key, val in chart_options.items():
        if val == selected_label:
            st.session_state.chart_type = key
            break

    if st.session_state.chart_type == 'bar_chart':
        plot_horizontal_bar_chart(filtered_data, colors=appliance_colors)
    else:
        plot_pie_chart(filtered_data, c11, 'pie_chart', colors=appliance_colors)

with c12:
    fig = create_line_chart(filtered_data, t_filter=time_period)
    st.plotly_chart(fig, use_container_width=True)