from utils import *
from pie_chart import plot_pie_chart
from line_chart import create_line_chart
from horizontal_bar_chart import plot_horizontal_bar_chart
from translations import t
from babel.dates import format_date
from datetime import datetime, date
from session_utils import store_value, load_value

st.title(t('page_1_title'))

c1, c2, c3, _ = st.columns([1.33, 1.33, 1.33, 6])

with st.container():
    with c1:
        load_value("time_period", default='Day')
        translated_labels = {key: t(key.lower()) for key in time_filter}
        label_options = list(translated_labels.keys())
        selected_label = st.selectbox(
            t('select_time_period'),
            options=label_options,
            format_func=lambda x: translated_labels[x],
            key='_time_period',
            on_change=store_value,
            args=('time_period',)
        )

        # Use partition-aware available years:
        available_years = get_available_years(dataset_path=inferred_dataset_path)

    if '_time_period' in st.session_state:
        st.session_state['time_period'] = st.session_state['_time_period']

    # === Day or Week Selection ===
    if st.session_state.time_period in ['Day', 'Week']:
        with c2:
            max_date = time_filter[st.session_state.time_period]['max_value']

            load_value('selected_date_1', default=max_date)

            st.date_input(
                t(time_filter[st.session_state.time_period]['input_string']),
                min_value=date_ranges['another_yesterday'],
                max_value=max_date,
                format=DATE_FORMAT,
                key='_selected_date_1',
                on_change=store_value,
                args=('selected_date_1',)
            )

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

            if '_selected_year_1' in st.session_state:
                st.session_state.selected_year_1 = st.session_state['_selected_year_1']

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

            if '_selected_year_1' in st.session_state:
                st.session_state.selected_year_1 = st.session_state['_selected_year_1']

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

            if '_selected_month_1' in st.session_state:
                st.session_state.selected_month_1 = st.session_state['_selected_month_1']

        st.session_state.selected_date_1 = pd.to_datetime(
            f'{st.session_state.selected_year_1}-{st.session_state.selected_month_1}-01'
        ).date()

# === Load filtered data partitions ===
filtered_date_1, filtered_data = load_and_filter_data_by_time(
    inferred_dataset_path,
    st.session_state.time_period,
    '_1'
)

if filtered_data.empty:
    st.warning(body=t('warning_message'), icon='ℹ️')
    st.stop()

c11, c12 = st.columns([4, 6], border=True)

with c11:
    chart_options = {
        "bar_chart": t('bar_chart'),
        "pie_chart": t('pie_chart')
    }

    st.session_state.chart_type = st.session_state.chart_type or 'bar_chart'

    def get_label(chart_key):
        return chart_options[chart_key]

    st.radio(
        label=f'{t("display_mode")}:',
        options=list(chart_options.keys()),
        format_func=get_label,
        horizontal=True,
        key='chart_type'
    )

    if st.session_state.chart_type == 'bar_chart':
        plot_horizontal_bar_chart(filtered_data, colors=appliance_colors)
    else:
        plot_pie_chart(filtered_data, c11, 'pie_chart', colors=appliance_colors)

with c12:
    fig = create_line_chart(filtered_data, t_filter=st.session_state.time_period)
    st.plotly_chart(fig, use_container_width=True)
