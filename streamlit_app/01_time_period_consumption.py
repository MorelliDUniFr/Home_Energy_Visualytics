from utils import *
from pie_chart import plot_pie_chart
from line_chart import create_line_chart
from horizontal_bar_chart import plot_horizontal_bar_chart
from translations import t
from babel.dates import format_date
from datetime import datetime
from session_utils import store_value, load_value

st.title(t('page_1_title'))

c1, c2, c3, _ = st.columns([1.33, 1.33, 1.33, 6])

with st.container():
    with c1:
        load_value("time_period", default='Day')
        translated_labels = {key: t(key.lower()) for key in time_filter}
        label_options = list(translated_labels.keys())
        selected_label = st.selectbox(t('select_time_period'), on_change=store_value, args=('time_period',), options=label_options, format_func=lambda x: translated_labels[x], key='_time_period')

        available_years = get_available_years(data)

    if '_time_period' in st.session_state:
        st.session_state['time_period'] = st.session_state['_time_period']

    # === Day/Week Selection ===
    if st.session_state.time_period in ['Day', 'Week']:
        with c2:
            # load_value('selected_date_1', default=time_filter[st.session_state.time_period]['value'])
            max_date = time_filter[st.session_state.time_period]['max_value']

            if 'selected_date_1' in st.session_state and st.session_state.selected_date_1 is not None:
                if st.session_state.selected_date_1 > max_date:
                    st.session_state.selected_date_1 = max_date

            load_value('selected_date_1', default=max_date)

            st.date_input(
                t(time_filter[st.session_state.time_period]['input_string']),
                min_value=data['date'].min(),
                max_value=time_filter[st.session_state.time_period]['max_value'],
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

            st.selectbox(t('select_year'), available_years, key='_selected_year_1',
                                         on_change=store_value, args=('selected_year_1',))

            if '_selected_year_1' in st.session_state:
                st.session_state.selected_year_1 = st.session_state['_selected_year_1']
                # st.session_state.selected_date_1 = pd.to_datetime(f"{st.session_state.selected_year_1}-01-01").date()

    # === Month Selection ===
    elif st.session_state.time_period == 'Month':
        load_value('selected_year_1',
                   default=available_years[0] if len(available_years) == 1 else available_years[-2])

        with c3:
            st.selectbox(t('select_year'), available_years, key='_selected_year_1',
                                         on_change=store_value, args=('selected_year_1',))

            if '_selected_year_1' in st.session_state:
                st.session_state.selected_year_1 = st.session_state['_selected_year_1']
                # st.session_state.selected_date_1 = pd.to_datetime(f"{st.session_state.selected_year_1}-01-01").date()

        available_months = get_available_months_for_year(data, st.session_state.selected_year_1)

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

        st.session_state.selected_date_1 = pd.to_datetime(f'{st.session_state.selected_year_1}-{st.session_state.selected_month_1}-01').date()

    # Filter data once after time selection
    filtered_data = filter_data_by_time(data, st.session_state.time_period, st.session_state.selected_date_1)

if filtered_data.empty:
    st.warning(body=t('warning_message'), icon='ℹ️')
    st.stop()

c11, c12 = st.columns([4, 6], border=True)
with c11:
    chart_options = {
        "bar_chart": t('bar_chart'),
        "pie_chart": t('pie_chart')
    }

    # Set default if not initialized yet
    st.session_state.chart_type = st.session_state.chart_type or 'bar_chart'

    # Get the translated label for display
    def get_label(chart_key):
        return chart_options[chart_key]

    st.radio(
        label=f'{t("display_mode")}:',
        options=list(chart_options.keys()),  # ["bar_chart", "pie_chart"]
        format_func=get_label,               # Show translated version
        horizontal=True,
        key='chart_type'                     # Tie widget directly to session_state.chart_type
    )

    # Now just use selected_chart or st.session_state.chart_type
    if st.session_state.chart_type == 'bar_chart':
        plot_horizontal_bar_chart(filtered_data, colors=appliance_colors)
    else:
        plot_pie_chart(filtered_data, c11, 'pie_chart', colors=appliance_colors)

with c12:
    fig = create_line_chart(filtered_data, t_filter=st.session_state.time_period)
    st.plotly_chart(fig, use_container_width=True)