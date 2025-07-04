import uuid
import json
from datetime import time
from streamlit import session_state as ss
from utils.translations import t, translate_appliance_name
from utils.session_state_utils import load_value, store_value
from utils.filters import time_filter, date_ranges
from utils.data_loader import get_earliest_date
import pandas as pd
from utils.appliances import appliance_colors, format_appliance_name
import streamlit as st
from utils.data_loader import load_data_by_date_range
from utils.config_utils import inferred_dataset_path, DATE_FORMAT, data_path, models_dir, scalers_dir, model_file, target_scalers_file, color_palette
import os

st.title(t('page_3_title'))

earliest_date = get_earliest_date(inferred_dataset_path)

if 'selected_date_2' in st.session_state:
    st.session_state.selected_date_2 = st.session_state.selected_date_2
else:
    st.session_state.selected_date_2 = date_ranges.get('yesterday', earliest_date)

dataframe = load_data_by_date_range(inferred_dataset_path, st.session_state.selected_date_1, st.session_state.selected_date_2)

# Columns that must always be included in the display
mandatory_columns = ['date', 'timestamp']

# Available columns excluding mandatory ones
available_columns = [col for col in dataframe.columns if col not in mandatory_columns]


def execute_cb():
    # Change dfk key to force refresh of dataframe widget
    ss.dfk = str(uuid.uuid4())


# Dialog for confirming deletion of appliance model and scaler files
@st.dialog(t('dialog_deletion'))
def dialog(name):
    st.write(f"{t('confirm_deletion')}: **{name}**?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ " + t('confirm')):
            base_name = name.replace(' ', '_').lower()
            try:
                os.remove(os.path.join(data_path, models_dir, base_name + model_file))
                os.remove(os.path.join(data_path, scalers_dir, base_name + target_scalers_file))
                st.success(f"{name} {t('deleted')}")
            except Exception as e:
                st.error(f"Error deleting files: {e}")
            st.rerun()
    with col2:
        if st.button("❌ " + t('cancel')):
            execute_cb()
            st.rerun()


# Columns for model/scaler list and upload section
col3, col4 = st.columns([5, 5])

with col3:
    model_files = sorted(os.listdir(os.path.join(data_path, models_dir)))
    scaler_files = sorted(os.listdir(os.path.join(data_path, scalers_dir)))

    appliances_names = [format_appliance_name(f) for f in model_files]
    appliances_names = [translate_appliance_name(name) for name in appliances_names]

    if len(model_files) != len(scaler_files):
        st.warning(t('files_mismatch'))
    else:
        files_df = pd.DataFrame({
            t('appliance'): appliances_names,
            t('model_files'): model_files,
            t('scaler_files'): scaler_files,
        })

        if 'dfk' not in ss:
            ss.dfk = str(uuid.uuid4())

        files_df.index = range(1, len(files_df) + 1)  # 1-based indexing for display

        st.write(t('available_appliances'))
        event = st.dataframe(
            files_df,
            on_select='rerun',
            selection_mode='single-row',
            height=8 * 35,
            key=ss.dfk,
            hide_index=True
        )

        if event['selection']['rows']:
            selected_idx = event['selection']['rows'][0]
            appliance_name = appliances_names[selected_idx]
            dialog(name=appliance_name)

with col4:
    st.write(t('upload'))
    uploaded_files = st.file_uploader(
        label=' ',
        label_visibility='collapsed',
        help="Upload Appliance Model (.pt file) and its corresponding Output Converter (.pkl file)",
        type=['pt', 'pkl'],
        accept_multiple_files=True
    )

if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith('.pt'):
            with open(os.path.join(data_path, models_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        elif file.name.endswith('.pkl'):
            with open(os.path.join(data_path, scalers_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

        appliance_name = file.name.split('.')[0].rsplit('_', 1)[0].replace('_', ' ').title()

        if appliance_name not in appliance_colors:
            # Assign a new color cyclically from palette
            appliance_colors[appliance_name] = color_palette[len(appliance_colors) % len(color_palette)]

            # Update colors json file with alphabetical sorting, 'Other' last
            with open('appliance_colors.json', 'r+') as f:
                data_colors = json.load(f)
                data_colors[appliance_name] = appliance_colors[appliance_name]
                sorted_data = dict(sorted(
                    data_colors.items(),
                    key=lambda item: (item[0].lower() == 'other', item[0].lower())
                ))
                f.seek(0)
                json.dump(sorted_data, f, indent=4)
                f.truncate()
    st.rerun()

st.divider()

if st.session_state.selected_date_1 > st.session_state.selected_date_2:
    st.session_state.selected_date_2 = st.session_state.selected_date_1
    st.session_state['_selected_date_2'] = st.session_state['_selected_date_1']
    load_value("selected_date_2")
    dataframe = load_data_by_date_range(inferred_dataset_path, st.session_state.selected_date_1,
                                        st.session_state.selected_date_2)

# Two columns for selecting columns and appliances to display
col1, col2 = st.columns(2)

# Map translated labels to internal column names
translated_column_labels = {t(col): col for col in available_columns}
translated_column_options = list(translated_column_labels.keys())

with col1:
    if 'selected_columns' not in st.session_state or not st.session_state.selected_columns:
        st.session_state.selected_columns = available_columns.copy()
    selected_translated = st.multiselect(
        t('columns_to_display'),
        options=translated_column_options,
        default=[t(col) for col in st.session_state.selected_columns]
    )
    # Update persistent selected columns with internal column names
    st.session_state.selected_columns = [translated_column_labels[label] for label in selected_translated]
    selected_columns = st.session_state.selected_columns + mandatory_columns

with col2:
    if 'selected_appliances' not in st.session_state or not st.session_state.selected_appliances:
        st.session_state.selected_appliances = dataframe['appliance'].unique().tolist()

    translated_appliances = {appliance: translate_appliance_name(appliance) for appliance in
                             dataframe['appliance'].unique()}
    reverse_mapping = {v: k for k, v in translated_appliances.items()}

    selected_translated = st.multiselect(
        t('appliances_to_display'),
        options=list(translated_appliances.values()),
        default=[translate_appliance_name(a) for a in st.session_state.selected_appliances]
    )

    selected_appliances = [reverse_mapping[label] for label in selected_translated]
    st.session_state.selected_appliances = selected_appliances

    # Filter dataframe by selected appliances
    dataframe = dataframe[dataframe['appliance'].isin(selected_appliances)]

# Date and time filters row
col3, col4, _, col5, col6, col7, _ = st.columns([1.2, 1.2, 2.6, 1.2, 1.2, 1.2, 1.4])

with col3:
    max_date = time_filter['Day']['max_value']
    start_date = st.date_input(
        t('start_date'),
        min_value=earliest_date,
        max_value=max_date,
        format=DATE_FORMAT,
        key='_selected_date_1',
        on_change=store_value,
        args=('selected_date_1',)
    )

with col4:
    max_date = time_filter['Day']['max_value']
    end_date = st.date_input(
        t('end_date'),
        min_value=earliest_date,
        max_value=max_date,
        format=DATE_FORMAT,
        key='_selected_date_2',
        on_change=store_value,
        args=('selected_date_2',)
    )

with col5:
    filter_by_time = st.checkbox(t('time_filter'), value=False)

with col6:
    if filter_by_time:
        start_time = st.time_input(t('start_time'), value=dataframe['timestamp'].min().time())

with col7:
    if filter_by_time:
        end_time = st.time_input(t('end_time'), value=time(23, 59))

# Apply time filtering if enabled
if filter_by_time:
    dataframe = dataframe[(dataframe['timestamp'].dt.time >= start_time) & (dataframe['timestamp'].dt.time <= end_time)]

# Select columns to display, always including mandatory ones
dataframe_display = dataframe[selected_columns]

# Display the filtered dataframe without the index, and dynamic height for 8 rows + header
st.dataframe(dataframe_display, use_container_width=True, hide_index=True, height=35 * (9 + 1))
