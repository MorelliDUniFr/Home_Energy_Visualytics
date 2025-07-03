import uuid
from datetime import time
from utils import *
import streamlit as st
from streamlit import session_state as ss
import json
from translations import t, translate_appliance_name

st.title(t('page_3_title'))
dataframe = data.copy()

# Columns that must always be included
mandatory_columns = ['date', 'timestamp']

# Remove mandatory columns from the options list
available_columns = [col for col in dataframe.columns.tolist() if col not in mandatory_columns]


def execute_cb():
    ss.dfk = str(uuid.uuid4())


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
                # TODO: CHANGE THIS TO A TRANSLATION
            st.rerun()
    with col2:
        if st.button("❌ " + t('cancel')):
            execute_cb()
            st.rerun()


col3, col4 = st.columns([5, 5])
with col3:
    model_files = os.listdir(os.path.join(data_path, models_dir))
    scaler_files = os.listdir(os.path.join(data_path, scalers_dir))

    model_files.sort()
    scaler_files.sort()

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

        files_df.index = range(1, len(files_df) + 1)  # Set index to start from 1

        st.write(t('available_appliances'))
        event = st.dataframe(files_df, on_select='rerun', selection_mode='single-row', height=8*35, key=ss.dfk, hide_index=True) # 35 is the height of each row in pixels (one is the header row)

        if event['selection']['rows']:
            appliance_name = appliances_names[event['selection']['rows'][0]]
            dialog(name=appliance_name)

with col4:
    st.write(t('upload'))
    uploaded_file = st.file_uploader(label=' ', label_visibility='collapsed', help="Upload Appliance Model (.pt file) and its corresponding Output Converter (.pkl file)", type=['pt', 'pkl'], accept_multiple_files=True)

if uploaded_file:
    # Process the uploaded files
    for file in uploaded_file:
        if file.name.endswith('.pt'):
            with open(os.path.join('data', 'models', file.name), "wb") as f:
                f.write(file.getbuffer())

        elif file.name.endswith('.pkl'):
            with open(os.path.join('data', 'scalers', file.name), "wb") as f:
                f.write(file.getbuffer())

        # If the name is not in the color dictionary, add it
        appliance_name = file.name.split('.')[0].rsplit('_', 1)[0].replace('_', ' ').title()

        if appliance_name not in appliance_colors:
            # Assign a new color from the color palette
            appliance_colors[appliance_name] = color_palette[len(appliance_colors) % len(color_palette)]

            # Append the color to the JSON file
            with open('appliance_colors.json', 'r+') as f:
                data = json.load(f)
                data[appliance_name] = appliance_colors[appliance_name]

                # Sort alphabetically, but keep 'Other' at the end
                sorted_data = dict(sorted(
                    data.items(),
                    key=lambda item: (item[0].lower() == 'other', item[0].lower())
                ))

                f.seek(0)
                json.dump(sorted_data, f, indent=4)
                f.truncate()
    st.rerun()

st.divider()

# Create two columns for the top selectors (columns and appliances)
col1, col2 = st.columns(2)

translated_column_labels = {t(col): col for col in available_columns}
translated_column_options = list(translated_column_labels.keys())

with col1:
    if not st.session_state.selected_columns:
        st.session_state.selected_columns = available_columns
    selected_translated = st.multiselect(
        t('columns_to_display'),
        options=translated_column_options,
        default=[t(col) for col in st.session_state.selected_columns],  # Show all translated names by default
    )

    st.session_state.selected_columns = [translated_column_labels[label] for label in selected_translated]
    selected_columns = ss.selected_columns + mandatory_columns

with col2:
    if not st.session_state.selected_appliances:
        st.session_state.selected_appliances = dataframe['appliance'].unique().tolist()
    translated_appliances = {appliance: translate_appliance_name(appliance) for appliance in dataframe['appliance'].unique().tolist()}
    reverse_mapping = {v: k for k, v in translated_appliances.items()}

    selected_translated = st.multiselect(
        t('appliances_to_display'),
        options=list(translated_appliances.values()),
        default=[translate_appliance_name(a) for a in st.session_state.selected_appliances]  # Select all by default
    )

    selected_appliances = [reverse_mapping[label] for label in selected_translated]
    st.session_state.selected_appliances = selected_appliances

    # Filter the dataframe based on selected appliances
    dataframe = dataframe[dataframe['appliance'].isin(selected_appliances)]

# Create a row for date and time filters
col3, col4, _, col5, col6, col7, _ = st.columns([1.2, 1.2, 2.6, 1.2, 1.2, 1.2, 1.4])

# Add date range filter
with col3:
    start_date = st.date_input(t('start_date'), value=dataframe['date'].max() - pd.Timedelta(days=7), min_value=dataframe['date'].min(), max_value=dataframe['date'].max(), format=DATE_FORMAT)

with col4:
    end_date = st.date_input(t('end_date'), value=dataframe['date'].max(), min_value=dataframe['date'].min(), max_value=dataframe['date'].max(), format=DATE_FORMAT)

# Add checkbox for filtering by time
with col5:
    filter_by_time = st.checkbox(t('time_filter'), value=False)

# Add time range filter (only show if checkbox is selected)
with col6:
    if filter_by_time:
        start_time = st.time_input(t('start_time'), value=dataframe['timestamp'].min().time())

with col7:
    if filter_by_time:
        end_time = st.time_input(t('end_time'), value=time(23, 59))

# Filter the dataframe based on date range
dataframe = dataframe[(dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)]

# Filter the dataframe based on time range if checkbox is selected
if filter_by_time:
    dataframe = dataframe[(dataframe['timestamp'].dt.time >= start_time) & (dataframe['timestamp'].dt.time <= end_time)]

# Filter dataframe by selected columns (including mandatory ones)
dataframe_display = dataframe[selected_columns]

# Show the filtered dataframe (only selected columns + date/timestamp if necessary)
st.dataframe(dataframe_display, use_container_width=True, hide_index=True, height=35*(8+1))  # 35 is the height of each row in pixels (one is the header row)
