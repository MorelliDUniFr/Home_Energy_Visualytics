import streamlit as st
from utils.translations import t
from utils.annotations import (
    load_annotations,
    save_annotations,
    add_annotation,
    get_grouped_annotations,
    delete_annotation
)
from utils.consumption import compute_total_consumption
from utils.formatting import format_value


def plot_annotations():
    # Load all saved annotations from the JSON file
    annotations = load_annotations()

    # Initialize the input text area only once to preserve state
    if "annotation_input" not in st.session_state:
        st.session_state.annotation_input = ""

    # Text area for users to add new annotations
    st.text_area(
        label=t('add_annotation'),      # Localized label for the input
        height=68,                      # Fixed height for the input box
        max_chars=240,                  # Max allowed characters in annotation
        key="annotation_input",         # Streamlit state key for input content
        placeholder=t('annotation_placeholder'),  # Placeholder text
    )

    # Handle submission: if the input is non-empty, add new annotation
    annotation_text = st.session_state.annotation_input
    if annotation_text.strip():
        # Add annotation for the selected date and period
        annotations = add_annotation(
            annotations,
            st.session_state.selected_date_1,
            annotation_text,
            st.session_state.time_period
        )
        # Save updated annotations back to disk
        save_annotations(annotations)
        st.success(t('annotation_saved'))  # Feedback message
        # Clear input box after saving
        st.session_state.pop("annotation_input", None)
        # Rerun to refresh UI and show updated list
        st.rerun()

    # Display existing annotations for the selected date and period
    selected_date = st.session_state.get("selected_date_1")
    if selected_date:
        # Group annotations relevant to the selected date and period, localized
        grouped_annotations = get_grouped_annotations(
            annotations,
            selected_date,
            st.session_state.time_period,
            st.session_state.lang
        )

        # Define the order of periods to display annotations by (from largest to smallest)
        period_display_order = ["Year", "Month", "Week", "Day"]
        any_found = False  # Flag to detect if any annotations exist

        for group in period_display_order:
            entries = grouped_annotations.get(group, [])
            if entries:
                any_found = True

                # Group annotations by their display label (e.g. formatted date/week/month/year)
                grouped_by_label = {}
                for date_obj, display_label, text in entries:
                    grouped_by_label.setdefault(display_label, []).append((date_obj, text))

                # For each group label, create an expander UI block
                for display_label, items in grouped_by_label.items():
                    with st.expander(display_label):
                        for date_obj, text in items:
                            period = group
                            # Create a stable unique ID for annotation deletion buttons
                            annotation_id = f"{hash((date_obj, text, period))}"

                            # Layout: main column for text, small column for delete button
                            col1, col2 = st.columns([0.95, 0.05], gap="small")

                            with col1:
                                # Display annotation text with styled container
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: #22222A;
                                        border: 1px solid #3D3E43;
                                        border-radius: 0.5em;
                                        padding: 0 1em;
                                        margin: 0em 0;
                                        font-size: 0.92em;
                                        height: 40px;
                                        display: flex;
                                        align-items: center;
                                    ">
                                        {text}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with col2:
                                # Delete button for annotation with a trash icon
                                if st.button("🗑️", key=f"del_{annotation_id}", help=t('delete_annotation')):
                                    # Remove the annotation from the data structure
                                    annotations = delete_annotation(annotations, date_obj, text, period)
                                    # Save updated annotations back to disk
                                    save_annotations(annotations)
                                    # Refresh the UI to reflect deletion
                                    st.rerun()

        # Show info message if no annotations found for the selected date and period
        if not any_found:
            st.info(t("no_annotations_found"))
