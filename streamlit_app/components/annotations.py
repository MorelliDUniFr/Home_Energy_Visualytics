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
    # Load annotations
    annotations = load_annotations()

    # Only initialize input once
    if "annotation_input" not in st.session_state:
        st.session_state.annotation_input = ""

    # Input box
    st.text_area(
        label=t('add_annotation'),
        height=68,
        max_chars=240,
        key="annotation_input",
        placeholder=t('annotation_placeholder'),
    )

    # Submission logic
    annotation_text = st.session_state.annotation_input
    if annotation_text.strip():
        annotations = add_annotation(
            annotations,
            st.session_state.selected_date_1,
            annotation_text,
            st.session_state.time_period
        )
        save_annotations(annotations)
        st.success(t('annotation_saved'))
        st.session_state.pop("annotation_input", None)
        st.rerun()

    # Display annotations
    selected_date = st.session_state.get("selected_date_1")
    if selected_date:
        grouped_annotations = get_grouped_annotations(
            annotations,
            selected_date,
            st.session_state.time_period,
            st.session_state.lang
        )

        period_display_order = ["Year", "Month", "Week", "Day"]
        any_found = False

        for group in period_display_order:
            entries = grouped_annotations.get(group, [])
            if entries:
                any_found = True

                # Group entries by display label, keep date and text for each
                grouped_by_label = {}
                for date_obj, display_label, text in entries:
                    grouped_by_label.setdefault(display_label, []).append((date_obj, text))

                for display_label, items in grouped_by_label.items():
                    with st.expander(display_label):
                        for date_obj, text in items:
                            period = group
                            annotation_id = f"{hash((date_obj, text, period))}"

                            col1, col2 = st.columns([0.95, 0.05], gap="small")

                            with col1:
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
                                if st.button("🗑️", key=f"del_{annotation_id}", help=t('delete_annotation')):
                                    annotations = delete_annotation(annotations, date_obj, text, period)
                                    save_annotations(annotations)
                                    st.rerun()

        if not any_found:
            st.info(t("no_annotations_found"))
