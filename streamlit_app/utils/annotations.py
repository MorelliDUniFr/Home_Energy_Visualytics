import json
import os
from babel.dates import format_date
from calendar import monthrange
from datetime import timedelta
from .config_utils import annotations_path

# Constants
PERIODS = ["Day", "Week", "Month", "Year"]
PERIOD_RANK = {p: i for i, p in enumerate(PERIODS)}  # Used to compare period hierarchy


# -------------- I/O Functions --------------

def load_annotations():
    """
    Load annotation data from the JSON file.
    Returns:
        dict: nested dictionary of annotations organized by year, month, day.
        Returns empty dict if file does not exist or is invalid.
    """
    if not os.path.exists(annotations_path):
        return {}
    try:
        with open(annotations_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_annotations(annotations):
    """
    Save the given annotations dictionary to the JSON file.
    Any IO errors are silently ignored.
    """
    try:
        with open(annotations_path, "w") as f:
            json.dump(annotations, f, indent=4)
    except IOError:
        pass


# -------------- Core Operations --------------

def add_annotation(annotations, date, text, period):
    """
    Add a new annotation entry for a specific date and period.

    Args:
        annotations (dict): existing annotations dictionary.
        date (datetime.date): the date to add the annotation to.
        text (str): annotation text.
        period (str): one of 'Day', 'Week', 'Month', 'Year'.

    Returns:
        dict: updated annotations dictionary.
    """
    y, m, d = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    entry = {"text": text.strip(), "period": period}
    # Nested dict structure: annotations[year][month][day] is a list of entries
    annotations.setdefault(y, {}).setdefault(m, {}).setdefault(d, []).append(entry)
    return annotations

def delete_annotation(annotations, date_obj, text, period):
    """
    Delete an annotation matching the given text and period on a specific date.

    Args:
        annotations (dict): existing annotations dictionary.
        date_obj (datetime.date): date of annotation to delete.
        text (str): text of annotation to match.
        period (str): period of annotation to match.

    Returns:
        dict: updated annotations dictionary.
    """
    y, m, d = date_obj.strftime("%Y"), date_obj.strftime("%m"), date_obj.strftime("%d")
    day_annotations = annotations.get(y, {}).get(m, {}).get(d, [])

    # Filter out matching annotation(s)
    filtered = [
        ann for ann in day_annotations
        if not (ann["text"] == text.strip() and ann.get("period", "Day") == period)
    ]

    if filtered != day_annotations:
        # Update or remove day entry if empty
        if filtered:
            annotations[y][m][d] = filtered
        else:
            del annotations[y][m][d]
            if not annotations[y][m]:
                del annotations[y][m]
            if not annotations[y]:
                del annotations[y]
    return annotations


# -------------- Annotation Retrieval --------------

def get_grouped_annotations(annotations, date, selected_period, locale_code='en'):
    """
    Retrieve and group annotations relevant to the selected period.

    Args:
        annotations (dict): nested dict of annotations.
        date (datetime.date): reference date.
        selected_period (str): one of 'Day', 'Week', 'Month', 'Year'.
        locale_code (str): locale for date formatting.

    Returns:
        dict: keys are periods ('Day', 'Week', etc.), values are lists of tuples:
              (date_obj, formatted_label, annotation_text)
    """
    grouped = {p: [] for p in PERIODS}

    def format_for_display(date_obj, saved_period):
        # Format the annotation label according to saved_period and locale
        if saved_period == "Day":
            return format_date(date_obj, format="full", locale=locale_code)
        elif saved_period == "Week":
            week_start = date_obj - timedelta(days=date_obj.weekday())
            week_end = week_start + timedelta(days=6)
            return f"{format_date(week_start, 'dd MMM', locale_code)} → {format_date(week_end, 'dd MMM yyyy', locale_code)}"
        elif saved_period == "Month":
            return format_date(date_obj, "LLLL yyyy", locale_code)
        elif saved_period == "Year":
            return format_date(date_obj, "yyyy", locale_code)
        return format_date(date_obj, format="full", locale=locale_code)

    def try_add_annotations_at(date_obj):
        # Add annotations at a specific date if their period is compatible with selected_period
        y, m, d = date_obj.strftime("%Y"), date_obj.strftime("%m"), date_obj.strftime("%d")
        entries = annotations.get(y, {}).get(m, {}).get(d, [])
        for entry in entries:
            entry_period = entry.get("period", "Day")
            # Include only annotations with period rank <= selected_period rank
            if PERIOD_RANK.get(entry_period, 0) <= PERIOD_RANK[selected_period]:
                label = format_for_display(date_obj, entry_period)
                grouped[entry_period].append((date_obj, label, entry["text"]))

    def iterate_dates():
        # Yield all relevant dates in the selected period
        if selected_period == "Day":
            yield date
        elif selected_period == "Week":
            start_of_week = date - timedelta(days=date.weekday())
            for i in range(7):
                yield start_of_week + timedelta(days=i)
        elif selected_period == "Month":
            days_in_month = monthrange(date.year, date.month)[1]
            for day in range(1, days_in_month + 1):
                try:
                    yield date.replace(day=day)
                except ValueError:
                    continue
        elif selected_period == "Year":
            for month in range(1, 13):
                for day in range(1, 32):
                    try:
                        yield date.replace(month=month, day=day)
                    except ValueError:
                        continue

    # Iterate all relevant dates, collect annotations
    for day in iterate_dates():
        try_add_annotations_at(day)

    # Sort annotations in each group by date
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: x[0])

    return grouped
