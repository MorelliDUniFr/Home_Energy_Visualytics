import json
import os
from babel.dates import format_date
from calendar import monthrange
from datetime import timedelta
from .config_utils import annotations_path


def load_annotations():
    if not os.path.exists(annotations_path):
        return {}
    with open(annotations_path, "r") as f:
        return json.load(f)

def save_annotations(annotations):
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=4)

def add_annotation(annotations, date, text, period):
    y, m, d = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    entry = {"text": text.strip(), "period": period}
    annotations.setdefault(y, {}).setdefault(m, {}).setdefault(d, []).append(entry)
    return annotations

def get_grouped_annotations(annotations, date, selected_period, locale_code='en'):
    grouped = {
        "Year": [],
        "Month": [],
        "Week": [],
        "Day": []
    }

    period_rank = {
        "Day": 0,
        "Week": 1,
        "Month": 2,
        "Year": 3
    }

    def format_for_display(date_obj, saved_period):
        if saved_period == "Day":
            return format_date(date_obj, format="full", locale=locale_code)
        elif saved_period == "Week":
            week_start = date_obj - timedelta(days=date_obj.weekday())
            week_end = week_start + timedelta(days=6)
            return f"{format_date(week_start, format='dd MMM', locale=locale_code)} → {format_date(week_end, format='dd MMM yyyy', locale=locale_code)}"
        elif saved_period == "Month":
            return format_date(date_obj, format="LLLL yyyy", locale=locale_code)
        elif saved_period == "Year":
            return format_date(date_obj, format="yyyy", locale=locale_code)
        else:
            return format_date(date_obj, format="full", locale=locale_code)

    def add(date_obj):
        y, m, d = date_obj.strftime("%Y"), date_obj.strftime("%m"), date_obj.strftime("%d")
        entries = annotations.get(y, {}).get(m, {}).get(d, [])

        for entry in entries:
            entry_period = entry.get("period", "Day")
            if period_rank.get(entry_period, 0) <= period_rank[selected_period]:
                display_key = format_for_display(date_obj, entry_period)
                grouped[entry_period].append((date_obj, display_key, entry["text"]))

    # Traverse calendar based on selected period
    if selected_period == "Day":
        add(date)

    elif selected_period == "Week":
        start_of_week = date - timedelta(days=date.weekday())
        for i in range(7):
            add(start_of_week + timedelta(days=i))

    elif selected_period == "Month":
        days_in_month = monthrange(date.year, date.month)[1]
        for day in range(1, days_in_month + 1):
            try:
                current = date.replace(day=day)
                add(current)
            except ValueError:
                pass

    elif selected_period == "Year":
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    current = date.replace(month=month, day=day)
                    add(current)
                except ValueError:
                    continue

    # Sort each group by real date
    for key in grouped:
        grouped[key] = sorted(
            [(d, label, txt) for d, label, txt in grouped[key]],
            key=lambda x: x[0]
        )

    return grouped