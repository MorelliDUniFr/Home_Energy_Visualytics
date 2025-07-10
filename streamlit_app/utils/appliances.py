import os
import pandas as pd
from .config_utils import data_path, models_dir, scalers_dir, color_palette, appliances_colors_file
from .storage import safe_load_json
import streamlit as st

# ---------------------- Appliance Name Parsing ----------------------

def format_appliance_name(filename: str) -> str:
    """
    Formats a model filename into a readable appliance name.
    Example: 'washing_machine_model.pt' -> 'Washing Machine'
    """
    base = filename.split('.')[0].rsplit('_', 1)[0]
    parts = base.split('_')
    return ' '.join([
        word.upper() if len(word) <= 2 and word.islower() else word.title()
        for word in parts
    ])

def read_appliances_from_files(models_dir_path) -> list[str]:
    """
    Reads appliance model filenames from a directory and formats them into names.
    """
    try:
        model_files = os.listdir(models_dir_path)
    except FileNotFoundError:
        return []

    appliance_names = [format_appliance_name(f) for f in model_files]
    appliance_names.sort(key=lambda x: (x == 'Other', x))  # 'Other' always last
    return appliance_names

# ---------------------- Appliance Colors ----------------------

def generate_appliance_colors(names: list[str], palette: list[str]) -> dict:
    """
    Assigns each appliance a color from the palette.
    """
    colors = {}
    for i, app in enumerate(names):
        if app == 'Other':
            colors[app] = 'rgb(153, 153, 153)'  # grey for 'Other'
        else:
            colors[app] = palette[i % len(palette)]
    return colors

def rgb_to_rgba(rgb_string, alpha=1.0) -> str:
    """
    Converts an 'rgb(r, g, b)' string to an 'rgba(r, g, b, a)' string.
    """
    r, g, b = rgb_string.strip()[4:-1].split(',')
    return f'rgba({r.strip()}, {g.strip()}, {b.strip()}, {alpha})'

def get_or_generate_appliance_colors(
    data_path: str,
    appliances_colors_file: str,
    models_dir: str,
    color_palette: list[str]
) -> dict:
    """
    Returns appliance colors, loading from file or generating and saving if not present.
    """
    color_path = os.path.join(data_path, appliances_colors_file)
    models_path = os.path.join(data_path, models_dir)

    if os.path.exists(color_path):
        return safe_load_json(color_path)

    appliance_names = read_appliances_from_files(models_path)
    colors = generate_appliance_colors(appliance_names, color_palette)

    try:
        with open(color_path, 'w') as f:
            pd.Series(colors).to_json(f, indent=4)
    except IOError:
        pass

    return colors

# ---------------------- Appliance Ordering ----------------------

def get_appliance_order(appliance_colors: dict) -> list[str]:
    """
    Returns appliance order based on the color dictionary keys (with 'Other' last).
    """
    return list(appliance_colors.keys())

# ---------------------- Lazy Getters ----------------------

def get_colors_file_mtime():
    path = os.path.join(data_path, appliances_colors_file)
    return os.path.getmtime(path) if os.path.exists(path) else None

def get_appliance_colors():
    mtime = get_colors_file_mtime()
    @st.cache_resource
    def _load(_mtime):
        return get_or_generate_appliance_colors(
            data_path=data_path,
            appliances_colors_file=appliances_colors_file,
            models_dir=models_dir,
            color_palette=color_palette
        )

    return _load(mtime)

def get_ordered_appliance_list() -> list[str]:
    return get_appliance_order(get_appliance_colors())
