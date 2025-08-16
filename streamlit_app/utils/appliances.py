import os
import pandas as pd
from .config_utils import data_path, models_dir, scalers_dir, color_palette, appliances_colors_file
from .storage import safe_load_json
import streamlit as st

# ---------------------- Appliance Name Parsing ----------------------

def format_appliance_name(filename: str) -> str:
    """
    Convert a model filename into a human-readable appliance name.
    Example: 'washing_machine_model.pt' -> 'Washing Machine'

    Steps:
    - Remove file extension.
    - Remove trailing '_model' suffix.
    - Split on underscores.
    - Capitalize words, except for short lowercase acronyms which stay uppercase.
    """
    base = filename.split('.')[0].rsplit('_', 1)[0]  # Strip extension and trailing suffix
    parts = base.split('_')
    return ' '.join([
        word.upper() if len(word) <= 2 and word.islower() else word.title()
        for word in parts
    ])

def read_appliances_from_files(models_dir_path) -> list[str]:
    """
    List model filenames from the directory and convert them to appliance names.

    If the directory doesn't exist, returns an empty list.

    Sorts appliances alphabetically but ensures 'Other' is always last.
    """
    try:
        model_files = os.listdir(models_dir_path)
    except FileNotFoundError:
        return []

    appliance_names = [format_appliance_name(f) for f in model_files]
    # Sort alphabetically but 'Other' always last
    appliance_names.append('Other')
    appliance_names.sort(key=lambda x: (x == 'Other', x))
    return appliance_names

# ---------------------- Appliance Colors ----------------------

def generate_appliance_colors(names: list[str], palette: list[str]) -> dict:
    """
    Assign colors to each appliance from the given color palette.
    'Other' gets a fixed gray color.
    Colors cycle through the palette if fewer colors than appliances.
    """
    colors = {}
    print(names)
    for i, app in enumerate(names):
        if app == 'Other':
            colors[app] = 'rgb(153, 153, 153)'  # fixed grey for 'Other'
        else:
            colors[app] = palette[i % len(palette)]
    return colors

def rgb_to_rgba(rgb_string, alpha=1.0) -> str:
    """
    Convert an 'rgb(r, g, b)' color string to 'rgba(r, g, b, a)' with specified alpha transparency.
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
    Load appliance colors from a JSON file if it exists.
    Otherwise, generate colors based on existing model files, save them, and return.

    Args:
        data_path: Base data directory.
        appliances_colors_file: Filename for colors JSON.
        models_dir: Directory with appliance model files.
        color_palette: List of color strings to assign.

    Returns:
        dict: Appliance names to color mappings.
    """
    color_path = os.path.join(data_path, appliances_colors_file)
    models_path = os.path.join(data_path, models_dir)

    if os.path.exists(color_path):
        return safe_load_json(color_path)

    appliance_names = read_appliances_from_files(models_path)
    colors = generate_appliance_colors(appliance_names, color_palette)

    # Save generated colors for persistence
    try:
        with open(color_path, 'w') as f:
            pd.Series(colors).to_json(f, indent=4)
    except IOError:
        # Ignore save errors, fallback to in-memory colors
        pass

    return colors

# ---------------------- Appliance Ordering ----------------------

def get_appliance_order(appliance_colors: dict) -> list[str]:
    """
    Returns appliance order as a list based on the keys in the colors dictionary.

    The order is maintained as saved (usually alphabetical with 'Other' last).
    """
    return list(appliance_colors.keys())

# ---------------------- Lazy Getters ----------------------

def get_colors_file_mtime():
    """
    Returns the last modified time of the appliance colors JSON file,
    or None if the file doesn't exist.
    """
    path = os.path.join(data_path, appliances_colors_file)
    return os.path.getmtime(path) if os.path.exists(path) else None

def get_appliance_colors():
    """
    Cached loader for appliance colors.
    The cache is invalidated if the colors file modification time changes.

    Returns:
        dict: Appliance colors mapping.
    """
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
    """
    Returns the list of appliance names ordered by the color mapping keys.
    """
    return get_appliance_order(get_appliance_colors())
