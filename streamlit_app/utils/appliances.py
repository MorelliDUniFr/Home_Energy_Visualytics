import os
from .config_utils import data_path, models_dir, scalers_dir, color_palette, appliances_colors_file
from .storage import safe_load_json
import pandas as pd


def read_appliances_from_files():
    model_files = os.listdir(os.path.join(data_path, models_dir))
    scaler_files = os.listdir(os.path.join(data_path, scalers_dir))

    appliances_names = [format_appliance_name(f) for f in model_files]

    appliances_names.sort(key=lambda x: (x == 'Other', x))

    return appliances_names


def format_appliance_name(filename: str) -> str:
    base = filename.split('.')[0].rsplit('_', 1)[0]
    parts = base.split('_')
    return ' '.join([
        word.upper() if len(word) <= 2 and word.islower() else word.title()
        for word in parts
    ])


def generate_appliance_colors(names):
    colors = {}
    # Assign colors using the original order
    for i, app in enumerate(names):
        if app == 'Other':
            colors[app] = 'rgb(153, 153, 153)'
        else:
            colors[app] = color_palette[i % len(color_palette)]

    return colors


def rgb_to_rgba(rgb_string, alpha=1.0):
    """
    Converts an 'rgb(r, g, b)' string to an 'rgba(r, g, b, alpha)' string.
    """
    # Extract numbers from the rgb string
    values = rgb_string.strip()[4:-1]  # Removes 'rgb(' and ')'
    r, g, b = values.split(',')
    return f'rgba({r.strip()}, {g.strip()}, {b.strip()}, {alpha})'


def get_or_generate_appliance_colors(data_path, appliances_colors_file, models_dir, color_palette):
    if os.path.exists(os.path.join(data_path, appliances_colors_file)):
        appliance_colors = safe_load_json(os.path.join(data_path, appliances_colors_file))
    else:
        appliance_names = read_appliances_from_files()
        appliance_colors = generate_appliance_colors(appliance_names)
        # Save the colors to a JSON file
        with open(os.path.join(data_path, appliances_colors_file), 'w') as f:
            pd.Series(appliance_colors).to_json(f, indent=4)

    return appliance_colors

appliance_colors = get_or_generate_appliance_colors(data_path=data_path, appliances_colors_file=appliances_colors_file,
                                                    models_dir=models_dir, color_palette=color_palette)

def get_appliance_order(appliance_colors):
    """
    Returns the order of appliances based on the keys of the appliance_colors dictionary.
    """
    # Sort by key, with 'Other' last
    # sorted_appliances = sorted(appliance_colors.keys(), key=lambda x: (x.lower() == 'other', x.lower()))
    # return sorted_appliances
    return list(appliance_colors.keys())

appliance_order = get_appliance_order(appliance_colors)