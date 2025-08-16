import os
from config_loader import load_config
import plotly.express as px

config, config_dir = load_config()
env = config['Settings']['environment']
data_path = str(config[env]['data_path'])
appliances_colors_file = config['Data']['appliances_colors_file']
models_dir = str(config['Data']['models_dir'])
scalers_dir = str(config['Data']['scalers_dir'])
target_scalers_dir = str(config['Data']['target_scalers_dir'])
model_file = str(config['Data']['model_file'])
target_scalers_file = str(config['Data']['target_scalers_file'])
inferred_dir = str(config['Data']['inferred_dir'])
annotations_file = str(config['Data']['annotations_file'])

inferred_dataset_path = os.path.join(data_path, inferred_dir)
annotations_path = os.path.join(data_path, annotations_file)

# Define a color palette
# color_palette = px.colors.qualitative.Pastel
color_palette = [
    "rgb(174, 198, 207)",  # Pastel blue
    "rgb(255, 179, 71)",   # Pastel orange
    "rgb(179, 158, 181)",  # Pastel purple
    "rgb(119, 221, 119)",  # Pastel green
    "rgb(255, 105, 97)",   # Pastel red
    "rgb(244, 154, 194)",  # Pastel pink
    "rgb(203, 153, 201)",  # Pastel violet
    "rgb(253, 253, 150)",  # Pastel yellow
    "rgb(132, 182, 244)",  # Pastel blue 2
    "rgb(204, 235, 197)",  # Light mint
    "rgb(255, 237, 111)",  # Pale yellow
    "rgb(241, 180, 197)",  # Light rose
    "rgb(222, 203, 228)",  # Lavender
    "rgb(229, 196, 148)",  # Beige
    "rgb(190, 186, 218)",  # Light purple
    "rgb(188, 128, 189)",  # Dusty purple
    "rgb(255, 204, 153)",  # Apricot
    "rgb(180, 151, 231)",  # Periwinkle
    "rgb(255, 222, 173)",  # Navajo white
    "rgb(175, 238, 238)",  # Pale turquoise
    "rgb(152, 251, 152)",  # Pale green
    "rgb(255, 182, 193)"   # Light pink
]

DATE_FORMAT = "DD.MM.YYYY"
