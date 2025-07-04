import os
from config_loader import load_config
import plotly.express as px

config, config_dir = load_config()
env = config['Settings']['environment']
data_path = str(config[env]['data_path'])
appliances_colors_file = config['Data']['appliances_colors_file']
models_dir = str(config['Data']['models_dir'])
scalers_dir = str(config['Data']['scalers_dir'])
model_file = str(config['Data']['model_file'])
target_scalers_file = str(config['Data']['target_scalers_file'])
inferred_dir = 'inferred_data'
inferred_dataset_path = os.path.join(data_path, inferred_dir)

# Define a color palette
color_palette = px.colors.qualitative.Pastel

DATE_FORMAT = "DD.MM.YYYY"
