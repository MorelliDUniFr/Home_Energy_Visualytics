import streamlit as st
import os
import json
from config_loader import load_config  # Custom function to load application configuration

# Load configuration only once at startup
config, config_dir = load_config()

# Set environment and resolve the corresponding data path from the config
env = config['Settings']['environment']
data_path = str(config[env]['data_path'])

# Cache translation loading to avoid re-reading files on every Streamlit rerun
@st.cache_data
def load_translations(filename):
    filepath = os.path.join(data_path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Load general UI translations and appliance name translations
translations = load_translations("translations.json")
appliance_translations = load_translations("appliance_translations.json")

# Define the supported UI languages for selection
languages = {
    "en": "English",
    "it": "Italiano",
    "fr": "Français",
    "de": "Deutsch"
}

# Translation function for general UI keys
# Falls back to displaying the key in brackets if translation is missing
def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return translations.get(key, {}).get(lang, f"[{key}]")

# Translation function for appliance names
# Returns the original name if no translation is found
def translate_appliance_name(name: str) -> str:
    lang = st.session_state.get("lang", "en")
    return appliance_translations.get(name, {}).get(lang, name)
