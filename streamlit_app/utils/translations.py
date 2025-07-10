import streamlit as st
import os
import json
from config_loader import load_config

# Load config once
config, config_dir = load_config()

# Set environment and data paths
env = config['Settings']['environment']
data_path = str(config[env]['data_path'])

# Load translations once and cache them to avoid repeated file I/O in Streamlit reruns
@st.cache_data
def load_translations(filename):
    filepath = os.path.join(data_path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations("translations.json")
appliance_translations = load_translations("appliance_translations.json")

# Define available languages
languages = {
    "en": "English",
    "it": "Italiano",
    "fr": "Français",
    "de": "Deutsch"
}

# Helper function to get translation with fallback
def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return translations.get(key, {}).get(lang, f"[{key}]")

def translate_appliance_name(name: str) -> str:
    lang = st.session_state.get("lang", "en")
    return appliance_translations.get(name, {}).get(lang, name)
