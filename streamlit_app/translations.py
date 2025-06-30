import streamlit as st
import os
import json

# Load configuration
from config_loader import load_config
config, config_dir = load_config()

# Set environment and data paths
env = config['Settings']['environment']
data_path = str(config[env]['data_path'])

# Load translations once
def load_translations(filename):
    with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
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

# Helper function to get translation
def t(key):
    return translations.get(key, {}).get(st.session_state.lang, f"[{key}]")


def translate_appliance_name(name: str) -> str:
    lang = st.session_state.get("lang", "en")
    try:
        return appliance_translations[name][lang]
    except KeyError:
        return name  # fallback to original