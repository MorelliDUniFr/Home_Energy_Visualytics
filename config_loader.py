import os
import configparser

def find_closest_ini(filename='config.ini', start_dir=None):
    if start_dir is None:
        start_dir = os.getcwd()
    current_dir = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(current_dir, filename)
        if os.path.isfile(candidate):
            return candidate
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    return None

def load_config():
    ini_path = find_closest_ini()
    if ini_path is None:
        raise FileNotFoundError("config.ini not found!")
    config = configparser.ConfigParser()
    config.read(ini_path)
    return config, os.path.dirname(ini_path)
