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
        raise FileNotFoundError('config.ini not found!')

    config = configparser.ConfigParser()
    config.read(ini_path)

    validate_config(config)

    return config, os.path.dirname(ini_path)


def validate_config(config):
    # Validation step
    # Ensure that the required sections and keys exist
    required_sections = ['Settings', 'development', 'production', 'ML', 'MQTT', 'Inference']
    for section in required_sections:
        if section not in config:
            raise ValueError(f'Missing required section: {section}')

    required_keys = {
        'Settings': ['environment'],
        'development': ['data_path'],
        'production': ['data_path'],
        'ML': ['path'],
        'MQTT': ['broker', 'port', 'topic', 'transfer_timestamp'],
        'Inference': ['inference_timestamp'],
        'Data': ['training_dataset_file', 'model_file', 'input_scaler_file', 'target_scalers_file', 'daily_data_file', 'whole_data_file', 'inferred_data_file', 'infer_data_file']
    }

    for section, keys in required_keys.items():
        for key in keys:
            if key not in config[section]:
                raise ValueError(f'Missing required key "{key}" in section "{section}"')

    # Ensure that the environment is either 'development' or 'production'
    env = config['Settings']['environment']
    if env not in ['development', 'production']:
        raise ValueError('Environment must be either "development" or "production"')

    # Ensure that the port is an integer and 1883
    try:
        port = int(config['MQTT']['port'])
        if port != 1883:
            raise ValueError('MQTT port must be 1883')
    except ValueError as e:
        raise ValueError('Invalid MQTT port: must be an integer') from e

    # Ensure that the transfer timestamp is in the correct format
    transfer_timestamp = config['MQTT']['transfer_timestamp']
    try:
        from datetime import datetime
        datetime.strptime(transfer_timestamp, '%H:%M')
    except ValueError:
        raise ValueError('Transfer timestamp must be in the format HH:MM')

    # Ensure that the data path exists
    data_path = config[env]['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data path "{data_path}" does not exist')

    # Ensure that the model file exists
    model_file = config['Data']['model_file']
    model_path = os.path.join(data_path, model_file)

    # Ensure that the daily data file is a parquet file
    daily_data_file = config['Data']['daily_data_file']
    if not daily_data_file.endswith('.parquet'):
        raise ValueError(f'"daily_data_file" must be a .parquet file, got: {daily_data_file}')

    # Ensure that the whole data file is a parquet file
    whole_data_file = config['Data']['whole_data_file']
    if not whole_data_file.endswith('.parquet'):
        raise ValueError(f'"whole_data_file" must be a .parquet file, got: {whole_data_file}')

    # Ensure that the training dataset file exists
    if env == 'development':
        training_dataset_file = config['Data']['training_dataset_file']
        training_dataset_path = os.path.join(data_path, training_dataset_file)
        # if not os.path.isfile(training_dataset_path):
            # raise FileNotFoundError(
                # f'Training dataset file "{training_dataset_file}" does not exist in data path "{data_path}"')

    # Ensure that the MQTT broker is a valid URL
    import re
    broker = config['MQTT']['broker']
    if not re.match(r'^[a-zA-Z0-9.-]+$', broker):
        raise ValueError(f'Invalid MQTT broker: "{broker}" must be a valid hostname or IP address')

    # Ensure that the MQTT topic is not empty
    topic = config['MQTT']['topic']
    if not topic:
        raise ValueError('MQTT topic cannot be empty')

    # Ensure that the inference timestamp is in the correct format
    inference_timestamp = config['Inference']['inference_timestamp']
    try:
        datetime.strptime(inference_timestamp, '%H:%M')
    except ValueError:
        raise ValueError('Inference timestamp must be in the format HH:MM')

    # Ensure that the inference timestamp is after the transfer timestamp
    transfer_time = datetime.strptime(transfer_timestamp, '%H:%M')
    inference_time = datetime.strptime(inference_timestamp, '%H:%M')
    if inference_time <= transfer_time:
        raise ValueError('Inference timestamp must be after the transfer timestamp')

    # Ensure that the model file is a .pt file
    if not model_file.endswith('.pt'):
        raise ValueError('Model file must have a .pt extension')

    # Ensure that the inferred data file is a .parquet file
    inferred_data_file = config['Data']['inferred_data_file']
    if not inferred_data_file.endswith('.parquet'):
        raise ValueError('Inferred data file must have a .parquet extension')

    # Ensure that the infer data file is a .parquet file
    infer_data_file = config['Data']['infer_data_file']
    if not infer_data_file.endswith('.parquet'):
        raise ValueError('Infer data file must have a .parquet extension')

    # Ensure that the daily data file is a .parquet file
    if not daily_data_file.endswith('.parquet'):
        raise ValueError('Daily data file must have a .parquet extension')

    # Ensure that the whole data file is a .parquet file
    if not whole_data_file.endswith('.parquet'):
        raise ValueError('Whole data file must have a .parquet extension')

    # Ensure that the training dataset file is a .parquet file
    if env == 'development':
        if not training_dataset_file.endswith('.parquet'):
            raise ValueError('Training dataset file must have a .parquet extension')
