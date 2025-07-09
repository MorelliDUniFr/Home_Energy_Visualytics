import os
import configparser
import logging
import socket
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('main')

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
        logger.error('config.ini not found!')

    config = configparser.ConfigParser()
    config.read(ini_path)

    validate_config(config)

    return config, os.path.dirname(ini_path)


def validate_config(config):
    required_sections = ['Settings', 'development', 'production', 'ML', 'MQTT', 'Inference', 'Data']
    for section in required_sections:
        if section not in config:
            logger.error(f'Missing required section: {section}')

    required_keys = {
        'Settings': ['environment'],
        'development': ['data_path'],
        'production': ['data_path'],
        'ML': ['path', 'max_epochs', 'patience', 'batch_size'],
        'MQTT': ['broker', 'port', 'topic', 'transfer_timestamp'],
        'Inference': ['inference_timestamp', 'batch_size'],
        'Data': ['models_dir', 'scalers_dir', 'training_dataset_file', 'training_dataset_columns_file', 'model_file', 'input_scaler_file', 'target_scalers_file', 'daily_data_file', 'inferred_data_file', 'infer_data_file', 'demo_dataset_ground_truth_file', 'appliances_colors_file'],
    }

    for section, keys in required_keys.items():
        for key in keys:
            if key not in config[section]:
                logger.error(f'Missing required key "{key}" in section "{section}"')

    # Ensure that the environment is either 'development' or 'production'
    env = config['Settings']['environment']
    if env not in ['development', 'production']:
        logger.error('Environment must be either "development" or "production"')

    # Ensure that the max_epochs is an integer
    max_epochs_str = config['ML']['max_epochs']
    try:
        max_epochs = int(max_epochs_str)
    except ValueError:
        logger.error('Invalid max_epochs: must be an integer')
    else:
        if max_epochs <= 0:
            logger.error('max_epochs must be a positive integer')

    # Ensure that the patience is an integer
    patience_str = config['ML']['patience']
    try:
        patience = int(patience_str)
    except ValueError:
        logger.error('Invalid patience: must be an integer')
    else:
        if patience <= 0:
            logger.error('patience must be a positive integer')

    # Ensure that the batch_size is an integer
    batch_size_str = config['ML']['batch_size']
    try:
        batch_size = int(batch_size_str)
    except ValueError:
        logger.error('Invalid batch_size: must be an integer')
    else:
        if batch_size <= 0:
            logger.error('batch_size must be a positive integer')

    # Ensure that broker is a valid hostname or IP address
    broker = config['MQTT']['broker']
    try:
        socket.inet_pton(socket.AF_INET, broker)
    except OSError:
        logger.error(f'Invalid MQTT broker: \"{broker}\" must be a valid IPv4 address')

    # Ensure that the port is an integer and exactly 1883
    port_str = config['MQTT']['port']
    try:
        port = int(port_str)
    except ValueError:
        logger.error('Invalid MQTT port: must be an integer')
    else:
        if port != 1883:
            logger.error('MQTT port must be 1883')

    # Ensure that the topic is valid
    topic = config['MQTT']['topic']
    if not topic or not isinstance(topic, str):
        logger.error('MQTT topic must be a non-empty string')
    if len(topic) > 65535:
        logger.error('MQTT topic length must not exceed 65535 characters')

    # Ensure that the transfer timestamp is in the correct format
    transfer_timestamp = config['MQTT']['transfer_timestamp']
    try:
        datetime.strptime(transfer_timestamp, '%H:%M')
    except ValueError:
        logger.error('Transfer timestamp must be in the format HH:MM (e.g., 23:45)')

    # Ensure that the inference timestamp is in the correct format
    inference_timestamp = config['Inference']['inference_timestamp']
    try:
        datetime.strptime(inference_timestamp, '%H:%M')
    except ValueError:
        logger.error('Inference timestamp must be in the format HH:MM (e.g., 23:45)')

    # Ensure that the inference time is after the transfer time
    transfer_dt = datetime.strptime(transfer_timestamp, '%H:%M')
    inference_dt = datetime.strptime(inference_timestamp, '%H:%M')
    # Minimum delay of 3 minutes
    min_delay = timedelta(minutes=3)
    if inference_dt - transfer_dt < min_delay:
        logger.error('Inference time must be at least 3 minutes after transfer time')

    # Ensure that the batch_size is an integer
    batch_size_str = config['Inference']['batch_size']
    try:
        batch_size = int(batch_size_str)
    except ValueError:
        logger.error('Invalid batch_size: must be an integer')
    else:
        if batch_size <= 0:
            logger.error('batch_size must be a positive integer')

    # Ensure that the training dataset file is a parquet file
    training_dataset_file = config['Data']['training_dataset_file']
    if not training_dataset_file.endswith('.parquet'):
        logger.error(f'"training_dataset_file" must be a .parquet file, got: {training_dataset_file}')

    # Ensure that the training dataset columns file is a parquet file
    training_dataset_columns_file = config['Data']['training_dataset_columns_file']
    if not training_dataset_columns_file.endswith('.json'):
        logger.error(f'"training_dataset_columns_file" must be a .parquet file, got: {training_dataset_columns_file}')

    # Ensure that the model file is a .pt file
    model_file = config['Data']['model_file']
    if not model_file.endswith('.pt'):
        logger.error(f'Model file must have a .pt extension, got: {model_file}')

    # Ensure that the input scaler file is a .pkl file
    input_scaler_file = config['Data']['input_scaler_file']
    if not input_scaler_file.endswith('.pkl'):
        logger.error(f'Input scaler file must have a .pkl extension, got: {input_scaler_file}')

    # Ensure that the target scalers file is a .pkl file
    target_scalers_file = config['Data']['target_scalers_file']
    if not target_scalers_file.endswith('.pkl'):
        logger.error(f'Target scalers file must have a .pkl extension, got: {target_scalers_file}')

    # Ensure that the daily data file is a parquet file
    daily_data_file = config['Data']['daily_data_file']
    if not daily_data_file.endswith('.parquet'):
        logger.error(f'"daily_data_file" must be a .parquet file, got: {daily_data_file}')

    # Ensure that the inferred data file is a parquet file
    inferred_data_file = config['Data']['inferred_data_file']
    if not inferred_data_file.endswith('.parquet'):
        logger.error(f'"inferred_data_file" must be a .parquet file, got: {inferred_data_file}')

    # Ensure that infer data file is a parquet file
    infer_data_file = config['Data']['infer_data_file']
    if not infer_data_file.endswith('.parquet'):
        logger.error(f'"infer_data_file" must be a .parquet file, got: {infer_data_file}')

    # Ensure that the demo dataset ground truth file is a parquet file
    demo_dataset_ground_truth_file = config['Data']['demo_dataset_ground_truth_file']
    if not demo_dataset_ground_truth_file.endswith('.parquet'):
        logger.error(f'"demo_dataset_ground_truth_file" must be a .parquet file, got: {demo_dataset_ground_truth_file}')

    # Ensure that the appliance colors file is a JSON file
    appliances_colors_file = config['Data']['appliances_colors_file']
    if not appliances_colors_file.endswith('.json'):
        logger.error(f'"appliances_colors_file" must be a .json file, got: {appliances_colors_file}')

    # Ensure that the appliance translation file is a JSON file
    appliance_translation_file = config['Data']['appliance_translation_file']
    if not appliance_translation_file.endswith('.json'):
        logger.error(f'"appliance_translation_file" must be a .json file, got: {appliance_translation_file}')

    # Ensure that the translation file is a JSON file
    translations_file = config['Data']['translations_file']
    if not translations_file.endswith('.json'):
        logger.error(f'"translations_file" must be a .json file, got: {translations_file}')

    # Ensure that the annotations file is a JSON file
    annotations_file = config['Data']['annotations_file']
    if not annotations_file.endswith('.json'):
        logger.error(f'"annotations_file" must be a .json file, got: {annotations_file}')
