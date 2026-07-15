import config as config
import csv
import glob
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import sys

from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


"""-------------------------------
    Logging
----------------------------------"""
def set_logger(new_logger):
    config.logger = new_logger

def get_logger():
    return config.logger

def init_logging(log_file, verbose=False, debug=False, quiet=False):
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    logger = logging.getLogger(config.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers if any (prevents double-logging)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set Log Levels
    if quiet:
        console_level = logging.CRITICAL
    elif verbose:
        console_level = logging.INFO
    elif debug:
        console_level = logging.DEBUG
    else:
        console_level = logging.WARNING
    file_level = logging.DEBUG

    # Set format
    file_format = logging.Formatter(config.LOG_FILE_FORMAT)
    if debug and verbose:
        console_format = file_format
    else:
        console_format = logging.Formatter(config.LOG_CONSOLE_FORMAT)

    # Set handlers
    console_handler = logging.StreamHandler(sys.stdout)

    # Rotate log files after 2MB, keep 200MB worth of logs
    file_handler = RotatingFileHandler(
        log_file, maxBytes=config.LOG_MAX_SIZE_BYTES, backupCount=config.LOG_MAX_BACKUPS, encoding="utf-8"
    )

    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logger.propagate = False
    logger.debug(f"\n\n==================== 🗃 {config.TIMESTAMP} 🗃 ====================\n")
    logger.debug("\t✔️  Successfully set up logger. ")
    logger.debug(f"\tConsole level: {logging.getLevelName(console_level)}")
    logger.debug(f"\tFile level: {logging.getLevelName(file_level)}")

    return logger


def check_log_rotation_limits(log_file):
    if config.LOG_MAX_WARN_THRESHOLD == -1:
        config.logger.debug("Skipping check of max log backups. After LOG_MAX_BACKUPS, oldest logs will be overwritten without warning.")
        return # Max log warning is suppressed

    log_dir = os.path.dirname(log_file)
    base_name = os.path.basename(log_file)

    # Matches files like DataPipeline.log.1, DataPipeline.log.250, etc.
    pattern = os.path.join(log_dir, f"{base_name}.*")
    existing_backups = glob.glob(pattern)

    # Filter to ensure we only count the extensions that are digits
    rotation_count = sum(1 for f in existing_backups if f.split('.')[-1].isdigit())



    if rotation_count >= config.LOG_MAX_BACKUPS:
        config.logger.warn(f"\nWARNING: Max log files reached. Oldest log files are being overwritten at:\n{log_dir}")
    elif rotation_count >= config.LOG_MAX_WARN_THRESHOLD:
        config.logger.warn(f"\nWARNING: Current backup file count: {rotation_count} / {max_backups}")
        config.logger.warn(f"After reaching {config.LOG_MAX_BACKUPS} the oldest logs will be permanently overwritten.")
        config.logger.warn(f"If you wish to preserve your older log files, please back them up:\n{log_dir}")
    else:
        config.logger.debug(f"Current backup log file count: {rotation_count} / {config.LOG_MAX_BACKUPS}")
        config.logger.debug(f"Backup log warning threshold: {config.LOG_MAX_WARN_THRESHOLD}")
        return

    config.logger.warn("To disable this warning, set LOG_MAX_WARN_THRESHOLD in config.py to -1\n")


"""-------------------------------
    Set/Get saved data from config
----------------------------------"""

""" Raw data from project files """
def set_raw_files_data(data):
    config.raw_files_data = data

def get_raw_files_data():
    return config.raw_files_data

def set_scenario_names(names):
    clean_names = [
        f"Scenario {match.group(1)}: {match.group(2).replace('-', ' ').title()}".replace("Gi", "GI")
        for name in names
        if (match := re.match(r"scenario(\d+)_(.+)\.json", name))
    ]

    config.scenario_names = clean_names

def get_scenario_names():
    return config.scenario_names

def set_raw_attributes(raw_values):
    config.raw_attr_values = np.array(raw_values, dtype=float)

def get_raw_attributes():
    return config.raw_attr_values

""" Normalized data """
def set_attributes_norm(A, scenarios):
    config.logger.debug(f"\t Setting attributes norm: {A}, shape: {A.shape}")
    sorted_norm = sorted(
        zip(scenarios, A),
        key=lambda x: x[0]
    )
    config.attributes_norm = sorted_norm
    config.logger.debug(f"\t✔️  Set attributes_norm: {sorted_norm}")

def get_attributes_norm(with_names=False):
    scenarios, A_norm = zip(*config.attributes_norm)
    A_norm = np.array(A_norm)
    config.logger.debug(f"\tReturning attributes_norm: {A_norm}, type: {type(A_norm)}, shape: {A_norm.shape}")

    if with_names:
        config.logger.debug(f"Returning scenarios sorted: {scenarios}")
        return scenarios, A_norm
    else:
        return A_norm

""" Weighted attributes """
def set_weighted_attributes(A_weight, scenarios):
    sorted_attributes = sorted(
        zip(scenarios, A_weight),
        key=lambda x: x[0]
    )
    config.weighted_attributes = sorted_attributes
    config.logger.debug(f"\t✔️  Set weight_attributes: {A_weight}")

def get_weighted_attributes(with_names=False):
    scenarios, sorted_weighted_attr = zip(*config.weighted_attributes)
    if with_names:
        return scenarios, sorted_weighted_attr
    else:
        return sorted_weighted_attr

""" Sorted Scores """
def set_sorted_scenario_scores(new_sorted_scores):
    config.sorted_scenario_scores = new_sorted_scores
    config.logger.debug(f"\t✔️  Set attributes_norm: {config.sorted_scenario_scores}")

def get_sorted_scenario_scores():
    return config.sorted_scenario_scores

""" Weights """
def set_weights(weight_arg):
    if weight_arg != config.WEIGHT_SCHEME:
        config.WEIGHTS = config.WEIGHT_OPTS.get(weight_arg, WEIGHT_OPTS["DEFAULT"])
        config.logger.debug(f"\t✔️  Set WEIGHTS: {get_weights()}")
    else:
        config.logger.debug(f"Using weight scheme defined in config.yaml: {config.WEIGHT_SCHEME}")

def get_weights():
    return config.WEIGHTS

"""-------------------------
    Directory init
-------------------------"""
def create_dirs(sens=False,proj=True, save=True, log=True, raw=True, processed=True, png=True):
    config.logger.debug("Creating directories for pipeline artifacts...")

    if proj:
        os.makedirs(config.JSON_DIR, exist_ok=True)
        config.logger.debug(f"\t✔️  Project directory: {config.JSON_DIR}")
    if save:
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        config.logger.debug(f"\t✔️  Save directory: {config.SAVE_DIR}")
    if log:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        config.logger.debug(f"\t✔️  Log files: {config.LOG_DIR}")
    if raw:
        os.makedirs(config.RAW_DIR, exist_ok=True)
        config.logger.debug(f"\t✔️  Raw data: {config.RAW_DIR}")
    if processed:
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        config.logger.debug(f"\t✔️  Processed data:{config.PROCESSED_DIR}")
    if png:
        os.makedirs(config.PNG_DIR, exist_ok=True)
        config.logger.debug("Created directories for pipeline artifacts")
    if sens:
        os.makedirs(config.SENS_DIR, exist_ok=True)
        config.logger.debug(f"\t✔️  Sensitivity: {config.SENS_DIR}")



def setup_totals_file():
    with open(config.TOTALS_FILEPATH, "w", newline="", encoding="utf-8") as savetotals:
        writer = csv.DictWriter(savetotals, fieldnames=config.HEADERS)
        writer.writeheader()
    config.logger.debug(f"\t✔️  Created CSV for totals: {config.TOTALS_FILEPATH}")


"""-------------------------
    File utils (json, csv, png)
-------------------------"""
# Throws JSONDecodeError if unsuccessful
def load_json(file_path):
    config.logger.info(f"\tReading data from {file_path}")

    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_to_csv(filepath, data):
    base_name = os.path.basename(filepath)
    csv_filename = f"{config.RAW_DIR}/{base_name}_raw.csv"
    config.logger.debug(f"\tWriting data to CSV {csv_filename}")


    if not os.path.exists(config.RAW_DIR):
        logger.debug(f"WARN: Expected {config.RAW_DIR} to already exist. Did you run init_pipeline()?")
        os.mkdir(config.RAW_DIR)
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config.HEADERS)
        writer.writeheader()
        writer.writerows(data)
    config.logger.info(f"\t✔️  Saved raw data to {os.path.basename(filepath)}")

def sort_csv(filepath, sort_col, has_headers):
    config.logger.debug(f"Sorting CSV file {filepath} by column {sort_col}")
    with open(filepath, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        if has_headers:
            header = next(reader)  # Save the header row
        data = list(reader)    # Load the remaining rows into a list

    # Sort rows by the specified column
    data.sort(key=lambda x: x[sort_col])

    # Overwrites the original file
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def write_scores_to_csv(filepath, scenario_scores):
    output_filename = f"weighted_scores.csv"
    output_filepath = f"{filepath}/{output_filename}"
    with open(output_filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Scenario", "WeightedScore"])
        for scenario, score in scenario_scores:
            writer.writerow([scenario, round(float(score), 4)])
    config.logger.debug(f"\t✔️  Weighted scores written to:\n\t{output_filepath}")

"""Load totals from config file or CSV. Returns None if no data is saved"""
def load_raw_values(use_config=False):
    return load_config_totals()


"""Load totals from config file. Assumes extract_data ran before process_data in pipeline"""
def load_config_totals():
    config.logger.debug("...Loading raw values from config")
    names = get_scenario_names()
    raw_values = get_raw_attributes()
    return names, raw_values


def load_csv_totals():
    filepath = config.TOTALS_FILEPATH
    if (not has_totals_csv()):
        raise RuntimeException("Expected {filepath}, but it doesn't exist")
    config.logger.debug(f"\tLoading totals CSV from {filepath}")
    names = []
    matrix = []

    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            names.append(row.get("Name", ""))
            values = []
            for attr in config.ATTRIBUTES_LIST:
                value = float(row.get(attr, 0.0))
                values.append(value)
            matrix.append(values)

    attributes = np.array(matrix, dtype=float)
    config.logger.debug(f"\t✔️  Loaded attributes matrix with shape {attributes.shape}")
    # Save values to config
    set_scenario_names(names)
    set_raw_attributes(attributes)

    return names, attributes


def is_load_successful(names, raw_values):
    if names is None or raw_values is None:
        config.logger.info(f"❓Failed to load raw data")
        return False
    config.logger.info(f"\t✔️  Successfully loaded raw data")
    return True


def save_png(file_name, save_dir):
    png_file = f"{file_name}.png"
    plt.savefig(f"{save_dir}/{png_file}", bbox_inches='tight')
    config.logger.info(f"\t✔️  Chart successfully saved to {png_file}")

"""-------------------------
    Validation utils
-------------------------"""

def validate_filepath(filepath: Path, expected_type: str) -> bool:
    filepath = Path(filepath)
    actual_type = filepath.suffix.lower()
    if actual_type != expected_type.lower():
        error_msg = f"Expected file type {expected_type} but received {actual_type}"
        raise TypeError(error_msg)
    if not filepath.is_file():
        error_msg = f"File '{filepath}' does not exist or is a directory."
        raise ValueError(error_msg)
    return True

def has_totals_csv():
    config.logger.debug("Checking if totals CSV is available")
    try:
        return validate_filepath(config.TOTALS_FILEPATH, ".csv")
    except ValueError as e:
        config.logger.debug("No saved data available")
        return False

def validate_crct_json(json):
    if "areas" not in json or not isinstance(json["areas"], list):
        raise ValueError("Invalid CRCT JSON project file, missing 'area' name.")
    config.logger.info("\t✔️  Verified CRTC project JSON file")
    return True


def validate_weights():
    if not np.isclose(config.WEIGHTS.sum(), 1.0):
        config.logger.error(f"...ERROR: Weights vector must sum to 1."
              f"\nCurrent sum = {config.WEIGHTS.sum()}")
        raise ValueError("Weights vector must sum to 1")
    if config.WEIGHTS.shape[0] != len(config.ATTRIBUTES_LIST):

        config.logger.error(f"...ERROR: Weights vector must be length {len(config.ATTRIBUTES_LIST)})"
              f"but is ({config.WEIGHTS.shape[0]}).")
        raise ValueError(f"Weights vector does not have expected length")

    config.logger.debug(f"\n⚖️  Using weight vector (sum={config.WEIGHTS.sum()}):")
    for attr, weight in zip(config.ATTRIBUTES_LIST, config.WEIGHTS):
        config.logger.debug(f"...{attr}: {weight}")
    return True

def validate_attributes_matrix(A):
    if A is None or len(A) == 0:
        config.logger.error("...ERROR: Attributes matrix is empty")
        raise ValueError("Attributes matrix is empty")
    shape = A.shape
    num_attr = len(config.ATTRIBUTES_LIST)
    num_scenarios = len(get_scenario_names())
    if shape[0] != num_scenarios:
        config.logger.error("...ERROR: Attributes matrix has {shape[0]} but expected {num_scenarios}")
        raise ValueError("Attributes matrix does not have the expected dimensions.")
    if shape[1] != num_attr:
        config.logger.error("...ERROR: Attributes matrix has {num_cols} but expected {num_attr}")
        raise ValueError("Attributes matrix does not have the expected dimensions.")
    return True
