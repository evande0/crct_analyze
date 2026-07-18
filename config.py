# config.py
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np

# Project JSON field map
FIELD_MAP = {
        "Name": ("properties", "name"),
        "MeasureCode": ("properties", "measure"),
        "ConstructionCost": ("apiData", "constructionCost"),
        "MaintenanceCost": ("apiData", "maintenanceCost"),
        "StorageCapacity": ("apiData", "storageCapacity"),
        "GroundwaterRecharge": ("apiData", "groundwater_recharge"),
        "Evapotranspiration": ("apiData", "evapotranspiration"),
        "TempReduction": ("apiData", "tempReduction"),
        "NutrientReduction": ("apiData", "captureUnit"),
        "PathogenReduction": ("apiData", "filteringUnit"),
        "AdsorbingPollutants": ("apiData", "settlingUnit"),
        "CoolSpot": ("apiData", "coolSpot"),
        "FMeasureArea": ("apiData", "Fmeas_area")
    }
HEADERS = list(FIELD_MAP.keys())

# Attributes included in analysis
ATTRIBUTES_LIST = [
    "ConstructionCost",
    "MaintenanceCost",
    "StorageCapacity",
    "GroundwaterRecharge",
    "Evapotranspiration",
    "TempReduction",
    "NutrientReduction",
    "PathogenReduction",
    "AdsorbingPollutants",
]

# Load user config
with open("config.yaml", "r") as f:
    _user_config = yaml.safe_load(f)

JSON_DIR = _user_config["paths"]["json_dir"]

SORT_OPTS = _user_config["score_sort"]
SORT_BY = _user_config["analysis"]["sort_by"]
SORT_IDX = SORT_OPTS.get(SORT_BY, 0)

WEIGHT_OPTS = {k: np.array(v) for k, v in _user_config["weights"].items()}
WEIGHT_SCHEME = _user_config["analysis"]["weights_selection"]
WEIGHTS = WEIGHT_OPTS.get(WEIGHT_SCHEME, WEIGHT_OPTS["DEFAULT"])

TARGET_STEP_SIZE = _user_config["sensitivity"]["step_size"]
NON_TARGET_STEP_SIZE = TARGET_STEP_SIZE / (len(ATTRIBUTES_LIST) - 1)

LOG_MAX_SIZE_MB = _user_config["logger"]["max_size_mb"]
LOG_MAX_SIZE_BYTES = LOG_MAX_SIZE_MB * 1024 * 1024
LOG_MAX_BACKUPS = _user_config["logger"]["max_backups"]
LOG_MAX_WARN_THRESHOLD = _user_config["logger"]["max_warn_threshold"]


# Directories
BASE_DIR = Path(__file__).resolve().parent
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_DIR = BASE_DIR / "results" / TIMESTAMP
RAW_DIR = SAVE_DIR / "raw"
PROCESSED_DIR = SAVE_DIR / "processed"
PNG_DIR = SAVE_DIR / "viz"
SENS_DIR = SAVE_DIR / "sensitivity"
LOG_DIR = BASE_DIR / "logs"

# File paths
SENS_FILEPATH = SENS_DIR / "sensitivity_summary.csv"
TOTALS_FILEPATH = RAW_DIR / "scenario_totals.csv"
LOG_FILEPATH = LOG_DIR / "pipeline.log"

# Logger Format
LOGGER_NAME = "DataPipeline"
LOG_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s): %(message)s"
LOG_CONSOLE_FORMAT = "[%(levelname)s] %(filename)s, %(lineno)d: %(message)s"


# Viz Defaults
WIDTH = 7
HEIGHT = 4
TITLE_SIZE = 13
LABEL_SIZE = 12
TICK_SIZE = 11
VALUE_SIZE = 10
ROTATION = 30
BAR_HEIGHT = 0.85
LINE_WIDTH = 0.25
GRID_ALPHA = 0.4
BAR_ALPHA = 0.9
PADDING = 0.05
ANCHOR = (0.5, -0.15)
EDGE_COLOR = '#555555'
AX_COLOR = '#333333'
VALUE_DARK = '#222222'
VALUE_LIGHT = '#FFFFFF'
COLOR_MAP = "coolwarm_r"
COMPACT_FONT_SIZE = 8
COMPACT_SCALE = 0.75         # Scales figure dimensions when using -c compact flag


# Help Strings
PROG_NAME = "crct_data_pipeline"
PROG_DESCR = "Extracts, processes & analyzes data from the CRCT project JSON files."
HELP_FOLDER = "Folder containing project JSON files. Defaults to PROJ_DIR set in config.py "
HELP_SHOW = "Display visualisations in addition to saving them as PNGs"
HELP_VERBOSE = "Print INFO logs to console"
HELP_DEBUG = "Print DEBUG logs to console"
HELP_QUIET = "Suppress all but CRITICAL logs from printing to console"
HELP_SENSITIVITY = "Run sensitivity analysis"
HELP_WEIGHTS = "Specify the name of the weights to use. DEFAULT, FLAT, CUSTOM1, CUSTOM2 are customizable in 'config.yaml'. Example: -w CUSTOM1"
HELP_COMPACT = "Compact visualisations without titles"
