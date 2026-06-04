import numpy as np
from datetime import datetime
from pathlib import Path
from enum import Enum

'''
------------------------------------------
#   Project Configuration - Edit me!
------------------------------------------
'''

'''
Folder containing CRCT project JSON files
'''
PROJ_DIR = "/Users/evandeo/Documents/PEPM/research_project/project_files"
PARENT_DIR = Path(PROJ_DIR).parent # Folder for results will be created here

'''
Directories for saved results
'''
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_DIR = f"{PARENT_DIR}/analysis/{TIMESTAMP}"
RAW_DIR = f"{SAVE_DIR}/raw"
PROCESSED_DIR = f"{SAVE_DIR}/processed"
PNG_DIR = f"{SAVE_DIR}/png_results"
SENS_DIR = f"{SAVE_DIR}/sensitivity"
SENS_FILEPATH = f"{SENS_DIR}/sensitivity_summary.csv"
TOTALS_FILENAME = f"scenario_totals_{TIMESTAMP}.csv"
TOTALS_FILEPATH = f"{RAW_DIR}/{TOTALS_FILENAME}"

'''
Logger
'''
LOG_DIR = f"{PARENT_DIR}/logs"
LOGGER_NAME = "DataPipeline"
LOG_FILE = f"{LOG_DIR}/DataPipeline.log"
LOG_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s): %(message)s"
LOG_CONSOLE_FORMAT = "%(message)s"

# Logs will rotate after 2MB. Set to 0 to disable log rotation.
LOG_MAX_SIZE_BYTES = 2 * 1024 * 1024

# After reaching max backup log files, the oldest logs will be overwritten
LOG_MAX_BACKUPS = 100

# Num backup files before user is warned. Set to -1 to disable checking.
LOG_MAX_WARN_THRESHOLD = 97


"""
Score sorting
0: Sort by index (Default)
1: Sort by score, ascending
2: Sort by score, descending
"""
SORT_TYPE = 0


'''
Weights - Edit me to reflect criteria priorities
'''
DEFAULT = np.array([
    0.20,     # ConstructionCost
    0.20,     # MaintenanceCost
    0.20,     # StorageCapacity
    0.10,     # GroundwaterRecharge
    0.10,     # Evapotranspiration
    0.05,     # TempReduction
    0.05,     # NutrientReduction
    0.05,     # PathogenReduction
    0.05      # AdsorbingPollutants
    ])


'''
Flat weights (.111111)
'''
FLAT = np.array([
    .111111,     # ConstructionCost
    .111111,     # MaintenanceCost
    .111111,     # StorageCapacity
    .111111,     # GroundwaterRecharge
    .111111,     # Evapotranspiration
    .111111,     # TempReduction
    .111111,     # NutrientReduction
    .111111,     # PathogenReduction
    .111111      # AdsorbingPollutants

    ])

'''
To use, run with: -w CUSTOM_WEIGHTS1
'''
CUSTOM1 = np.array([
    .111111,     # ConstructionCost
    .111111,     # MaintenanceCost
    .111111,     # StorageCapacity
    .111111,     # GroundwaterRecharge
    .111111,     # Evapotranspiration
    .111111,     # TempReduction
    .111111,     # NutrientReduction
    .111111,     # PathogenReduction
    .111111      # AdsorbingPollutants
    ])

'''
To use, run with: -w CUSTOM_WEIGHTS2
'''
CUSTOM2 = np.array([
    .111111,     # ConstructionCost
    .111111,     # MaintenanceCost
    .111111,     # StorageCapacity
    .111111,     # GroundwaterRecharge
    .111111,     # Evapotranspiration
    .111111,     # TempReduction
    .111111,     # NutrientReduction
    .111111,     # PathogenReduction
    .111111      # AdsorbingPollutants
    ])

WeightsIndex = Enum('Weights', [('DEFAULT', 0), ('FLAT', 1), ('CUSTOM1', 2), ('CUSTOM2', 3)])
WEIGHTS_OPTS = [DEFAULT, FLAT, CUSTOM1, CUSTOM2]

'''
------------------------------------------
#   WARNING
    Edits below could break the pipeline.
------------------------------------------
'''
'''
# Field mapping for project JSON files
'''
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

'''
# Attributes included in analysis
'''
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

'''
Sensitivity
'''
TARGET_STEP_SIZE = 0.05
NON_TARGET_STEP_SIZE = TARGET_STEP_SIZE / (len(ATTRIBUTES_LIST) - 1)
''''''


# Help Strings
PROG_NAME = "crct_data_pipeline"
PROG_DESCR = "Extracts, processes & analyzes data from the CRCT project JSON files."
HELP_EXTRACT = "Force extract of data from project JSON files, overwriting any saved data"
HELP_FOLDER = "Folder containing project JSON files. Defaults to PROJ_DIR set in config.py "
HELP_SENSITIVITY = "Run sensitivity analysis"
HELP_SORTTYPE = "0: Sort by scenario name. 1: Sort by score ascending. 2: Sort by score descending"
HELP_VERBOSE = "Print INFO logs to console"
HELP_DEBUG = "Print DEBUG logs to console"
HELP_QUIET = "Suppress all but CRITICAL logs from printing to console"
HELP_SHOWRADAR = "Display radar charts in addition to saving them as PNGs"
HELP_WEIGHTS = "Specify the name of the weights to use (DEFAULT, FLAT, CUSTOM1, CUSTOM2). Example: -w CUSTOM1"


