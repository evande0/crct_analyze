import numpy as np
from datetime import datetime
from pathlib import Path


'''
------------------------------------------
#   Project Configuration - Edit me!
------------------------------------------
'''

'''
# Folder containing CRCT project JSON files
'''
PROJ_DIR = "/Users/evandeo/Documents/PEPM/research_project/project_files"
PARENT_DIR = Path(PROJ_DIR).parent # Folder for results will be created here

'''
# Directories for saved results
'''
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
SAVE_DIR = f"{PARENT_DIR}/pipeline_results/{TIMESTAMP}"
RAW_DIR = f"{SAVE_DIR}/raw"
PROCESSED_DIR = f"{SAVE_DIR}/processed"
PNG_DIR = f"{SAVE_DIR}/png_results"
TOTALS_FILENAME = f"scenario_totals_{TIMESTAMP}.csv"
TOTALS_FILEPATH = f"{RAW_DIR}/{TOTALS_FILENAME}"

'''
# Logger
'''
LOGGER_NAME = "DataPipeline"
LOG_DIR = f"{PARENT_DIR}/logs"
LOG_FILE = f"{LOG_DIR}/DataPipeline.log"
LOG_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s): %(message)s"
LOG_CONSOLE_FORMAT = "%(message)s"

# Logs will rotate after 2MB. Set to 0 to disable log rotation.
LOG_MAX_SIZE_BYTES = 2 * 1024 * 1024

# After reaching max backup log files, the oldest logs will be overwritten
LOG_MAX_BACKUPS = 100

# Remaining log backup files before user is warned. Set to -1 to disable checking.
LOG_MAX_WARN_THRESHOLD = 3


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
WEIGHTS = np.array([
    .2,     # ConstructionCost
    .2,     # MaintenanceCost
    .05,     # TempReduction
    .05,     # NutrientReduction
    .05,     # PathogenReduction
    .05,     # AdsorbingPollutants
    .1,      # GroundwaterRecharge
    .1,      # Evapotranspiration
    .2       # StorageCapacity
    ])


'''
Flat weights (.111111)
'''
# WEIGHTS = np.array([
#     .111111,     # ConstructionCost
#     .111111,     # MaintenanceCost
#     .111111,     # TempReduction
#     .111111,     # NutrientReduction
#     .111111,     # PathogenReduction
#     .111111,     # AdsorbingPollutants
#     .111111,      # GroundwaterRecharge
#     .111111,      # Evapotranspiration
#     .111111       # StorageCapacity
#     ])


'''
------------------------------------------
#   WARNING
    Edits below could break the pipeline.
------------------------------------------
'''

'''
Variables saved for use in other pipeline stages
'''
logger = None
scenarios = None
attributes_norm = None
totals = None

'''
# Field mapping for project JSON files
'''
FIELD_MAP = {
        "Name": ("properties", "name"),
        "MeasureCode": ("properties", "measure"),
        "ConstructionCost": ("apiData", "constructionCost"),
        "MaintenanceCost": ("apiData", "maintenanceCost"),
        "CoolSpot": ("apiData", "coolSpot"),
        "TempReduction": ("apiData", "tempReduction"),
        "NutrientReduction": ("apiData", "captureUnit"),
        "PathogenReduction": ("apiData", "filteringUnit"),
        "AdsorbingPollutants": ("apiData", "settlingUnit"),
        "FMeasureArea": ("apiData", "Fmeas_area"),
        "GroundwaterRecharge": ("apiData", "groundwater_recharge"),
        "Evapotranspiration": ("apiData", "evapotranspiration"),
        "StorageCapacity": ("apiData", "storageCapacity")
    }
HEADERS = list(FIELD_MAP.keys())

'''
# Attributes included in analysis
'''
ATTRIBUTES_LIST = [
    "ConstructionCost",
    "MaintenanceCost",
    "TempReduction",
    "NutrientReduction",
    "PathogenReduction",
    "AdsorbingPollutants",
    "GroundwaterRecharge",
    "Evapotranspiration",
    "StorageCapacity"
]


# Help Strings
PROG_NAME = "crct_data_pipeline"
PROG_DESCR = "Extracts, processes, and analyzes data from the CRCT project JSON files saved in the specified folder."
HELP_FOLDER = "Folder containing project JSON files. Defaults to PROJ_DIR set in config.py "
HELP_SORTTYPE = "0: Sort by scenario name. 1: Sort by score ascending. 2: Sort by score descending"
HELP_VERBOSE = "Print INFO logs to console"
HELP_DEBUG = "Print DEBUG logs to console"
HELP_QUIET = "Suppress all but CRITICAL logs from printing to console"
HELP_SHOWRADAR = "Display radar charts in addition to saving them as PNGs"

