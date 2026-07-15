import json
import csv
import os
import sys
import argparse
from datetime import datetime
from config import *
import glob
from pathlib import Path
import itertools
import src.utils as utils


logger = None

"""
Extracts area data from a Climate Resilient Cities Tool (CRCTool) project file
and saves it as a CSV.
The CSV includes a totals row for all numeric columns.
"""
def extract_all_data(folder):
    count = 0
    logger.info(f"\n⏳Extracting data from all JSON files in '{folder}'...")
    path = Path(folder)
    project_files = path.glob("*.json")
    file = get_next(project_files)

    # Includes scenario names and attribute totals for each scenario
    files_data = []
    while file is not None:
        totals = extract_project_data(file)
        files_data.append(totals)
        count += 1
        file = get_next(project_files)
    if count == 0:
        raise FileNotFoundError(f"No JSON files found at '{folder}'")

    utils.sort_csv(TOTALS_FILEPATH, 0, True) # Sort by scenario name
    names, raw_attributes = split_names_values_lists(files_data)

    save_extracted_to_config(files_data, names, raw_attributes)
    logger.info(f"\t✔️  Extracted data from {count} JSON files in '{folder}'")
    logger.warning(f"✅ Data extraction complete.")

    return names, raw_attributes


def extract_project_data(filepath):
    logger.info(f"⏳Extracting data from {filepath}")

    # Open the project JSON
    json = utils.load_json(filepath)
    utils.validate_crct_json(json)

    # Extract data from project JSON
    file_data = extract_data_from_json(json)

    # Calculate attribute totals for this scenario
    totals = calc_file_totals(file_data, filepath)
    file_data.append(totals)

    # Save extracted data to CSV
    append_to_totals_csv(totals)
    utils.write_to_csv(filepath, file_data)
    logger.info(f"\t✔️  Finished extracting data from {os.path.basename(filepath)}.")

    return totals


"""
Strips files_data down to a list of scenario names and list of scenario attributes values
"""
def split_names_values_lists(files_data):
    names = []
    attributes = []
    for scenario in files_data:
        names.append(scenario["Name"])
        attributes.append(extract_attributes(scenario))

    sorted_pairs = sorted(
        zip(names, attributes),
        key=lambda x: x[0]
    )
    names, attributes = zip(*sorted_pairs)
    scenario_names = list(names)
    scenarios_attributes = np.array(attributes)
    return scenario_names, scenarios_attributes

def extract_attributes(scenario):
    attributes = []
    for attr in ATTRIBUTES_LIST:
        attributes.append(scenario[attr])
    return attributes

def calc_file_totals(rows, filepath):
    logger.debug(f"⏳Calculating totals for {os.path.basename(filepath)}...")
    totals = {"Name" : f"{os.path.basename(filepath)}", "MeasureCode": "-"}
    for key in HEADERS[2:]:
        total = 0.0
        for row in rows:
            if isinstance(row[key], float):
                try:
                    total += float(row[key])
                except (ValueError, TypeError):
                    pass
            else:
                totals[key] = row[key]
        totals[key] = round(total, 5)
        logger.debug(f"....Set totals value for key {key} to {totals[key]}")
    logger.info(f"\t✔️  Calculated totals for {os.path.basename(filepath)}")
    return totals


"""
JSON, CSV, config.py
"""

def extract_data_from_json(data):
    logger.debug("⏳Getting raw attribute data from json...")
    rows = []
    for area in data["areas"]:
        props = area.get("properties", {})
        api_data = props.get("apiData", {})
        row = {}
        for field, (group, key) in FIELD_MAP.items():
            if group == "properties":
                row[field] = props.get(key, "")
            elif group == "apiData":
                row[field] = api_data.get(key, "")
#         logger.debug(f"....Extracted row: {row}")
        rows.append(row)
    logger.info("\t✔️  Extracted raw attribute data from json")
    return rows

def append_to_totals_csv(totals):
    logger.debug(f"⏳Appending totals to {TOTALS_FILEPATH}...")
    with open(TOTALS_FILEPATH, "a", newline="", encoding="utf-8") as savetotals:
        writer = csv.DictWriter(savetotals, fieldnames=HEADERS)
        writer.writerow(totals)
    logger.info(f"\t✔️  Appended scenario totals to {TOTALS_FILEPATH}")

def save_extracted_to_config(files_data, names, raw_attributes):
    utils.set_raw_files_data(files_data)
    logger.debug(f"\t✔️  Set raw files data: {utils.get_raw_files_data()}")
    utils.set_scenario_names(names)
    logger.debug(f"\t✔️  Set scenario names: {utils.get_scenario_names()}")
    utils.set_raw_attributes(raw_attributes)
    logger.debug(f"\t✔️ Set raw attributes data by scenario: {utils.get_raw_attributes()}")


"""
Misc
"""
def get_next(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None # Iterable is empty
    return first

def init_extract():
    global logger
    logger = utils.get_logger()
    utils.setup_totals_file()


