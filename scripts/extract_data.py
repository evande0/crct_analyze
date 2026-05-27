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
import utils

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
        file_data, file_totals = extract_project_data(file)
        files_data.append(file_data)
        count += 1
        file = get_next(project_files)
    if count == 0:
        raise FileNotFoundError(f"No JSON files found at '{folder}'")

    # Sort totals CSV by scenario name
    utils.sort_csv(TOTALS_FILEPATH, 0, True)
    save_extracted_to_config(files_data)
    logger.info(f"\t✔️  Extracted data from {count} JSON files in '{folder}'")

def save_extracted_to_config(files_data):
    set_raw_files_data(files_data)
    config.logger.debug(f"\t✔️  Set raw files data: {config.raw_files_data}")

    names, raw_attr_values = split_names_values_lists(files_data)
    set_scenario_names(names)
    config.logger.debug(f"\t✔️  Set scenario names: {config.scenario_names}")
    set_raw_attr_values(raw_attr_values)
    config.logger.debug(f"\t✔️  Set raw attributes data by scenario: {config.raw_attr_values}")

def extract_project_data(filepath):
    logger.info(f"⏳Extracting data from {filepath}")

    # Open the project JSON
    json = utils.load_json(filepath)
    utils.validate_crct_json(json)

    # Extract data from project JSON
    project_data = get_rows(json)

    # Calculate attribute totals for this scenario
    scenario_totals = calc_file_totals(project_data, filepath)
    project_data.append(scenario_totals)

    # Save extracted data to CSV
    append_to_totals_csv(scenario_totals)
    utils.write_to_csv(filepath, project_data)
    logger.info(f"\t✔️  Finished extracting data from {os.path.basename(filepath)}.")

    return file_data, scenario_totals



"""
Strips files_data down to a list of scenario names and list of scenario attributes values
"""
def split_names_values_lists(files_data):
    scenario_names = []
    scenarios_attributes = []
    for scenario in files_data:
        scenario_names.append(scenario["Name"])
        scenarios_attributes.append(extract_attributes(scenario))
    return scenario_names, scenarios_attributes

def extract_attributes(scenario):
    attributes = []
    for attr in config.ATTRIBUTES_LIST:
        attributes.append(scenario[attr])
    scenario_names.append(scenario["Name"])
    return attributes


"""
def set_raw_data(new_totals):
    all_totals = []
    scenarios = []
    for scenario in new_totals:
        scenarios.append(scenario["Name"])
        totals = []
        for attr in config.ATTRIBUTES_LIST:
            totals.append(scenario[attr])
        all_totals.append(totals)
    config.scenarios = scenarios
    config.totals = all_totals
    config.logger.debug(f"\t✔️  Set raw totals: {config.totals}")
    config.logger.debug(f"\t✔️  Set scenario names: {config.scenarios}")
"""

def get_rows(data):
    logger.debug("⏳Getting rows from data...")

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
    logger.info("\t✔️  Extracted rows from data")
    return rows

def calc_file_totals(rows, filepath):
    logger.debug(f"⏳Calculating totals for {os.path.basename(filepath)}...")
    scenario_name = os.path.basename(filepath);
    scenario_totals = {"Name" : f"{scenario_name}", "MeasureCode": "-"}
    for key in HEADERS[2:]:
        total = 0.0
        for row in rows:
            if isinstance(row[key], float):
                try:
                    total += float(row[key])
                except (ValueError, TypeError):
                    pass
            else:
                scenario_totals[key] = row[key]
        scenario_totals[key] = round(total, 5)
    logger.info(f"\t✔️  Calculated totals for {scenario_name}")
    return scenario_totals

def append_to_totals_csv(totals):
    logger.debug(f"⏳Appending totals to {TOTALS_FILENAME}...")
    with open(TOTALS_FILEPATH, "a", newline="", encoding="utf-8") as savetotals:
        writer = csv.DictWriter(savetotals, fieldnames=HEADERS)
        writer.writerow(totals)
    logger.info(f"\t✔️  Appended scenario totals to {TOTALS_FILENAME}")

def get_next(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None # Iterable is empty
    return first

def init_extract():
    global logger
    logger = utils.get_logger()


