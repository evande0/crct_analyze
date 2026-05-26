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
    all_totals = []
    while file is not None:
        all_totals.append(extract_data(file))
        count += 1
        file = get_next(project_files)
    if count == 0:
        raise FileNotFoundError(f"No JSON files found at '{folder}'")
    utils.sort_csv(TOTALS_FILEPATH, 0, True) # Sort by scenario name
    utils.set_raw_data(all_totals)
    logger.info(f"\t✔️  Extracted data from {count} JSON files in '{folder}'")

def extract_data(filepath):
    logger.info(f"⏳Extracting data from {filepath}")

    # Open the JSON
    json = utils.load_json(filepath)
    utils.validate_crct_json(json)

    # Extract attribute data from the scenarios
    project_data = get_rows(json)

    # Calculate scenario totals
    totals = calc_totals(project_data, filepath)
    project_data.append(totals)

    # Append scenario totals to the totals file
    append_to_totals_csv(totals)

    # Write data to CSV
    utils.write_to_csv(filepath, project_data)

    logger.info(f"\t✔️  Finished extracting data from {os.path.basename(filepath)}.")

    return totals


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

def calc_totals(rows, filepath):
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
#         logger.debug(f"....Set totals value for key {key} to {totals[key]}")
    logger.info(f"\t✔️  Calculated totals for {os.path.basename(filepath)}")
    return totals

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


