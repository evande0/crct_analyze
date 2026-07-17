import argparse
import csv
import config
import os

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from src.utils import get_logger, is_none, validate_attr_matrix, validate_weights, write_scores_to_csv


logger = None


"""
Loads extracted data, inverts cost criteria, and applies L2 normalization
"""
def process_data(extracted_data):
    scenarios = extracted_data["scenario_names"]
    raw_values = extracted_data["attributes_data"]
    if (is_none([scenarios, raw_values])):
        raise ValueError(f"Error using extracted scenario data. \nscenarios: {scenarios} \nraw_values: {raw_values}")

    # Normalize attributes
    attributes_norm = normalize_attributes(raw_values)
    validate_attr_matrix(attributes_norm, len(scenarios))
    np.savetxt(f"{config.PROCESSED_DIR}/attributes_norm_l2.csv", attributes_norm, delimiter=',', fmt='%10.5f')

    # Validate weights vector
    validate_weights();
    logger.info(f"Using weights: {config.WEIGHTS}")

    # Compute weighted attributes
    attributes_wtd = np.round(attributes_norm * config.WEIGHTS, 10)
    np.savetxt(f"{config.PROCESSED_DIR}/attributes_weighted.csv", attributes_wtd, delimiter=',', fmt='%10.5f')

    # Compute scores
    logger.debug("⏳Computing weighted scores")
    scores = compute_weighted_scores(attributes_norm, config.WEIGHTS)
    if scores is None:
        logger.error("❗Failed to compute scores. Aborting score computation.\n")
        return

    # Sort results
    sorted_scenario_scores = sort_scores(zip(scenarios[1:], scores[1:]))
    print_scenario_scores(sorted_scenario_scores)
    write_scores_to_csv(config.PROCESSED_DIR, sorted_scenario_scores)

    logger.warning(f"✅ Processing complete.")
    return {"scenario_names": scenarios, "attributes_norm": attributes_norm, "attributes_wtd": attributes_wtd, "sorted_results": sorted_scenario_scores}



"""-------------------------
    Normalization
-------------------------"""

def normalize_attributes(raw_values):
    attributes_matrix = invert_costs(raw_values);
    attributes_norm = l2_norm(attributes_matrix)
    return attributes_norm

"""
Returns matrix after applying L2 normalization to columns.
"""
def l2_norm(A):
    logger.debug("⏳Performing L2 column normalization")
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0     # Avoid dividing by 0 in all 0 column
    A_norm = np.round(A / norms, 10)
    return A_norm

"""
Linear Max normalisation
"""
def max_abs_norm(data):
    # Max absolute value in each column
    col_max_abs = np.max(np.abs(data), axis=0)

    # Avoid dividing by 0 in all 0 column
    col_max_abs[col_max_abs == 0] = 1.0

    # Divide each element in a column by that column's max absolute value
    normalized_data = np.round(data / col_max_abs, 10)
    return normalized_data

def init_process():
    global logger
    logger = get_logger()



"""-------------------------
    Score Computation
-------------------------"""

def invert_costs(matrix):
    construction_idx = config.ATTRIBUTES_LIST.index("ConstructionCost")
    maintenance_idx = config.ATTRIBUTES_LIST.index("MaintenanceCost")
    A = matrix.copy()
    A[:, construction_idx] = -A[:, construction_idx]
    A[:, maintenance_idx] = -A[:, maintenance_idx]
    return A

def compute_weighted_scores(A_norm, weights):
    try:
        return np.round(A_norm @ weights, 10)
    except Exception as e:
        logger.error(f"....ERROR computing weighted scores: {e}")
        return None

def sort_scores(scenario_scores):
    if config.SORT_IDX == 0:
        logger.debug(f"\t✔️  Sorted by Scenario name")
    elif config.SORT_IDX == 1:
        logger.debug(f"\t✔️  Sorted by Score (ascending)")
        return sorted(scenario_scores, key=lambda x: x[1], reverse=True)
    elif config.SORT_IDX == 2:
        logger.debug(f"\t✔️  Sorted by Score (descending)")
        return sorted(scenario_scores, key=lambda x: x[1])
    else:
        logger.error(f"\tERROR: Invalid sort index {args.sortindex}."
              f"See SORT_IDX in config.py")
        logger.error("! Defaulting to sort by Name.")
    return sorted(scenario_scores, key=lambda x: x[0].lower())


def print_scenario_scores(scenario_scores):
    logger.info("\n📊Weighted Scores:")
    for scenario, score in scenario_scores:
        logger.info(f"{round(float(score), 4)}\t{scenario}: ")
    logger.info("Weighted scores are between -1 and 1. "
          "A score of 0 means no change from current conditions.")
