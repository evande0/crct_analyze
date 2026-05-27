import csv
import os
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from config import *
from utils import *

logger = None


"""
Loads extracted data, inverts cost criteria, and applies L2 normalization
"""
def process_data():
    scenarios, raw_values = load_raw_values()


    # Sort by scenario name
    sorted_pairs = sorted(
        zip(scenarios, raw_values),
        key=lambda x: x[0]
    )
    scenarios, raw_values = zip(*sorted_pairs)
    scenarios = list(scenarios)
    matrix = np.array(raw_values)

    validate_attributes_matrix(matrix)

    # Costs are a negative criteria
    attributes_matrix = invert_costs(matrix);

    # Normalize column vectors of matrix
    attributes_norm = l2_norm(attributes_matrix)
    if attributes_norm is None:
        raise RuntimeError("❗Failed to normalize attribute values. Aborting score computation.\n")
    validate_attributes_matrix(attributes_norm)

    # Validate weights vector
    validate_weights();
    logger.info(f"Using weights: {WEIGHTS}")

    # Compute weighted attributes
    weighted_attributes = attributes_norm * config.WEIGHTS
    np.savetxt(f"{PROCESSED_DIR}/weighted_attributes.csv", weighted_attributes, delimiter=',', fmt='%10.5f')
    set_weighted_attributes(weighted_attributes)

    # Compute scores
    scores = compute_weighted_scores(attributes_norm, WEIGHTS)
    if scores is None:
        logger.error("❗Failed to compute scores. Aborting score computation.\n")
        return

    # Sort results
    sorted_scenario_scores = sort_scores(zip(scenarios, scores))
    set_sorted_scenario_scores(sorted_scenario_scores)
    print_scenario_scores(sorted_scenario_scores)
    write_scores_to_csv(PROCESSED_DIR, sorted_scenario_scores)

    logger.debug("⏳Saving processed data for analysis")
    np.savetxt(f"{PROCESSED_DIR}/l2norm.csv", attributes_norm, delimiter=',', fmt='%10.5f')
    set_attributes_norm(attributes_norm)



"""-------------------------
    Score Computation
-------------------------"""

def invert_costs(matrix):
    construction_idx = ATTRIBUTES_LIST.index("ConstructionCost")
    maintenance_idx = ATTRIBUTES_LIST.index("MaintenanceCost")
    A = matrix.copy()
    A[:, construction_idx] = -A[:, construction_idx]
    A[:, maintenance_idx] = -A[:, maintenance_idx]
    return A

def compute_weighted_scores(A_norm, weights):
    logger.debug("\n⏳Computing weighted scores")
    try:
        return A_norm @ weights
    except Exception as e:
        logger.error(f"....ERROR computing weighted scores: {e}")
        return None


def sort_scores(scenario_scores):
    if SORT_TYPE == 0:
        logger.debug(f"\t✔️  Sorted by Scenario name")
        return sorted(scenario_scores, key=lambda x: x[0].lower())
    elif SORT_TYPE == 1:
        logger.debug(f"\t✔️  Sorted by Score (ascending)")
        return sorted(scenario_scores, reverse=True, key=lambda x: x[1])
    elif SORT_TYPE == 2:
        logger.debug(f"\t✔️  Sorted by Score (descending)")
        return sorted(scenario_scores, key=lambda x: x[1])
    else:
        logger.error(f"\tERROR: Invalid sort index {args.sortindex}."
              f"See SORT_TYPE in config.py")
        logger.error("! Returning unsorted scores.")
        return scenario_scores

def print_scenario_scores(scenario_scores):
    logger.info("\n📊Weighted Scores:")
    for scenario, score in scenario_scores:
        logger.info(f"...{os.path.basename(scenario)}: {round(float(score), 6)}")
    logger.info("\nWeighted scores are between -1 and 1. "
          "A score of 0 means no change from current conditions.")


"""
Returns matrix after applying L2 normalization to columns.
"""
def l2_norm(A):
    logger.debug("⏳Performing L2 column normalization")
    # Calc L2 norm for each column
    norms = np.linalg.norm(A, axis=0)

    # Avoid dividing by 0 in all 0 column
    norms[norms == 0] = 1.0

    # Divide each element in a column by that column's L2 norm
    A_norm = np.round(A / norms, 4) # Norm and round
    return A_norm

"""
Alternate normalisation method
"""
def max_abs_norm(data):
    # Max absolute value in each column
    col_max_abs = np.max(np.abs(data), axis=0)

    # Avoid dividing by 0 in all 0 column
    col_max_abs[col_max_abs == 0] = 1.0

    # Divide each element in a column by that column's max absolute value
    normalized_data = data / col_max_abs
    return normalized_data

def init_process():
    global logger
    logger = get_logger()
