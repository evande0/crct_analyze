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
    scenarios = get_scenarios()
    totals = get_totals()
    if scenarios is None or totals is None:
        logger.warn("No totals saved. Extracting totals from CSV instead...")
        logger.debug(f"\nProcessing totals file:\n\t{TOTALS_FILEPATH}")
        scenarios, matrix = load_totals_csv(TOTALS_FILEPATH)
    else:
        matrix = np.array(get_totals(), dtype=float)

    validate_attributes_matrix(matrix)

    # Costs are a negative criteria
    attributes_matrix = invert_costs(matrix);

    # Normalize column vectors of matrix
    attributes_norm = l2_norm(attributes_matrix)
    if attributes_norm is None:
        raise RuntimeError("❗Failed to normalize attribute values. Aborting score computation.\n")

    logger.debug("...Saving processed data for analysis")
    np.savetxt(f"{PROCESSED_DIR}/l2norm.csv", attributes_norm, delimiter=',', fmt='%10.5f')
    set_attributes_norm(attributes_norm)


def invert_costs(matrix):
    construction_idx = ATTRIBUTES_LIST.index("ConstructionCost")
    maintenance_idx = ATTRIBUTES_LIST.index("MaintenanceCost")
    A = matrix.copy()
    A[:, construction_idx] = -A[:, construction_idx]
    A[:, maintenance_idx] = -A[:, maintenance_idx]
    return A


"""
Returns matrix after applying L2 normalization to columns.
"""
def l2_norm(A):
    logger.debug("...Performing L2 column normalization")
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
