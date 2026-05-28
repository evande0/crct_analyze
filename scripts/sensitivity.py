import os
import argparse
import numpy as np
import csv
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from config import *
import utils
from extract_data import init_extract, extract_all_data
import process_data as proc

logger = None
scenario_names = None
attributes_raw = None
attributes_norm = None
baseline_winner = None
base_weights = None

"""
Run sensitivity analysis
"""
def run_sensitivity_analysis(step_size=TARGET_STEP_SIZE):
    logger.warning(f"\n⏳ Starting Criteria Sensitivity Analysis (Step Size: {TARGET_STEP_SIZE})")
    # Sweep and record results
    with open(SENS_FILEPATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["TargetAttr", "TargetWeight", "RankReversed", "NewWinner", "Spread"])
        sweep_attribute_weights(writer, attributes_norm)

    logger.warning(f"🎉 Sensitivity analysis complete. Results saved to: {SENS_DIR}\n")

def sweep_attribute_weights(writer, attributes_norm):
    test_weights = np.arange(0.0, 1.0 + TARGET_STEP_SIZE, TARGET_STEP_SIZE)
    for target_idx, target_attr in enumerate(ATTRIBUTES_LIST):
        logger.info(f"\n⚖️  Sweeping weights for attribute: {target_attr} ({target_idx})")
        weights = get_attr_start_weights(target_idx)
        attr_winners = []
        while weights[target_idx] < 1 + TARGET_STEP_SIZE:
            this_winner, score_spread = calculate_scores(writer, attributes_norm, weights)
            rank_reversal = "YES" if this_winner != baseline_winner else "NO"
            attr_winners.append(this_winner)
            writer.writerow([target_attr, round(weights[target_idx], 4), rank_reversal, this_winner, round(score_spread, 4)])
            weights = increment_weights(weights, target_idx)
        plot_stability(target_attr, test_weights, attr_winners, SENS_DIR)


"""
Weights & Scores
"""

def get_attr_start_weights(target_idx):
    weights = base_weights.copy()
    weights[target_idx] = 0
    logger.debug(f"...Starting weights:\n{weights}")
    return weights

def increment_weights(weights, target_idx):
    for i in range(len(ATTRIBUTES_LIST)):
        if i == target_idx:
            weights[target_idx] += TARGET_STEP_SIZE
        else:
            non_target_weight = weights[i] - NON_TARGET_STEP_SIZE
            weights[i] = max(non_target_weight, 0)

    # Skip print for last iteration
    if weights[target_idx] < 1.05:
        w_sum = round(weights.sum(),4)
        logger.debug(f"   Incremented weights: {weights}, sum = {w_sum}")
    return weights

def calculate_scores(writer, attributes_norm, attr_weights):
    if not np.isclose(attr_weights.sum(), 1.0):
        raise ValueError("Something is wrong with the weights vector.")
    scores = proc.compute_weighted_scores(attributes_norm, attr_weights)
    sorted_indices = np.argsort(scores)[::-1]
    score_spread = float(scores[sorted_indices[0]] - scores[sorted_indices[-1]])
    this_winner = scenario_names[sorted_indices[0]]
    return this_winner, score_spread


"""
Plots
"""
def plot_stability(attr_name, weights, winners, save_path):
    logger.info(f"...Plotting stability for {attr_name}")
    plt.figure(figsize=(10, 2))
    unique_winners, winner_colors = get_unique_winner_colors(winners);
    plot_winners(weights, winners, winner_colors)
    plot_labels(attr_name)
    plot_legend(unique_winners, winner_colors, attr_name, save_path)

def get_unique_winner_colors(winners):
    # Map unique winners to specific tracking colors
    unique_winners = list(set(winners))
    color_map = plt.get_cmap('Set3', len(unique_winners))
    winner_colors = {winner: color_map(i) for i, winner in enumerate(unique_winners)}
    return unique_winners, winner_colors

def plot_winners(weights, winners, winner_colors):
    for idx, (w, winner) in enumerate(zip(weights, winners)):
        edgecolor = 'black' if winner != baseline_winner else 'navy'
        hatch = '//' if winner != baseline_winner else ''
        plt.barh(y=0, width=0.05, left=w-0.025, color=winner_colors[winner],
                 edgecolor=edgecolor, hatch=hatch, height=0.4)

def plot_labels(attr_name):
    plt.title(f"Stability Map: Swapping Weights for {attr_name}", fontsize=11, fontweight='bold')
    plt.xlabel("Assigned Attribute Weight")
    plt.xlim(-TARGET_STEP_SIZE, 1 + TARGET_STEP_SIZE)
    plt.yticks([])

def plot_legend(unique_winners, winner_colors, attr_name, save_path):
    legend_elements = [Patch(facecolor=winner_colors[w], edgecolor='black', label=f"Winner: {os.path.basename(w)}") for w in unique_winners]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(f"{save_path}/stability_{attr_name}.png", bbox_inches='tight')
    plt.close()


"""
Initialize sensitivity analysis
"""
def init_sensitivity():
    init_logger()
    init_dirs()
    init_extract()
    proc.init_process()
    load_data()
    init_baseline_winner()
    init_base_weights()

def init_logger():
    global logger
    logger = utils.get_logger()
    if logger is None:
        logger = utils.init_logging(LOG_FILE, verbose=True)
        utils.set_logger(logger)
    logger.debug("Logger initiatied")

def init_dirs():
    utils.create_dirs(png=False, processed=False, sens=True)

def load_data():
    global scenario_names, attributes_raw, attributes_norm
    scenario_names, attributes_raw = extract_all_data(PROJ_DIR)
    attributes_norm = proc.normalize_attributes(attributes_raw)

def init_baseline_winner():
    global baseline_winner
    logger.info("...Computing baseline scores using even weights")
    baseline_scores = proc.compute_weighted_scores(attributes_norm, FLAT_WEIGHTS)
    baseline_ranking = [scenario_names[idx] for idx in np.argsort(baseline_scores)[::-1]]
    logger.info(f"\t✔️  Computed baseline ranking: {baseline_ranking}")
    baseline_winner = baseline_ranking[0]
    logger.info(f"Baseline Winner: {baseline_winner}")

def init_base_weights():
    num_attributes = len(ATTRIBUTES_LIST)
    step_size = TARGET_STEP_SIZE
    non_target_count = num_attributes - 1
    target_start = 0
    non_target_start = 1 / non_target_count
    non_target_step = step_size / non_target_count
    global base_weights
    base_weights = np.full(num_attributes, non_target_start)
    # Target attribute still needs to be set to 0
    logger.debug(f"Base Weights: {base_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run independent criteria sensitivity analysis sweeps.")
    parser.add_argument("-s", "--step", type=float, default=TARGET_STEP_SIZE, help="Weight perturbation increment value (defaults to TARGET_STEP_SIZE in config.py)")
    parser.add_argument("-v", "--verbose", action="store_true", help=HELP_VERBOSE)
    parser.add_argument("-d", "--debug", action="store_true", help=HELP_DEBUG)
    parser.add_argument("-q", "--quiet", action="store_true", help=HELP_QUIET)

    init_sensitivity()
    args = parser.parse_args()
    logger.debug(f"\nArgs: {args}\n")

    run_sensitivity_analysis(step_size=args.step)
