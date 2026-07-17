import os
import argparse
import numpy as np
import csv
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from config import *
import src.utils as utils
from src.extract_data import init_extract, extract_all_data
import src.process_data as proc

logger = None
scenario_names = None
attributes_raw = None
attributes_norm = None
baseline_winner = None
base_weights = None


"""
Run sensitivity analysis

"""
def run_sensitivity_analysis(step_size=TARGET_STEP_SIZE, pipeline=False):
    logger.warning(f"⏳ Starting Criteria Sensitivity Analysis (Step Size: {TARGET_STEP_SIZE})")
    load_data(pipeline)
    with open(SENS_FILEPATH, "w", newline="", encoding="utf-8") as csvfile:
        sweep_attribute_weights(init_writer(csvfile), attributes_norm)

    logger.warning("🎉 Sensitivity Analysis completed successfully.")
    logger.warning(f"📁 See results in {SENS_DIR}\n")

def sweep_attribute_weights(writer, attributes_norm):
    # For each attribute, get it's starting weights
    for target_idx, target_attr in enumerate(ATTRIBUTES_LIST):
        logger.info(f"\tTesting the sensitivity of {target_attr}...")
        weights = get_attr_start_weights(target_idx)
        attr_winners = []
        attr_scores = []
        first_reverse = None
        reverse_score = None
        new_winner = None
        # Find scores & winner, increment weights by step size
        while weights[target_idx] < 1 + TARGET_STEP_SIZE:
            scores = calculate_scores(writer, attributes_norm, weights)
            score_spread, this_winner, rank_reversal = analyze_scores(scores)
            if rank_reversal == "YES" and first_reverse is None:
                first_reverse = weights[target_idx]
                new_winner = this_winner
                winner_idx = scenario_names.index(this_winner)
                reverse_score = scores[winner_idx]
            attr_scores.append(scores)
            attr_winners.append(this_winner)
            writer.writerow([target_attr, round(weights[target_idx], 4), rank_reversal, this_winner, round(score_spread, 4)])
            weights = increment_weights(weights, target_idx)
        plot_stability(target_attr, attr_scores, SENS_DIR, first_reverse, reverse_score, new_winner)


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
    return np.round(weights, 10)

def calculate_scores(writer, attributes_norm, attr_weights):
    if not np.isclose(attr_weights.sum(), 1.0):
        raise ValueError(f"Something is wrong with the weights vector: {attr_weights} sum={attr_weights.sum()}")
    scores = proc.compute_weighted_scores(attributes_norm, attr_weights)
    return scores

def analyze_scores(scores):
    sorted_indices = np.argsort(scores)[::-1]
    score_spread = float(scores[sorted_indices[0]] - scores[sorted_indices[-1]])
    this_winner = scenario_names[sorted_indices[0]]
    rank_reversal = "YES" if this_winner != baseline_winner else "NO"
    return score_spread, this_winner, rank_reversal

def plot_stability(attr_name, all_scores, save_path, reversal_weight, reversal_score, new_winner):
    scores_matrix = np.array(all_scores)
    plot_lines(scores_matrix, reversal_weight, reversal_score, new_winner)
    plot_labels(attr_name)
    utils.save_png(f"sensitivity_{attr_name}", SENS_DIR)
    plt.close()

def plot_lines(scores_matrix, reversal_weight, baseline_score, new_winner):
    test_weights = np.arange(0.0, 1.0 + TARGET_STEP_SIZE, TARGET_STEP_SIZE)
    plt.figure(figsize=(10, 4))
    for idx, scenario in enumerate(scenario_names):
        scenario_scores = scores_matrix[:, idx]
        linewidth = 2.5 if scenario == baseline_winner else 1.5
        linestyle = '-' if scenario == baseline_winner else '--'
        plt.plot(test_weights, scenario_scores, label=os.path.basename(scenario), linewidth=linewidth, linestyle=linestyle)

    if reversal_weight is not None:
        plt.axvline(x=reversal_weight, color='red', linestyle='-', alpha=0.2,
                    label=f'Rank Reversal (weight={reversal_weight:.2f})')

        # Highlight the intersection point on the baseline curve
        plt.scatter(reversal_weight, baseline_score, color='red', s=30, zorder=5)

        # Add a text annotation near the point
        plt.annotate(
            f"Reversal @ {reversal_weight:.2f}\n{new_winner}",
            xy=(reversal_weight, baseline_score),
            xytext=(reversal_weight - 0.3 , baseline_score + 0.3), # Slightly offset the text
            arrowprops=dict(arrowstyle="->", color='red', lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="red"),
            fontsize=9
        )

def plot_labels(attr_name):
    plt.title(f"Sensitivity to {attr_name} Weight", fontsize=12, fontweight='bold')
    plt.xlabel(f"Assigned Weight for {attr_name} (0.0 to 1.0)", fontweight='bold')
    plt.ylabel("Final Weighted SAW Score", fontweight='bold')
    plt.xlim(0.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title="Scenarios")
    plt.tight_layout()

def save_plot_png(file_name, save_path):
    png_file = f"{file_name}.png"
    plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
    logger.info(f"\t✔️  Chart successfully saved to {png_file}")


"""
Initialize sensitivity analysis
"""
def init_sensitivity(args, pipeline=False):
    init_logger(args)
    if not pipeline:
        utils.create_dirs(png=False, processed=False, sens=True)

def init_logger(args):
    global logger
    logger = utils.get_logger()
    if logger is None:
        logger = utils.init_logging(LOG_FILEPATH, verbose=args.verbose, debug=args.debug, quiet=args.quiet)
        utils.set_logger(logger)
    logger.debug("Logger initiatied")

def load_data(pipeline):
    load_attr_data(pipeline)
    set_baseline_winner()
    set_base_weights()

def load_attr_data(pipeline):
    global scenario_names, attributes_raw, attributes_norm
    if (pipeline): # If run from the pipeline, values should be saved in the config
        scenario_names = utils.get_scenario_names()
        attributes_raw = utils.get_raw_attributes()
    else:
        init_for_extract()
        scenario_names, attributes_raw = extract_all_data(PROJ_DIR)
    attributes_norm = proc.normalize_attributes(attributes_raw, scenario_names)

def init_for_extract():
    utils.create_dirs(sens=True, processed=False, png=False)
    utils.setup_totals_file()
    init_extract()
    proc.init_process()

def set_baseline_winner():
    global baseline_winner
    logger.info("...Computing baseline scores using even weights")
    baseline_scores = proc.compute_weighted_scores(attributes_norm, WEIGHT_OPTS["FLAT"])
    baseline_ranking = [scenario_names[idx] for idx in np.argsort(baseline_scores)[::-1]]
    logger.info(f"\t✔️  Computed baseline ranking: {baseline_ranking}")
    baseline_winner = baseline_ranking[0]
    logger.info(f"\t✔️  Baseline Winner: {baseline_winner}")

def set_base_weights():
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

def init_writer(csvfile):
    writer = csv.writer(csvfile)
    writer.writerow(["TargetAttr", "TargetWeight", "RankReversed", "NewWinner", "Spread"])
    return writer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run independent criteria sensitivity analysis sweeps.")
    parser.add_argument("-s", "--step", type=float, default=TARGET_STEP_SIZE, help="Weight perturbation increment value (defaults to TARGET_STEP_SIZE in config.py)")
    parser.add_argument("-v", "--verbose", action="store_true", help=HELP_VERBOSE)
    parser.add_argument("-d", "--debug", action="store_true", help=HELP_DEBUG)
    parser.add_argument("-q", "--quiet", action="store_true", help=HELP_QUIET)
    args = parser.parse_args()

    init_sensitivity(args, pipeline=False)
    logger.debug(f"\nArgs: {args}\n")

    run_sensitivity_analysis(step_size=args.step)
