import os
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
from config import *
import utils
from process_data import invert_costs, l2_norm, compute_weighted_scores

logger = None

def run_sensitivity_analysis(step_size=0.05, extract=False):
    global logger
    logger = utils.get_logger()
    if logger is None:
        logger = utils.init_logging(LOG_FILE, verbose=True)
        utils.set_logger(logger)
    logger.warning(f"\n⏳ Starting Criteria Sensitivity Analysis (Step Size: {step_size})")

    # Load data
    scenarios, matrix = utils.load_totals(TOTALS_FILEPATH, extract)

    # Process data
    logger.info("...Processing data")
    attributes_matrix = invert_costs(matrix)
    attributes_norm = l2_norm(attributes_matrix)
    num_scenarios = len(scenarios)
    num_attributes = len(ATTRIBUTES_LIST)

    # Establish baseline rankings using even weights
    logger.info("...Computing baseline scores using even weights")
    baseline_scores = compute_weighted_scores(attributes_norm, FLAT_WEIGHTS)
    baseline_ranking = [scenarios[idx] for idx in np.argsort(baseline_scores)[::-1]]
    logger.debug(f"\t✔️  Computed baseline ranking: {baseline_ranking}")
    baseline_winner = baseline_ranking[0]
    logger.info(f"Baseline Winner: {baseline_winner}")

    # Init sensitivity analysis
    sensitivity_dir = f"{SAVE_DIR}/sensitivity"
    os.makedirs(sensitivity_dir, exist_ok=True)
    summary_filepath = f"{sensitivity_dir}/sensitivity_summary.csv"
    last_weights = SENSITIVITY_START_WEIGHTS

    with open(summary_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Target_Attribute", "Test_Weight", "Rank_Reversal_Occurred", "New_Winner", "Score_Spread"])

        # 3. Iterate through each attribute independently
        for target_idx, target_attr in enumerate(ATTRIBUTES_LIST):
            logger.info(f"\n⚖️ Sweeping weights for attribute: {target_attr}")

            # Generate the test interval from 0.0 to 1.0 using the designated step size
            test_weights = np.arange(0.0, 1.0 + step_size, step_size)
            test_weights = np.clip(test_weights, 0.0, 1.0) # Guard against floating point creep
            logger.warn(f"Test weights: {test_weights}")

            attr_winners = []
            attr_weight_profiles = []

            for tw in test_weights:
                # Copy original weight array
                w_new = last_weights.copy()
                orig_w = w_new[target_idx]
                non_target_count = num_attributes - 1
                distributed_step = step_size / non_target_count

                # Assign the adjusted target weight
                w_new[target_idx] = tw

#                 # Proportional redistribution to preserve the 1.0 sum boundary
#                 remaining_mass = 1.0 - tw
#                 orig_remaining_mass = 1.0 - orig_w

                if np.isclose(orig_remaining_mass, 0.0):
                    # Edge case: If the original weight was 1, distribute remainder equally

                    for i in range(num_attributes):
                        if i != target_idx:
                            w_new[i] = remaining_mass / non_target_count
                else:
                    # Scale remaining elements proportionally
                    for i in range(num_attributes):
                        if i != target_idx:
                            w_new[i] = w_new[i]

                # Force precise mathematical closing sum to clear validate_weights conditions
                if not np.isclose(w_new.sum(), 1.0):
                    w_new /= w_new.sum()

                # Compute alternative scores with the modified weight distribution profile
                scores = compute_weighted_scores(attributes_norm, w_new)
                sorted_indices = np.argsort(scores)[::-1]
                new_winner = scenarios[sorted_indices[0]]

                rank_reversal = "YES" if new_winner != baseline_winner else "NO"
                score_spread = float(scores[sorted_indices[0]] - scores[sorted_indices[-1]])

                writer.writerow([target_attr, round(tw, 4), rank_reversal, new_winner, round(score_spread, 4)])
                attr_winners.append(new_winner)
                attr_weight_profiles.append(w_new)

            # 4. Generate Visual Stability Plot for the attribute profile
            plot_attribute_stability(target_attr, test_weights, attr_winners, baseline_winner, sensitivity_dir)

    logger.warning(f"🎉 Sensitivity analysis complete. Results saved to: {sensitivity_dir}\n")

def plot_attribute_stability(attr_name, weights, winners, baseline_winner, save_path):
    plt.figure(figsize=(10, 2))

    # Map unique winners to specific tracking colors
    unique_winners = list(set(winners))
    color_map = plt.cm.get_cmap('Set3', len(unique_winners))
    winner_colors = {winner: color_map(i) for i, winner in enumerate(unique_winners)}

    for idx, (w, winner) in enumerate(zip(weights, winners)):
        edgecolor = 'black' if winner != baseline_winner else 'navy'
        hatch = '//' if winner != baseline_winner else ''
        plt.barh(y=0, width=0.05, left=w-0.025, color=winner_colors[winner],
                 edgecolor=edgecolor, hatch=hatch, height=0.4)

    plt.title(f"Stability Map: Swapping Weights for {attr_name}", fontsize=11, fontweight='bold')
    plt.xlabel("Assigned Attribute Weight")
    plt.xlim(-0.05, 1.05)
    plt.yticks([])

    # Create clean legend handles
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=winner_colors[w], edgecolor='black', label=f"Winner: {os.path.basename(w)}") for w in unique_winners]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.savefig(f"{save_path}/stability_{attr_name}.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run independent criteria sensitivity analysis sweeps.")
    parser.add_argument("-s", "--step", type=float, default=0.05, help="Weight perturbation increment value (default: 0.05)")
    parser.add_argument("-x","--extract", action="store_true", help=HELP_EXTRACT)
    args = parser.parse_args()

    # Use fallback configuration directories if running completely isolated
    if not os.path.exists(TOTALS_FILEPATH):
        print(f"❗ Error: Extracted pipeline metrics not found at target location: {TOTALS_FILEPATH}")
        print("Please run 'python3 run_pipeline.py' at least once to generate initial data matrices.")
    else:
        run_sensitivity_analysis(step_size=args.step, extract=args.extract)
