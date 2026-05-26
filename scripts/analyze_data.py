import csv
import os
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from config import *
from utils import *

logger = None

def init_analyze():
    global logger
    logger = get_logger()


def analyze_data(showradar):
    # Load data from Process stage
    scenarios = get_scenarios()
    if not scenarios:
        logger.error("❗Could not retrieve scenarios data from Config.")
        raise ValueError("Array of scenario names was empty.")
    attributes_norm = get_attributes_norm()
    validate_attributes_matrix(attributes_norm)

    # Validate weights vector
    validate_weights();
    logger.info(f"Using weights: {WEIGHTS}")
    plot_weights(config.ATTRIBUTES_LIST, config.WEIGHTS)

    # Compute weighted attributes
    weighted_attributes = attributes_norm * config.WEIGHTS
    np.savetxt(f"{PROCESSED_DIR}/weighted_attributes.csv", weighted_attributes, delimiter=',', fmt='%10.5f')

    # Compute scores
    scores = compute_weighted_scores(attributes_norm, WEIGHTS)
    np.savetxt(f"{config.PROCESSED_DIR}/scores.csv", scores, delimiter=',', fmt='%2.5f')
    if scores is None:
        logger.error("❗Failed to compute scores. Aborting score computation.\n")
        return

    # Sort results
    sorted_scenario_scores = sort_scores(zip(scenarios, scores))
    write_scores_to_csv(TOTALS_FILEPATH, sorted_scenario_scores)

    # Plot scores
    plot_scores(sorted_scenario_scores)

    # Print results
    print_scenario_scores(sorted_scenario_scores)

    # Radar chart analysis
    plot_radar_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, showradar)



"""-------------------------
    Score Computation
-------------------------"""

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



"""-------------------------
    Analysis
-------------------------"""

def plot_scores(scenario_scores):
    logger.debug("\n⏳Generating scenario score bar chart...")

    try:
        scenarios, scores = zip(*scenario_scores)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenarios, scores, color='skyblue', edgecolor='navy')

        plt.ylim(-1.0, 1.0)
        plt.xlabel('Scenario', fontweight='bold')
        plt.ylabel('Weighted Score', fontweight='bold')
        plt.title('Analysis of Scenario Performance', fontsize=14)
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability

        # Add grid for easier value estimation (y-axis only)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval,
                     round(yval, 4), va='bottom', ha='center', fontsize=9)

        plt.tight_layout() # Prevents label cutoff
        png_file = "weighted_scores.png"
        plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
        logging.debug(f"\tSaved {png_file}")
        plt.close()

        logger.info(f"\t✔️  Chart successfully saved to {png_file}")

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)

def plot_weights(attributes, weights):
    logger.debug("\n⏳Plotting weights  vector...")
    ylim = min(1, np.max(weights)*2)

    try:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(attributes, weights, color='skyblue', edgecolor='navy')

        plt.xlabel('Attribute', fontweight='bold')
        plt.ylabel('Weight', fontweight='bold')
        plt.ylim(0, ylim)
        plt.title('Weight Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability

        # Add grid for easier value estimation (y-axis only)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval,
                     round(yval, 4), va='bottom', ha='center', fontsize=9)

        plt.tight_layout() # Prevents label cutoff
        png_file = "weights.png"
        plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
        logging.debug(f"....Saved {png_file}")
        plt.close()

        logger.info(f"\t✔️  Chart successfully saved to {png_file}")

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)



def plot_radar_charts(scenarios, A, attributes, showradar):
    validate_attributes_matrix(A)
    logger.debug("\n⏳Generating radar charts")

    m, n = A.shape
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # Combined Chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    fig_title = "Scenario Performance Comparison"
    for i in range(m):
        plot_values(ax, scenarios[i], A[i].tolist(), angles)

    plot_radar(ax, attributes, fig_title, angles, showradar)

    # Scenario + Control charts
    for i in range(m):
        fig_title = f"Scenario {i} vs Control Performance"
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plot_values(ax, scenarios[0], A[0].tolist(), angles)
        plot_values(ax, scenarios[i], A[i].tolist(), angles)
        plot_radar(ax, attributes, fig_title, angles, showradar)

    logger.debug("\t✔️  Finished generating radar charts")


def plot_values(ax, label, values, angles):
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.2)


def plot_radar(ax, labels, fig_title, angles, showradar):
    ylim = max(WEIGHTS)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
    ax.set_yticklabels([-ylim, "", "0 (Control)", "", ylim])
    ax.set_title(fig_title, fontsize=14)
    ax.legend(loc='lower left', bbox_to_anchor=(1., 0.))

    png_file = f"{fig_title.replace(' ', '_')}.png"
    plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
    logging.debug(f"....Saved {png_file}")

    if showradar:
        plt.tight_layout()
        plt.show()

def init_save_png():
    PNG_DIR = SAVE_DIR + "/png"
    if not os.path.exists(PNG_DIR):
        logger.debug(f"Creating PNG folder for radar charts")
        os.mkdir(PNG_DIR)

def init_save_dir():
    PNG_DIR = SAVE_DIR + "/png"
    if not os.path.exists(PNG_DIR):
        logger.debug(f"\nCreating PNG folder for radar charts")
        os.mkdir(PNG_DIR)
