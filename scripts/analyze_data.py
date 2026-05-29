import csv
import os
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from config import *
from utils import *

logger = None

def analyze_data(showradar):
    # Load data from Process stage
    scenarios = get_scenario_names()
    if not scenarios:
        logger.error("❗Could not retrieve scenarios data from Config.")
        raise ValueError("Array of scenario names was empty.")
    weighted_attributes = get_weighted_attributes()
    sorted_scenario_scores = get_sorted_scenario_scores()

    # Sort by scenario name
    sorted_pairs = sorted(
        zip(scenarios, weighted_attributes),
        key=lambda x: x[0]
    )
    scenarios, weighted_attributes = zip(*sorted_pairs)
    scenarios = list(scenarios)
    weighted_attributes = np.array(weighted_attributes)

    # Plot weights & scores
    plot_weights(config.ATTRIBUTES_LIST, config.WEIGHTS)
    plot_scores(sorted_scenario_scores)

    # Radar chart analysis
    plot_radar_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, showradar)

    logger.warning(f"✅ Analysis complete.")


"""
Weights & Scores
"""
def plot_scores(scenario_scores):
    logger.debug("\n⏳Generating scenario score bar chart...")
    try:
        scenarios, scores = zip(*scenario_scores)
        plot_score_labels()
        plot_score_bars(scenarios, scores)
        save_png("weighted_scores", PNG_DIR)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


def plot_score_labels():
    plt.figure(figsize=(10, 6))
    plt.ylim(-1.0, 1.0)
    plt.xlabel('Scenario', fontweight='bold')
    plt.ylabel('Weighted Score', fontweight='bold')
    plt.title('Scenario Scores', fontsize=14)
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def plot_score_bars(scenarios, scores):
    bars = plt.bar(scenarios, scores, color='skyblue', edgecolor='navy')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center', fontsize=9)
    plt.tight_layout()

def plot_weights(attributes, weights):
    logger.debug("\n⏳Plotting weights distribution...")
    try:
#         plt.figure(figsize=(10, 6))
        plot_weight_labels(min(1, np.max(weights)*2))
        plot_weight_bars(attributes, weights)
        save_png("weight_distribution", PNG_DIR)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)

def plot_weight_labels(ylim):
    plt.figure(figsize=(10, 6))
    plt.xlabel('Attribute', fontweight='bold')
    plt.ylabel('Weight', fontweight='bold')
    plt.ylim(0, ylim)
    plt.title('Weight Distribution', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def plot_weight_bars(attributes, weights):
    bars = plt.bar(attributes, weights, color='skyblue', edgecolor='navy')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center', fontsize=9)
    plt.tight_layout() # Prevents label cutoff

def plot_weight_bars(attributes, weights):
    bars = plt.bar(attributes, weights, color='skyblue', edgecolor='navy')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center', fontsize=9)
    plt.tight_layout() # Prevents label cutoff


"""
Radar Charts
"""

"""
Radar Charts
"""

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
    plot_labels(ax, attributes, fig_title, angles, showradar)

    # Scenario + Control charts
    for i in range(m):
        fig_title = f"Scenario {i} vs Control Performance"
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plot_values(ax, scenarios[0], A[0].tolist(), angles)
        plot_values(ax, scenarios[i], A[i].tolist(), angles)
        plot_labels(ax, attributes, fig_title, angles, showradar)

    logger.debug("\t✔️  Finished generating radar charts")

def plot_values(ax, label, values, angles):
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.2)

def plot_labels(ax, labels, fig_title, angles, showradar):
    ylim = np.round(max(config.WEIGHTS),3)
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


def init_analyze():
    global logger
    logger = get_logger()
