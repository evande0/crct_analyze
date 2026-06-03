import csv
import os
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

    # Radar & tornado chart analysis
    plot_comprehensive_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, False)

    logger.warning(f"✅ Analysis complete.")


"""
Scenario Charts
"""
def plot_comprehensive_charts(scenarios, A, attributes, show_plots=False):
    logger.debug("\n⏳ Sorting scenarios and generating refined visualizations")
    control_name = scenarios[0]
    control_scores = A[0]
    sorted_names, sorted_scores = sorted_attributes(A, scenarios)

    m_test, n = sorted_scores.shape
    ylim = np.round(max(config.WEIGHTS), 3)

    plot_top_scores(sorted_scores, sorted_names, attributes, ylim, show_plots)

    norm = mcolors.Normalize(vmin=-ylim, vmax=ylim)
    cmap_diverging = plt.cm.get_cmap('RdYlBu')

    # Scenario radar & tornado charts
    for i in range(m_test):
        current_name = sorted_names[i]
        current_scores = sorted_scores[i].tolist()

        fig = plt.figure(figsize=(16, 6))
        grid = plt.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.2)        # Scenario radar
        plot_radar(fig, grid, n, control_name, control_scores, current_name, current_scores, attributes, ylim)

        # Tornado chart - Decompose
        ax_bar = fig.add_subplot(grid[1])
        y_pos_bar = np.arange(n)
        bar_colors = [cmap_diverging(norm(score)) for score in current_scores]
        ax_bar.barh(y_pos_bar, current_scores, align='center', color=bar_colors, edgecolor='#555555', linewidth=0.5, alpha=0.9, height=0.55)

        ax_bar.axvline(0, color='#333333', linestyle='-', linewidth=1.5)
        ax_bar.set_yticks(y_pos_bar)
        ax_bar.set_yticklabels([]) # We drop standard y-ticks to draw them manually in the center
        ax_bar.invert_yaxis()
        ax_bar.set_xlim([-ylim, ylim])
        ax_bar.set_xticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
        ax_bar.set_xticklabels([-ylim, "", "Scenario 0: Control (Do Nothing)", "", ylim])
        ax_bar.grid(axis='x', linestyle='--', alpha=0.4)
        for idx in range(n):
            ax_bar.text(ylim, idx, attributes[idx], ha='left', va='center',
                        fontsize=9.5, fontweight='bold', color='#222222')
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)
        fig.suptitle(f"{current_name} Performance Overview", fontsize=14, fontweight='bold', y=1.02)

        png_file = f"{current_name.replace(':', '')} Performance Overview.png"
        plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
        if show_plots: plt.show()
        plt.close()

    logger.debug("\t✔️ Finished generating prioritized passports")


def sorted_attributes(A, scenarios):
    test_names = np.array(scenarios[1:])
    test_scores = A[1:]
    row_sums = np.sum(test_scores, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]
    sorted_names = test_names[sorted_indices].tolist()
    sorted_scores = test_scores[sorted_indices]
    return sorted_names, sorted_scores


def plot_top_scores(sorted_scores, sorted_names, attributes, ylim, show_plots):
    m_test, n = sorted_scores.shape
    top_k = min(3, m_test)
    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(n)

    plot_top_bars(ax, top_k, sorted_scores, sorted_names, y_pos)
    plot_top_labels(ax, y_pos, attributes, ylim, top_k)

    plt.savefig(f"{PNG_DIR}/Top_Scenarios.png", bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def plot_top_bars(ax, top_k, sorted_scores, sorted_names, y_pos):
    bar_height = 0.25
    green_shades = [plt.cm.Greens(val) for val in np.linspace(0.9, 0.5, top_k)]
    for idx in range(top_k):
        offset = (idx - (top_k - 1) / 2) * bar_height
        ax.barh(y_pos + offset, sorted_scores[idx], bar_height,
                color=green_shades[idx], label=f"Rank {idx+1} ({sorted_names[idx]})", alpha=0.9)


def plot_top_labels(ax, y_pos, attributes, ylim, top_k):
    ax.set_title(f"Top {top_k} Scenarios by Attribute", fontsize=14, pad=15)
    ax.axvline(0, color='#333333', linestyle='-', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attributes, fontsize=10)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_yaxis()
    ax.set_xlim([-ylim, ylim])
    ax.set_xlabel("◄  Losses   │   Gains  ►", fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.5)
    ax.legend(loc='upper right')


def plot_radar(fig, grid, n, control_name, control_scores, current_name, current_scores, attributes, ylim):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    ax_radar = fig.add_subplot(grid[0], projection='polar')
    control_radar_vals = control_scores.tolist() + [control_scores[0]]

    ax_radar.plot(angles, control_radar_vals, linewidth=2, label=f"{control_name} (Baseline)")
    ax_radar.fill(angles, control_radar_vals, alpha=0.25)
    active_radar_vals = current_scores + [current_scores[0]]
    ax_radar.plot(angles, active_radar_vals, linewidth=2.5, label=current_name)
    ax_radar.fill(angles, active_radar_vals, alpha=0.15)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(attributes, fontsize=8)
    ax_radar.set_yticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
    ax_radar.set_yticklabels(["", "", "0 (Control)", "", ""])
    ax_radar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9)


"""
Weights & Scores
"""
def plot_scores(scenario_scores):
    logger.debug("\n⏳ Generating scenario score bar chart...")
    try:
        scenarios, scores = zip(*scenario_scores)
        max_score = max(abs(s) for s in scores)
        max_weight = max(abs(w) for w in config.WEIGHTS)

        plot_score_labels(max_score)
        plot_score_bars(scenarios, scores, max_score)
        save_png("Scenario Scores", PNG_DIR)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


def plot_score_labels(max_score):
    plt.figure(figsize=(10, 6))
    plt.ylim(-max_score * 1.5, max_score * 1.5)
    plt.xlabel('Scenario', fontweight='bold')
    plt.ylabel('Weighted Score', fontweight='bold')
    plt.title('Scenario Scores', fontsize=14)
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)


def plot_score_bars(scenarios, scores, max_score):
    norm = mcolors.Normalize(vmin=-max_score * 1.5, vmax=max_score * 1.5)
    cmap_scores = plt.cm.get_cmap('RdYlBu')
    bar_colors = [cmap_scores(norm(score)) for score in scores]

    bars = plt.bar(scenarios, scores, color=bar_colors, edgecolor='#555555', linewidth=0.7, alpha=0.9)
    plt.axhline(0, color='#333333', linestyle='-', linewidth=1.2)

    for bar in bars:
        yval = bar.get_height()
        va_dir = 'bottom' if yval >= 0 else 'top'
        offset = 0.02 if yval >= 0 else -0.05
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset,
                 f"{yval:.4f}", va=va_dir, ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()


def plot_weights(attributes, weights):
    logger.debug("\n⏳ Plotting weights distribution...")
    try:
        max_weight = (np.max(weights)) * 1.2
        plot_weight_labels(min(1, max_weight))
        plot_weight_bars(attributes, weights, max_weight)
        save_png("Weights", PNG_DIR)
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


def plot_weight_bars(attributes, weights, max_weight):
    norm_weights = mcolors.Normalize(vmin=-max_weight * 0.2, vmax=max_weight*1.2)
    cmap_weights = plt.cm.get_cmap('Blues')

    bar_colors = [cmap_weights(norm_weights(w)) for w in weights]
    dark_edge = cmap_weights(0.9)

    bars = plt.bar(attributes, weights, color=bar_colors, edgecolor=dark_edge, linewidth=0.8, alpha=0.9)

    text_offset = max_weight * 0.015
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + text_offset,
                 f"{yval:.4f}", va='bottom', ha='center', fontsize=9)
    plt.tight_layout()


def init_analyze():
    global logger
    logger = get_logger()
