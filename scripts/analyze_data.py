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
    scenarios, A_norm = get_attributes_norm(with_names=True)
    weighted_attributes = get_weighted_attributes(with_names=False)
    sorted_scenario_scores = get_sorted_scenario_scores()
    scenarios = list(scenarios)
    weighted_attributes = np.array(weighted_attributes)

    # Plot heat map of normalised performance scores
    generate_scenario_heatmap(A_norm[1:], scenarios[1:], config.ATTRIBUTES_LIST, config.PNG_DIR, "norm_heatmap.png" )

    # Plot weights & scores
    plot_weights(config.ATTRIBUTES_LIST, get_weights())
    plot_scores(sorted_scenario_scores)

    # Radar & tornado chart analysis
    plot_scenario_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, False)

    logger.warning(f"✅ Analysis complete.")

"""
Heat Map
"""

def generate_scenario_heatmap(A, scenario_names, attribute_names, png_dir, filename="heatmap.png"):
    if A.shape != (len(scenario_names), len(attribute_names)):
        raise ValueError(f"Matrix shape ({A.shape}) does not match provided labels: "
                         f"({len(scenario_names)}x{len(attribute_names)}).")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cax = ax.imshow(A, cmap='PRGn', vmin=-1, vmax=1, aspect='auto')
    plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names)
    fig.tight_layout()

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    save_path = os.path.join(png_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Matplotlib heatmap successfully saved to: {save_path}")


def plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names):
    ax.set_xticks(np.arange(len(attribute_names)))
    ax.set_yticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(attribute_names)
    ax.set_yticklabels(scenario_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(attribute_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(scenario_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(cax, ax=ax, pad=0.02)
    cbar.set_label('Normalized Value Scale (-1 to 1)', rotation=270, labelpad=15)
    ax.set_title("Scenario vs. Attribute Heatmap", fontsize=16, pad=20, weight='bold')
    ax.set_xlabel("Attributes", fontsize=12, labelpad=10)
    ax.set_ylabel("Scenarios", fontsize=12, labelpad=10)


"""
Scenario Charts
"""
def plot_scenario_charts(scenarios, A, attributes, show_plots=False):
    logger.debug("\n⏳ Sorting scenarios and generating refined visualizations")
    control_name = scenarios[0]
    control_scores = A[0]
    sorted_names, sorted_scores = sorted_attributes(A, scenarios)
    m_test, n = sorted_scores.shape
    ylim = 0.2
    plot_top_scores(sorted_scores, sorted_names, attributes, ylim, show_plots)
    norm = mcolors.Normalize(vmin=-ylim*1.1, vmax=ylim*1.1)
    cmap_diverging = plt.cm.get_cmap('PRGn')

    for i in range(m_test):
        current_name = sorted_names[i]
        current_scores = sorted_scores[i].tolist()
        plot_radar(n, control_name, control_scores, current_name, current_scores, attributes, ylim)

        # Tornado chart
        fig, ax_bar = plt.subplots(figsize=(7,5))
        bar_colors = [cmap_diverging(norm(score)) for score in current_scores]
        y_pos_bar = np.arange(n)
        plot_tornado_labels(ax_bar, ylim, current_name, current_scores, bar_colors, y_pos_bar)
        text_padding = ylim * 0.04
        for idx in range(n):
            score = current_scores[idx]
            place_tornado_values(ax_bar, idx, attributes[idx], score, ylim, text_padding)
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)
        fig.suptitle(f"Attribute Scores", fontsize=14, fontweight='bold', y=1.02)

        png_file = f"{current_name.replace(':', '')} - Tornado.png"
        plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
        if show_plots: plt.show()
        plt.close()

    logger.debug("\t✔️ Finished generating prioritized passports")

def plot_tornado_labels(ax_bar, ylim, current_name, current_scores, bar_colors, y_pos_bar):
    ax_bar.barh(y_pos_bar, current_scores, align='center', color=bar_colors, edgecolor='#555555', linewidth=0.5, alpha=0.9, height=0.55)
    ax_bar.axvline(0, color='#333333', linestyle='-', linewidth=1.5)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
    ax_bar.set_xticklabels([-ylim, "", f"({current_name})", "", ylim])
    ax_bar.grid(axis='x', linestyle='--', alpha=0.4)
    ax_bar.set_xlim([-ylim, ylim * 1.4])


def place_tornado_values(ax_bar, idx, attr, score, ylim, text_padding):
    ax_bar.text(ylim * 1.05, idx, attr, ha='left', va='center',
                fontsize=9.5, fontweight='bold', color='#000000')
    if score >= 0:
        if score < ylim * 0.75:
            # Positive bar: place score numbers just to the right of the bar tip
            ax_bar.text(score + text_padding, idx, f"{score:+.4f}", ha='left', va='center',
                        fontsize=8.5, fontweight='semibold', color='#222222')
        else:
            ax_bar.text(score - text_padding, idx, f"{score:+.4f}", ha='right', va='center',
                        fontsize=8.5, fontweight='semibold', color='#ffffff')
    else:
        if score > -ylim * 0.75:
            # Negative bar: place score numbers just to the left of the bar tip
            ax_bar.text(score - text_padding, idx, f"{score:+.4f}", ha='right', va='center',
                        fontsize=8.5, fontweight='semibold', color='#222222')
        else:
            ax_bar.text(score + text_padding, idx, f"{score:+.4f}", ha='left', va='center',
                        fontsize=8.5, fontweight='semibold', color='#ffffff')


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
    fig, ax = plt.subplots(figsize=(4,7))
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
                color=green_shades[idx], label=f"#{idx+1} - {sorted_names[idx]}", alpha=0.9)


def plot_top_labels(ax, y_pos, attributes, ylim, top_k):
    ax.set_title(f"Top {top_k} Scenarios Attribute Scores", fontsize=14, pad=15)
    ax.axvline(0, color='#333333', linestyle='-', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attributes, fontsize=10)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_yaxis()
    ax.set_xlim([-ylim, ylim])
    ax.set_xlabel("Score Contribution ", fontsize=11)
    ax.grid(axis='x', alpha=0.5)
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2))


def plot_radar(n, control_name, control_scores, current_name, current_scores, attributes, ylim):
    fig, ax_radar = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    control_radar_vals = control_scores.tolist() + [control_scores[0]]

    ax_radar.plot(angles, control_radar_vals, linewidth=2, label=f"{control_name} (Baseline)")
    ax_radar.fill(angles, control_radar_vals, alpha=0.25)
    active_radar_vals = current_scores + [current_scores[0]]
    ax_radar.plot(angles, active_radar_vals, linewidth=2.5, label=current_name)
    ax_radar.fill(angles, active_radar_vals, alpha=0.15)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(attributes, fontsize=8)
    ax_radar.set_yticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
    ax_radar.set_yticklabels(["", "", "(Control)", "", ""])
    ax_radar.legend(loc='center', bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"Performance Profile", fontsize=14, fontweight='bold', y=1.02)
    png_file = f"{current_name.replace(':', '')} - Radar.png"
    plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
    plt.close()


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
        plot_score_bars(scenarios[1:], scores[1:], max_score)
        save_png("Scenario Scores", PNG_DIR)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


def plot_score_labels(max_score):
    plt.figure(figsize=(8,6))
    plt.ylim(-max_score * 1.5, max_score * 1.5)
    plt.ylabel('Score', fontweight='bold')
    plt.xlabel('Scenario', fontweight='bold')
    plt.title('Weighted Scenario Scores', fontsize=14)
    plt.xticks(rotation=45, ha='right')


def plot_score_bars(scenarios, scores, max_score):
    norm = mcolors.Normalize(vmin=-max_score * 1.5, vmax=max_score * 1.5)
    cmap_scores = plt.cm.get_cmap('RdYlBu')
    bar_colors = [cmap_scores(norm(score)) for score in scores]
    short_names = scenarios
    bars = plt.bar(short_names, scores, color=bar_colors, edgecolor='#555555', linewidth=0.7, alpha=0.9)
    plt.axhline(0, color='#333333', linestyle='-', linewidth=1.2)

    for bar in bars:
        yval = bar.get_height()
        va_dir = 'bottom' if yval >= 0 else 'top'
        offset = 0.02 if yval >= 0 else -0.05
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset,
                 f"{yval:.4f}", va=va_dir, ha='center')
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
    plt.figure(figsize=(7,4))
    plt.xlabel('Attribute', fontweight='bold')
    plt.ylabel('Weight', fontweight='bold')
    plt.ylim(0, 0.25)
    plt.title('Weight Distribution', fontsize=14)
    plt.xticks(rotation=45, ha='right')


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
