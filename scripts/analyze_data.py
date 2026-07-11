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
    apply_lncs_formatting()
    generate_perf_heatmap(A_norm[1:], scenarios[1:], config.ATTRIBUTES_LIST, "norm_heatmap.png" )

    # Plot weights & scores
    plot_weights(config.ATTRIBUTES_LIST, get_weights())
    plot_scores(sorted_scenario_scores)

    # Radar & tornado chart analysis
    plot_scenario_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, False)

    logger.warning(f"✅ Analysis complete.")

"""
Heat Map
"""

def generate_perf_heatmap(A, scenario_names, attribute_names, filename="heatmap.png"):
    if A.shape != (len(scenario_names), len(attribute_names)):
        raise ValueError(f"Matrix shape ({A.shape}) does not match provided labels: "
                         f"({len(scenario_names)}x{len(attribute_names)}).")

    fig, ax = plt.subplots(figsize=(4.8, 2.2))
    cax = ax.imshow(A, cmap='PRGn', vmin=-1, vmax=1, aspect='auto')

    plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names)
    plot_heatmap_values(A, ax)

    fig.tight_layout(pad=0.5)
    save_path = os.path.join(config.PNG_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names):
    ax.set_title("Normalised Performance Values", fontsize=12, pad=20, weight='bold')
    ax.set_xlabel("Attributes", fontsize=10, labelpad=10)
    ax.set_ylabel("Scenarios", fontsize=10, labelpad=10)
    ax.set_xticks(np.arange(len(attribute_names)))
    ax.set_yticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(attribute_names)
    ax.set_yticklabels(scenario_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(attribute_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(scenario_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(cax, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label('Scale (-1 to 1)', rotation=270, labelpad=10, fontsize=8)


def plot_heatmap_values(A, ax):
    m,n = A.shape
    for i in range(m):
        for j in range(n):
            value = A[i, j]
            text_label = f"{value:.2f}"
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, text_label,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=7)


"""
Scenario Charts
"""
def plot_scenario_charts(scenarios, A, attributes, show_plots=False):
    logger.debug("\n⏳ Sorting scenarios and generating refined visualizations")
    control_name = scenarios[0]
    control_scores = A[0]
    m_test, n = A.shape
    ylim = 0.2
    plot_top_scores(A, scenarios, attributes, ylim, show_plots)
    norm = mcolors.Normalize(vmin=-ylim*1.1, vmax=ylim*1.1)
    cmap_diverging = plt.cm.get_cmap('PRGn')

    for i in range(m_test):
        current_name = scenarios[i]
        current_scores = A[i].tolist()
        plot_radar(n, control_name, control_scores, current_name, current_scores, attributes, ylim)

        # Tornado chart
        fig, ax_bar = plt.subplots(figsize=(4.8, 2.5))
        bar_colors = [cmap_diverging(norm(score)) for score in current_scores]
        y_pos_bar = np.arange(n)
        plot_tornado_labels(ax_bar, ylim, current_name, current_scores, bar_colors, y_pos_bar)
        text_padding = ylim * 0.04
        for idx in range(n):
            score = current_scores[idx]
            place_tornado_values(ax_bar, idx, attributes[idx], score, ylim, text_padding)
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)

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
    ax_bar.text(ylim * 1.02, idx, attr, ha='left', va='center',
                fontsize=7.5, fontweight='bold', color='#000000')
    if score >= 0:
        if score < ylim * 0.75:
            # Positive bar: place score numbers just to the right of the bar tip
            ax_bar.text(score + text_padding, idx, f"{score:+.3f}", ha='left', va='center',
                        fontsize=7, color='#222222')
        else:
            ax_bar.text(score - text_padding, idx, f"{score:+.3f}", ha='right', va='center',
                        fontsize=7, color='#ffffff')
    else:
        if score > -ylim * 0.75:
            # Negative bar: place score numbers just to the left of the bar tip
            ax_bar.text(score - text_padding, idx, f"{score:+.3f}", ha='right', va='center',
                        fontsize=7, color='#222222')
        else:
            ax_bar.text(score + text_padding, idx, f"{score:+.3f}", ha='left', va='center',
                        fontsize=7, color='#ffffff')



def plot_radar(n, control_name, control_scores, current_name, current_scores, attributes, ylim):
    fig, ax_radar = plt.subplots(figsize=(3.5,3.5), subplot_kw=dict(polar=True))
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
    ax_radar.legend(loc='center', bbox_to_anchor=(0.5, -0.12), fontsize=7, frameon=False)
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
    plt.figure(figsize=(4.8, 2.5))
    plt.ylim(-max_score * 1.5, max_score * 1.5)
    plt.ylabel('Score', fontweight='bold')
    plt.xlabel('Scenario', fontweight='bold')
    plt.title('Weighted Scenario Scores', fontsize=14)
    plt.xticks(rotation=30, ha='right')


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
    plt.figure(figsize=(4.8,2.2))
    plt.xlabel('Attribute', fontweight='bold')
    plt.ylabel('Weight', fontweight='bold')
    plt.ylim(0, 0.22)
    plt.xticks(rotation=30, ha='right')


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

def apply_lncs_formatting():
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.titlesize': 10
    })


def init_analyze():
    global logger
    logger = get_logger()
