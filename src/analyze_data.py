import csv
import os
import argparse
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from config import *
from src.utils import *

CMAP = mpl.colormaps[config.COLOR_MAP]
logger = None

def analyze_data(show, compact):
    # Load data from Process stage
    scenarios, A_norm = get_attributes_norm(with_names=True)
    weighted_attributes = get_weighted_attributes(with_names=False)
    sorted_scenario_scores = get_sorted_scenario_scores()
    scenarios = list(scenarios)
    weighted_attributes = np.array(weighted_attributes)

    # Plot heat map of normalised performance scores
    generate_perf_heatmap(A_norm[1:], scenarios[1:], config.ATTRIBUTES_LIST, show, compact, "heatmap.png")

    # Plot weights & scores
    plot_weights(config.ATTRIBUTES_LIST, config.WEIGHTS, show, compact)
    plot_scores(sorted_scenario_scores, show, compact)

    # Radar & tornado chart analysis
    plot_scenario_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, show, compact)

    logger.warning(f"✅ Analysis complete.")

def save_png(filename, compact):
    save_path = os.path.join(config.PNG_DIR, filename)
    pad = PADDING if compact else HEIGHT * 0.15
    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches = pad)


"""
Heat Map
"""

def generate_perf_heatmap(A, scenario_names, attribute_names, show, compact, filename="heatmap.png"):
    if A.shape != (len(scenario_names), len(attribute_names)):
        raise ValueError(f"Matrix shape ({A.shape}) does not match provided labels: "
                         f"({len(scenario_names)}x{len(attribute_names)}).")
    scale_x = 1.0 if compact else 1.3
    scale_y = 0.9 if compact else 1.2

    fig, ax = plt.subplots(figsize=(WIDTH * scale_x, HEIGHT * scale_y))
    cax = ax.imshow(A, cmap='coolwarm_r', vmin=-1, vmax=1, aspect='auto')

    plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names, compact)
    plot_heatmap_values(A, ax)

    fig.tight_layout(pad=PADDING)
    save_png(filename, compact)
    if show: plt.show()
    plt.close(fig)


def plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names, compact):
    if not compact:
        ax.set_title("Normalised Performance Values", fontsize=TITLE_SIZE, pad=TITLE_SIZE, weight='bold')
    short_names = [f"Scenario {i+1}" for i in range(len(scenario_names))]
    ax.set_xlabel("Attributes", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_xticks(np.arange(len(attribute_names)))
    ax.set_yticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(attribute_names, fontsize=TICK_SIZE)
    ax.set_yticklabels(short_names, fontsize=TICK_SIZE)
    plt.setp(ax.get_xticklabels(), rotation=ROTATION, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(attribute_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(scenario_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=LINE_WIDTH)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(cax, ax=ax, pad=PADDING, shrink=0.75)
    cbar.set_label('Scale (-1 to 1)', rotation=270, labelpad=LABEL_SIZE, fontsize=VALUE_SIZE)


def plot_heatmap_values(A, ax):
    m,n = A.shape
    for i in range(m):
        for j in range(n):
            value = A[i, j]
            text_label = f"{value:.3f}"
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, text_label, ha="center", va="center", color=text_color, fontsize=VALUE_SIZE)


"""
Scenario Charts
"""
def plot_scenario_charts(scenarios, A, attributes, show, compact):
    logger.debug("\n⏳ Sorting scenarios and generating visualizations")
    control_name = scenarios[0]
    control_scores = A[0]
    active_names = np.array(scenarios[1:]) # sort by total score descending
    active_scores = A[1:]
    row_sums = np.sum(active_scores, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]  # sort by total score descending

    sorted_names = active_names[sorted_indices].tolist()
    sorted_scores = active_scores[sorted_indices]

    m_active, n = sorted_scores.shape
    ylim = np.round(max(config.WEIGHTS), 3)
    plot_top_scores(sorted_scores, sorted_names, attributes, ylim, show, compact)
    norm = mcolors.Normalize(vmin=-ylim*1.1, vmax=ylim*1.1)

    for i in range(m_active):
        current_name = active_names[i]
        current_scores = active_scores[i].tolist()

        # Radar Chart
        plot_radar(n, control_name, control_scores, current_name, current_scores, attributes, ylim, show, compact)

        # Tornado Chart
        fig, ax_bar = plt.subplots(figsize=(HEIGHT * 0.6, HEIGHT * 0.75))
        plt.title(current_name, fontsize=LABEL_SIZE)
        bar_colors = [CMAP(norm(score)) for score in current_scores]
        y_pos_bar = np.arange(n)
        plot_tornado_labels(ax_bar, ylim, current_name, current_scores, bar_colors, y_pos_bar)
        text_padding = ylim * 0.05
        for idx in range(n):
            score = current_scores[idx]
            place_tornado_values(ax_bar, idx, attributes[idx], score, ylim, text_padding)
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)

        filename = f"{current_name.replace(':', '')} - Tornado.png"
        save_png(filename, compact)
        if show: plt.show()
        plt.close()

    logger.debug("\t✔️ Finished generating prioritized passports")

def plot_tornado_labels(ax_bar, ylim, current_name, current_scores, bar_colors, y_pos_bar):
    ax_bar.barh(y_pos_bar, current_scores, align='center', color=bar_colors, edgecolor=EDGE_COLOR, linewidth=0.25, alpha=0.9, height=BAR_HEIGHT * 0.6)
    ax_bar.axvline(0, color=AX_COLOR, linestyle='-', linewidth=LINE_WIDTH*2)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
    ax_bar.set_xticklabels([-ylim, "", 0, "", ylim])
    ax_bar.grid(axis='x', linestyle='--', alpha=GRID_ALPHA)
    ax_bar.set_xlim([-ylim, ylim * 1.4])


def place_tornado_values(ax_bar, idx, attr, score, ylim, text_padding):
    ax_bar.text(ylim * 1.02, idx, attr, ha='left', va='center',
                fontsize=VALUE_SIZE, fontweight='bold', color=VALUE_DARK)
    if score >= 0:
        if score < ylim * 0.75:
            # Positive bar: place score numbers just to the right of the bar tip
            ax_bar.text(score + text_padding, idx - text_padding, f"{score:+.3f}", ha='left', va='center',
                        fontsize=VALUE_SIZE, color=VALUE_DARK)
        else:
            ax_bar.text(score - text_padding, idx - text_padding, f"{score:+.3f}", ha='right', va='center',
                        fontsize=VALUE_SIZE, color=VALUE_LIGHT)
    else:
        if score > -ylim * 0.75:
            # Negative bar: place score numbers just to the left of the bar tip
            ax_bar.text(score - text_padding, idx - text_padding, f"{score:+.3f}", ha='right', va='center',
                        fontsize=VALUE_SIZE, color=VALUE_DARK)
        else:
            ax_bar.text(score + text_padding, idx - text_padding, f"{score:+.3f}", ha='left', va='center',
                        fontsize=VALUE_SIZE, color=VALUE_LIGHT)


"""
Top 3 Scenarios
"""
def plot_top_scores(sorted_scores, sorted_names, attributes, ylim, show, compact):
    m_test, n = sorted_scores.shape
    top_k = min(3, m_test)
    fig, ax = plt.subplots(figsize=(HEIGHT * 0.80, HEIGHT * 1.4))
    y_pos = np.arange(n) * 2.5

    plot_top_bars(ax, top_k, sorted_scores, sorted_names, y_pos)
    plot_top_labels(ax, y_pos, attributes, ylim, top_k)
    save_png("Top3_Scenarios.png", compact)
    if show: plt.show()
    plt.close()


def plot_top_bars(ax, top_k, sorted_scores, sorted_names, y_pos):
    ylim = 0.1
    bar_height = 0.65
    green_shades = [plt.cm.Greens(val) for val in np.linspace(0.9, 0.5, top_k)]
    for idx in range(top_k):
        offset = (idx - (top_k - 1) / 2) * bar_height
        bars = ax.barh(y_pos + offset, sorted_scores[idx], bar_height,
                color=green_shades[idx], label=f"#{idx+1} - {sorted_names[idx]}", alpha=0.9)
        draw_bar_values(ax, bars, ylim)


def plot_top_labels(ax, y_pos, attributes, ylim, top_k):
    ax.set_title(f"Top {top_k} Scenarios Attribute Scores", fontsize=TITLE_SIZE, pad=VALUE_SIZE)
    ax.axvline(0, color=AX_COLOR, linestyle='-', linewidth=LINE_WIDTH * 6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attributes, fontsize=TICK_SIZE)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_yaxis()
    ax.set_xlim([-ylim, ylim])
    ax.set_xlabel("Score", fontsize=LABEL_SIZE)
    ax.grid(axis='x', alpha=GRID_ALPHA)
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2))


def draw_bar_values(ax, bars, ylim):
    padding = ylim * PADDING
    for bar in bars:
        width = bar.get_width()
        y_coord = bar.get_y() + bar.get_height() / 2
        text_color = "black"
        font_weight = "normal"
        if width > 0.15:
            x_coord = width * 0.6
            alignment = 'left'
            text_color = "white"
        elif width >= 0:
            x_coord = width + padding
            alignment = 'left'
        else:
            x_coord = width - padding
            alignment = 'right'
        ax.text(x_coord, y_coord, f"{width:.3f}",
                va='center', ha=alignment,
                fontsize=VALUE_SIZE, fontweight=font_weight,
                color=text_color)


# RADAR

def plot_radar(n, control_name, control_scores, current_name, current_scores, attributes, ylim, show, compact):
    fig, ax_radar = plt.subplots(figsize=(HEIGHT * 1.1 ,HEIGHT * 1.1), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    control_radar_vals = control_scores.tolist() + [control_scores[0]]

    ax_radar.plot(angles, control_radar_vals, linewidth=LINE_WIDTH*8, label=f"{control_name}")
    ax_radar.fill(angles, control_radar_vals, alpha=0.25)
    active_radar_vals = current_scores + [current_scores[0]]
    ax_radar.plot(angles, active_radar_vals, linewidth=LINE_WIDTH*8, label=current_name)
    ax_radar.fill(angles, active_radar_vals, alpha=GRID_ALPHA/2)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(attributes, fontsize=TICK_SIZE)
    ax_radar.set_yticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
    ax_radar.set_yticklabels(["", "", "(Control)", "", ""])
    ax_radar.legend(loc='center', bbox_to_anchor=ANCHOR, fontsize=VALUE_SIZE, frameon=False)
    filename = f"{current_name.replace(':', '')} - Radar.png"
    save_png(filename, compact)
    plt.close()


"""
Weights & Scores
"""
def plot_scores(scenario_scores, show, compact):
    logger.debug("\n⏳ Generating scenario score bar chart...")
    try:
        scenarios, scores = zip(*scenario_scores)
        short_names = [name.split(':')[0] for name in scenarios]
        max_score = max(abs(s) for s in scores)
        max_weight = max(abs(w) for w in config.WEIGHTS)

        plot_score_labels(max_score)
        plot_score_bars(short_names, scores, max_score)
        save_png("Weighted Scenario Scores.png", compact)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


def plot_score_labels(max_score):
    plt.figure(figsize=(HEIGHT, HEIGHT * 0.7))
    plt.ylim(-max_score * 1.55, max_score * 1.55)
    plt.ylabel('Composite Score', fontweight='bold')
    plt.xticks(rotation=ROTATION, ha='right')



def plot_score_bars(short_names, scores, max_score):
    norm = mcolors.Normalize(vmin=-max_score * 1.5, vmax=max_score * 1.5)
    bar_colors = [CMAP(norm(score)) for score in scores]
    bars = plt.bar(short_names, scores, color=bar_colors, edgecolor=EDGE_COLOR, linewidth=LINE_WIDTH, alpha=BAR_ALPHA)
    plt.axhline(0, color=AX_COLOR, linestyle='-', linewidth=LINE_WIDTH)

    for bar in bars:
        yval = bar.get_height()
        va_dir = 'bottom' if yval >= 0 else 'top'
        offset = PADDING if yval >= 0 else PADDING * -2
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset,
                 f"{yval:.3f}", va=va_dir, ha='center')
    plt.tight_layout()


def plot_weights(attributes, weights, show, compact):
    logger.debug("\n⏳ Plotting weights distribution...")
    try:
        max_weight = (np.max(weights)) * 1.1
        plot_weight_labels(min(1, max_weight))
        plot_weight_bars(attributes, weights, max_weight)
        save_png("Weight Distribution.png", compact)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


def plot_weight_labels(ylim):
    plt.figure(figsize=(WIDTH * 0.8,HEIGHT * 0.65))
    plt.xlabel('Attribute', fontweight='bold')
    plt.ylabel('Weight', fontweight='bold')
    plt.ylim(0, 0.27)
    plt.xticks(rotation=ROTATION, ha='right')


def plot_weight_bars(attributes, weights, max_weight):
    norm_weights = mcolors.Normalize(vmin=-max_weight * 0.2, vmax=max_weight*1.2)
    bar_colors = [CMAP(norm_weights(w)) for w in weights]
    dark_edge = CMAP(0.9)

    bars = plt.bar(attributes, weights, color=bar_colors, edgecolor=dark_edge, linewidth=LINE_WIDTH, alpha=BAR_ALPHA)

    text_offset = max_weight * PADDING
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + text_offset,
                 f"{yval:.3f}", va='bottom', ha='center', fontsize=VALUE_SIZE)
    plt.tight_layout()


def init_analyze():
    global logger
    logger = get_logger()
