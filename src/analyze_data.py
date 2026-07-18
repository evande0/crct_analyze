import config
import csv
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from src.utils import get_logger



CMAP = mpl.colormaps[config.COLOR_MAP]
YLIM = np.round(max(config.WEIGHTS), 3)
MAX_WEIGHT = max(abs(w) for w in config.WEIGHTS)
logger = None


def analyze_data(processed_data, show, compact):
    scenario_names = processed_data["scenario_names"]
    A_norm = processed_data["attributes_norm"]
    A_wtd = processed_data["attributes_wtd"]
    config_sorted_scores = processed_data["sorted_results"] # Sort By is set in config.yaml
    set_plot_styles()

    plot_perf_heatmap(scenario_names[1:], A_norm[1:], show, compact) # Skip Control [0]
    plot_weights(show, compact)
    max_score = plot_scores(config_sorted_scores, show, compact)
    plot_scenario_charts(scenario_names, A_wtd, max_score, show, compact)
    plot_topk_scores(scenario_names, A_wtd, show, compact, k=3)
    logger.warning(f"✅ Analysis complete.")


def set_plot_styles():
    plt.rcParams['axes.titlesize'] = config.TITLE_SIZE
    plt.rcParams['axes.titlepad'] = config.TITLE_SIZE
    plt.rcParams['axes.labelsize'] = config.LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = config.TICK_SIZE
    plt.rcParams['ytick.labelsize'] = config.TICK_SIZE

    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    plt.rcParams['figure.figsize'] = [config.WIDTH, config.HEIGHT]
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['grid.alpha'] = config.GRID_ALPHA
    plt.rcParams['axes.edgecolor'] = config.EDGE_COLOR

"""
Heat Map
"""

def plot_perf_heatmap(scenario_names, A, show, compact):
    if A.shape != (len(scenario_names), len(config.ATTRIBUTES_LIST)):
        raise ValueError(f"Matrix shape ({A.shape}) does not match provided labels: "
                         f"({len(scenario_names)}x{len(attribute_names)}).")

    fig, ax = plt.subplots(figsize=(config.WIDTH, config.HEIGHT))

    cax = ax.imshow(A, cmap='coolwarm_r', vmin=-1, vmax=1, aspect='auto')

    plot_heatmap_labels(fig, ax, cax, scenario_names, config.ATTRIBUTES_LIST, compact)
    plot_heatmap_values(A, ax)
    plt_title = "Normalized Performance Data"
    if not compact: plt.title(plt_title)
    fig.tight_layout(pad=config.PADDING)
    save_png(f"Heatmap - {plt_title}", compact)
    if show: plt.show()
    plt.close(fig)


def plot_heatmap_labels(fig, ax, cax, scenario_names, attribute_names, compact):
    short_names = [f"Scenario {i+1}" for i in range(len(scenario_names))]
    ax.set_xlabel("Attributes")
    ax.set_xticks(np.arange(len(attribute_names)))
    ax.set_yticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(attribute_names)
    ax.set_yticklabels(short_names)
    plt.setp(ax.get_xticklabels(), rotation=config.ROTATION, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(attribute_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(scenario_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=config.LINE_WIDTH)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(cax, ax=ax, pad=config.PADDING, shrink=0.75)


def plot_heatmap_values(A, ax):
    m,n = A.shape
    for i in range(m):
        for j in range(n):
            value = A[i, j]
            text_label = f"{value:.3f}"
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, text_label, ha="center", va="center", color=text_color, fontsize=config.VALUE_SIZE)

"""
Weights & Scores
"""
def plot_scores(scenario_scores, show, compact):
    logger.debug("\n⏳ Generating scenario score bar chart...")
    try:
        scenario_names, scores = zip(*scenario_scores)
        short_names = [name.split(':')[0] for name in scenario_names]
        max_score = max(abs(s) for s in scores)
        plt.figure(figsize=(config.WIDTH * 0.8, config.HEIGHT))
        plot_score_labels(max_score)
        plot_score_bars(short_names, scores, max_score)
        plt_title = "Scenario Scores"
        if not compact: plt.title(plt_title)
        save_png(f"{plt_title}.png", compact)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)
    return max_score


def plot_score_labels(max_score):
    ylim = round(max_score * 1.55, 2)
    plt.ylim(-ylim, ylim)
    plt.ylabel('Composite Score')
    plt.xticks(rotation=config.ROTATION, ha='right')
    plt.yticks([-ylim, -ylim/2, 0, ylim/2, ylim])


def plot_score_bars(short_names, scores, max_score):
    norm = mcolors.Normalize(vmin=-max_score * 1.5, vmax=max_score * 1.5)
    bar_colors = [CMAP(norm(score)) for score in scores]
    bars = plt.bar(short_names, scores, color=bar_colors, edgecolor=config.EDGE_COLOR, linewidth=config.LINE_WIDTH, alpha=config.BAR_ALPHA)
    plt.axhline(0, color=config.AX_COLOR, linestyle='-', linewidth=config.LINE_WIDTH)

    for bar in bars:
        yval = bar.get_height()
        va_dir = 'bottom' if yval >= 0 else 'top'
        offset = config.PADDING if yval >= 0 else config.PADDING * -2
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset,
                 f"{yval:.3f}", va=va_dir, ha='center', fontsize=config.VALUE_SIZE)
    plt.tight_layout()


def plot_weights(show, compact):
    logger.debug("\n⏳ Plotting weights distribution...")
    try:
        ylim = round(min(1, MAX_WEIGHT), 1)
        plt.figure(figsize=(config.WIDTH * 0.8, config.HEIGHT * 0.7))
        plot_weight_labels(ylim)
        plot_weight_bars(config.WEIGHTS)
        plt_title  = "Weight Distribution"
        if not compact: plt.title(plt_title)
        save_png(f"{plt_title}.png", compact)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)


def plot_weight_labels(ylim):
    plt.xlabel('Attribute')
    plt.ylabel('Weight')
    plt.ylim(0, ylim + 0.05)
    plt.xticks(rotation=config.ROTATION, ha='right')
    plt.yticks([0, ylim/2, ylim])


def plot_weight_bars(weights):
    norm_weights = mcolors.Normalize(vmin=-MAX_WEIGHT * 0.2, vmax=MAX_WEIGHT*1.2)
    bar_colors = [CMAP(norm_weights(w)) for w in weights]
    dark_edge = CMAP(0.9)
    bars = plt.bar(config.ATTRIBUTES_LIST, weights, color=bar_colors, edgecolor=dark_edge, linewidth=config.LINE_WIDTH, alpha=config.BAR_ALPHA)
    text_offset = MAX_WEIGHT * config.PADDING
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + text_offset,
                 f"{yval:.3f}", va='bottom', ha='center', fontsize=config.VALUE_SIZE - 1)
    plt.tight_layout()


"""
Top 3 Scenarios
"""

def plot_topk_scores(scenario_names, A_wtd, show, compact, k=3):
    sorted_names, sorted_scores = sort_by_score_descending(scenario_names, A_wtd)
    m_test, n = sorted_scores.shape
    top_k = min(k, m_test)
    fig, ax = plt.subplots(figsize=(config.WIDTH * 0.75, config.HEIGHT * 2))
    bar_height = 2
    y_pos = np.arange(n) * bar_height * 4

    plot_top_bars(ax, top_k, sorted_scores, sorted_names, y_pos, bar_height)
    plot_top_labels(ax, y_pos, top_k)
    plt_title  = f"Top {k} Scenario Attribute Performance"
    save_png(f"{plt_title} - Tornado Chart.png", compact)
    if not compact: plt.title(plt_title)
    if show: plt.show()
    plt.close()


def sort_by_score_descending(scenario_names, A_wtd):
    names = np.array(scenario_names[1:])
    A = A_wtd[1:]
    row_sums = np.sum(A, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]  # sort by total score descending
    sorted_names = names[sorted_indices].tolist()
    sorted_scores = A[sorted_indices]
    return sorted_names, sorted_scores


def plot_top_bars(ax, top_k, sorted_scores, sorted_names, y_pos, bar_height):
    ylim = 0.1
    green_shades = [plt.cm.Greens(val) for val in np.linspace(0.9, 0.5, top_k)]
    for idx in range(top_k):
        offset = (idx - (top_k - 1) / 2) * bar_height
        bars = ax.barh(y_pos + offset, sorted_scores[idx], bar_height,
                color=green_shades[idx], label=f"#{idx+1} - {sorted_names[idx]}", alpha=0.9)
        draw_bar_values(ax, bars)


def plot_top_labels(ax, y_pos, top_k):
    ax.axvline(0, color=config.AX_COLOR, linestyle='-', linewidth=config.LINE_WIDTH * 6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(config.ATTRIBUTES_LIST, fontsize=config.TICK_SIZE)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_yaxis()
    ax.set_xlim([-YLIM, YLIM])
    ax.set_xlabel("Score", fontsize=config.LABEL_SIZE)
    ax.grid(axis='x', alpha=config.GRID_ALPHA)
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2))


def draw_bar_values(ax, bars):
    padding = YLIM * config.PADDING
    for bar in bars:
        width = bar.get_width()
        y_coord = (bar.get_y() + bar.get_height() / 2)  + 0.2
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
                fontsize=config.VALUE_SIZE , fontweight=font_weight,
                color=text_color)


"""
Scenario Charts
"""
def plot_scenario_charts(scenario_names, A_wtd, max_score, show, compact):
    control_name = scenario_names[0]
    control_scores = A_wtd[0]
    m_test, n = A_wtd.shape
    norm = mcolors.Normalize(vmin=-YLIM*1.1, vmax=YLIM*1.1)
    for i in range(1, m_test):
        current_name = scenario_names[i]
        current_scores = A_wtd[i].tolist()
        plot_radar(n, control_name, control_scores, current_name, current_scores, show, compact)
        plot_tornado(n, current_name, current_scores, show, compact)
    logger.debug("\t✔️ Finished generating prioritized passports")


# RADAR
def plot_radar(n, control_name, control_scores, current_name, current_scores, show, compact):
    fig, ax_radar = plt.subplots(figsize=(config.HEIGHT * 1.1 ,config.HEIGHT * 1.1), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    control_radar_vals = control_scores.tolist() + [control_scores[0]]
    ax_radar.plot(angles, control_radar_vals, linewidth=config.LINE_WIDTH*8, label=f"{control_name}")
    ax_radar.fill(angles, control_radar_vals, alpha=0.25)
    active_radar_vals = current_scores + [current_scores[0]]
    ax_radar.plot(angles, active_radar_vals, linewidth=config.LINE_WIDTH*8, label=current_name)
    ax_radar.fill(angles, active_radar_vals, alpha=config.GRID_ALPHA/2)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(config.ATTRIBUTES_LIST, fontsize=config.TICK_SIZE)
    ax_radar.set_yticks([-YLIM, -YLIM/2.0, 0.0, YLIM/2.0, YLIM])
    ax_radar.set_yticklabels(["", "", "(Control)", "", ""])
    ax_radar.legend(loc='center', bbox_to_anchor=(0.5, -0.3), fontsize=config.VALUE_SIZE, frameon=False)
    plt_title = f"{current_name.replace(':', '')}s"
    if not compact: plt.title(plt_title)
    save_png(f"{plt_title} - Radar.png", compact)
    if show: plt.show()
    plt.close()


# TORNADO
def plot_tornado(n, current_name, current_scores, show, compact):
    fig, ax_bar = plt.subplots(figsize=(config.WIDTH * 0.6, config.HEIGHT * 0.8))
    norm = mcolors.Normalize(vmin=-MAX_WEIGHT, vmax=MAX_WEIGHT)
    bar_colors = [CMAP(norm(score)) for score in current_scores]
    y_pos_bar = np.arange(n)
    bars = ax_bar.barh(y_pos_bar, current_scores, align='center', color=bar_colors, edgecolor=config.EDGE_COLOR, linewidth=0.25, alpha=0.9, height=config.BAR_HEIGHT)

    plot_tornado_labels(ax_bar, current_scores, bar_colors, y_pos_bar)
    for idx in range(n):
        score = current_scores[idx]
        place_tornado_values(ax_bar, idx, config.ATTRIBUTES_LIST[idx], score)
    for spine in ['top', 'right', 'left']:
        ax_bar.spines[spine].set_visible(False)
    plt_title = f"{current_name.replace(':', '')}"
    title_size = config.VALUE_SIZE if compact else config.TITLE_SIZE
    title_weight = 'normal' if compact else 'bold'
    plt.title(plt_title, fontsize=title_size, fontweight=title_weight)
    save_png(f"{plt_title} - Tornado.png", compact)
    if show: plt.show()
    plt.close()


def plot_tornado_labels(ax_bar, current_scores, bar_colors, y_pos_bar):

    ax_bar.axvline(0, color=config.AX_COLOR, linestyle='-', linewidth=config.LINE_WIDTH*2)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xticks([-YLIM, -YLIM/2.0, 0.0, YLIM/2.0, YLIM])
    ax_bar.set_xticklabels([-YLIM, "", 0, "", YLIM])
    ax_bar.grid(axis='x', linestyle='--', alpha=config.GRID_ALPHA)
    ax_bar.set_xlim([-YLIM * 1.4, YLIM * 1.4])


def place_tornado_values(ax_bar, idx, attr, score):
    x_padding = 0.01
    y_padding = 0.07
    ax_bar.text(YLIM * 1.02, idx, attr, ha='left', va='center', fontsize=config.VALUE_SIZE, fontweight='bold', color=config.VALUE_DARK)
    if score >= 0:
        if score < YLIM * 0.75:
            # Positive bar: place score numbers just to the right of the bar tip
            ax_bar.text(score + x_padding, idx + y_padding, f"{score:+.3f}", ha='left', va='center', fontsize=config.VALUE_SIZE-1, color=config.VALUE_DARK)
        else:
            ax_bar.text(score - x_padding, idx + y_padding, f"{score:+.3f}", ha='right', va='center', fontsize=config.VALUE_SIZE-1, color=config.VALUE_LIGHT)
    else:
        if score > -YLIM * 0.75:
            # Negative bar: place score numbers just to the left of the bar tip
            ax_bar.text(score - x_padding, idx + y_padding, f"{score:+.3f}", ha='right', va='center', fontsize=config.VALUE_SIZE-1, color=config.VALUE_DARK)
        else:
            ax_bar.text(score + x_padding, idx + y_padding, f"{score:+.3f}", ha='left', va='center', fontsize=config.VALUE_SIZE-1, color=config.VALUE_LIGHT)

def save_png(filename, compact):
    save_path = os.path.join(config.PNG_DIR, filename)
    pad = config.PADDING if compact else config.HEIGHT * 0.15
    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches = pad)


def init_analyze():
    global logger
    logger = get_logger()
