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
#     plot_radar_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, showradar)
    plot_comprehensive_charts(scenarios, weighted_attributes, config.ATTRIBUTES_LIST, False)

    logger.warning(f"✅ Analysis complete.")


"""
Tornado Charts
"""

def plot_comprehensive_charts(scenarios, A, attributes, show_plots=False):
    """
    Generates a suite of advanced performance bar charts and psychological radar charts.
    Assumes scenarios[0] is the Control baseline.
    """
    validate_attributes_matrix(A)
    logger.debug("\n⏳ Sorting scenarios and generating refined visualizations")

    # 1. Isolate Control baseline
    control_name = scenarios[0]
    control_scores = A[0]

    # 2. Extract and Sort Active Scenarios by their total score summation
    active_names = np.array(scenarios[1:])
    active_scores = A[1:]

    row_sums = np.sum(active_scores, axis=1)
    sorted_indices = np.argsort(row_sums)[::-1]  # Sort descending (Highest overall first)

    sorted_names = active_names[sorted_indices].tolist()
    sorted_scores = active_scores[sorted_indices]

    m_active, n = sorted_scores.shape
    ylim = np.round(max(config.WEIGHTS), 3)

    # Angles for polar axes
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # -------------------------------------------------------------------------
    # CHART 1: TOP 3 SCENARIOS OVERVIEW (Clustered Bar Chart - Sequential Greens)
    # -------------------------------------------------------------------------
    top_k = min(3, m_active)
    fig, ax = plt.subplots(figsize=(11, 8))

    y_pos = np.arange(n)
    bar_height = 0.25

    # Using a sequential ColorBrewer Greens palette
    # We sample from the darker end (0.5 to 0.9) so the text remains readable and distinct
    green_shades = [plt.cm.Greens(val) for val in np.linspace(0.9, 0.5, top_k)]

    for idx in range(top_k):
        offset = (idx - (top_k - 1) / 2) * bar_height
        ax.barh(y_pos + offset, sorted_scores[idx], bar_height,
                color=green_shades[idx], label=f"Rank {idx+1} ({sorted_names[idx]})", alpha=0.9)

    ax.axvline(0, color='#333333', linestyle='-', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attributes, fontsize=10)

    # Move labels to the right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.invert_yaxis()
    ax.set_xlim([-ylim, ylim])
    ax.set_xlabel("◄ Losses (Below Control)  │  Gains (Above Control) ►", fontsize=11, fontweight='bold')
    ax.set_title(f"Dominance Profile: Top {top_k} Scenarios Compared", fontsize=14, pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right') # Shifting legend out of right-axis labels

    plt.savefig(f"{PNG_DIR}/Top_Scenarios_Clustered_Overview.png", bbox_inches='tight')
    if show_plots: plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    # CHART 2: SCENARIO PERFORMANCE PASSPORTS (With Central Labels & RdYlBu)
    # -------------------------------------------------------------------------
    # Setup standard color map scaling between the dynamic bounds [-ylim, ylim]
    norm = mcolors.Normalize(vmin=-ylim, vmax=ylim)
    cmap_diverging = plt.cm.get_cmap('RdYlBu')

    for i in range(m_active):
        current_name = sorted_names[i]
        current_scores = sorted_scores[i].tolist()

        fig = plt.figure(figsize=(16, 6))
        grid = plt.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.2)

        # --- A. RADAR THUMBNAIL (Left Panel) ---
        ax_radar = fig.add_subplot(grid[0], projection='polar')

        # Tangible Solid Control Area
        control_radar_vals = control_scores.tolist() + [control_scores[0]]
        ax_radar.plot(angles, control_radar_vals, color='#555555', linewidth=2, label=f"{control_name} (Baseline)")
        ax_radar.fill(angles, control_radar_vals, color='#999999', alpha=0.25)

        # Active Profile Overlay
        active_radar_vals = current_scores + [current_scores[0]]
        ax_radar.plot(angles, active_radar_vals, linewidth=2.5, label=current_name)
        ax_radar.fill(angles, active_radar_vals, alpha=0.15)
#         ax_radar.plot(angles, active_radar_vals, color='#2b7bba', linewidth=2.5, label=current_name)
#         ax_radar.fill(angles, active_radar_vals, color='#2b7bba', alpha=0.15)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(attributes, fontsize=8)
        ax_radar.set_yticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
        ax_radar.set_yticklabels(["", "", "0 (Control)", "", ""])
        ax_radar.set_title("Visual Footprint (Gestalt Shape)", fontsize=11, pad=10, style='italic')
        ax_radar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9)

        # --- B. DIVERGING TORNADO CHART (Right Panel with Central Axis Labels) ---
        ax_bar = fig.add_subplot(grid[1])
        y_pos_bar = np.arange(n)

        # Assign colors where intensity directly mirrors magnitude via RdYlBu
        bar_colors = [cmap_diverging(norm(score)) for score in current_scores]
        ax_bar.barh(y_pos_bar, current_scores, align='center', color=bar_colors, edgecolor='#555555', linewidth=0.5, alpha=0.9, height=0.55)

        # Axis Framework
        ax_bar.axvline(0, color='#333333', linestyle='-', linewidth=1.5)
        ax_bar.set_yticks(y_pos_bar)
        ax_bar.set_yticklabels([]) # We drop standard y-ticks to draw them manually in the center
        ax_bar.invert_yaxis()
        ax_bar.set_xlim([-ylim, ylim])
        ax_bar.set_xticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
        ax_bar.set_xticklabels([-ylim, "", "Control Baseline", "", ylim])
        ax_bar.grid(axis='x', linestyle='--', alpha=0.4)

        # Smart Central Labels Injection
        # We push text slightly opposite to the bar's direction to prevent overlapping
        for idx in range(n):
            ax_bar.text(ylim, idx, attributes[idx], ha='left', va='center',
                        fontsize=9.5, fontweight='bold', color='#222222')

        ax_bar.set_title("Quantifiable Deviation (Accurate Magnitude)", fontsize=11, pad=10, style='italic')

        # Modern Border Clean
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)

        fig.suptitle(f"{current_name} Performance Passport", fontsize=14, fontweight='bold', y=1.02)

        png_file = f"Passport_{current_name.replace(' ', '_')}.png"
        plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
        if show_plots: plt.show()
        plt.close()

    logger.debug("\t✔️ Finished generating prioritized passports")



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
        save_png("weighted_scores", PNG_DIR)
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
    # Setup dynamic ColorBrewer RdYlBu scaling centered exactly at 0.0
    norm = mcolors.Normalize(vmin=-max_score * 1.5, vmax=max_score * 1.5)
    cmap_scores = plt.cm.get_cmap('RdYlBu')
    bar_colors = [cmap_scores(norm(score)) for score in scores]

    bars = plt.bar(scenarios, scores, color=bar_colors, edgecolor='#555555', linewidth=0.7, alpha=0.9)

    # Draw a clean baseline reference at 0
    plt.axhline(0, color='#333333', linestyle='-', linewidth=1.2)

    # Adjust text placement dynamically so text doesn't overlap the bars
    for bar in bars:
        yval = bar.get_height()
        va_dir = 'bottom' if yval >= 0 else 'top'
        # Tiny offset push so text sits cleanly just past the edge of the bar
        offset = 0.02 if yval >= 0 else -0.05

        plt.text(bar.get_x() + bar.get_width()/2, yval + offset,
                 f"{yval:.4f}", va=va_dir, ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()


def plot_weights(attributes, weights):
    logger.debug("\n⏳ Plotting weights distribution...")
    try:
        plot_weight_labels(min(1, np.max(weights)*1.2))
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
    # Using a uniform, high-visibility green derived straight from the standard ColorBrewer Greens colormap
    unified_green = plt.cm.Greens(0.6)
    dark_green_edge = plt.cm.Greens(0.9)
    green_shades = [plt.cm.Greens(val) for val in np.linspace(0.9, 0.5, len(attributes))]

    bars = plt.bar(attributes, weights, color=green_shades, edgecolor=dark_green_edge, linewidth=1.0, alpha=0.9)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (ylim*0.01 if 'ylim' in locals() else 0.005),
                 f"{yval:.4f}", va='bottom', ha='center', fontsize=9)
    plt.tight_layout() # Prevents label cutoff



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




'''
    """
    Generates horizontal diverging bar charts to compare scenarios against a baseline.
    Gains over control (0) extend right; losses extend left.
    """
    logger.debug("\n⏳ Generating horizontal bar charts")
    m, n = A.shape
    ylim = np.round(max(config.WEIGHTS), 3)

    # Overview by attribute with range bars
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(n)

    # Calculate min and max bounds across all scenarios for the background range
    min_bounds = np.min(A, axis=0)
    max_bounds = np.max(A, axis=0)

    # Draw a subtle background span showing the performance envelope
    ax.barh(y_pos, max_bounds - min_bounds, left=min_bounds,
            color='gray', alpha=0.15, height=0.5, label='Scenario Range Variance')

    # Plot each scenario as a distinct marker point to avoid bar-clutter
    colors = plt.cm.get_cmap('tab10', m)
    for i in range(m):
        # We skip plotting control explicitly if scenarios[0] IS the control,
        # but plotting all lines makes comparison seamless.
        ax.scatter(A[i], y_pos, color=colors(i), s=60, zorder=3, label=scenarios[i])

    # Formatting the overview chart
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7) # Baseline
    ax.set_yticks(y_pos)
    ax.set_yticklabels(attributes, fontsize=10)
    ax.invert_yaxis()  # Read top-to-bottom
    ax.set_xlim([-ylim, ylim])
    ax.set_xlabel("Losses vs. Gains relative to Control", fontsize=11)
    ax.set_title("Unified Scenario Attribute Space", fontsize=14, pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', bbox_to_anchor=(1., 0.))

    overview_title = "All_Scenarios_Bar_Overview"
    plt.savefig(f"{PNG_DIR}/{overview_title}.png", bbox_inches='tight')
    logging.debug(f"....Saved {overview_title}.png")
    if showbar: plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    # CHART 2: Scenario vs Control Profiles (Faceted/Individual Layout)
    # -------------------------------------------------------------------------
    # We use a shared layout so the "shape" is immediately comparable across saves
    for i in range(m):
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(n)
        scores = A[i]

        # Color bars dynamically: Green for gains, Red for losses
        bar_colors = ['#2ca02c' if score >= 0 else '#d62728' for score in scores]

        # Draw the horizontal bars
        bars = ax.barh(y_pos, scores, align='center', color=bar_colors, alpha=0.8, edgecolor='none', height=0.6)

        # Styling and Baseline setting
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(attributes, fontsize=10)
        ax.invert_yaxis()  # Top-to-bottom reading
        ax.set_xlim([-ylim, ylim])

        # Ticks alignment
        ax.set_xticks([-ylim, -ylim/2.0, 0.0, ylim/2.0, ylim])
        ax.set_xticklabels([-ylim, "", "0 (Control)", "", ylim])

        fig_title = f"{scenarios[i]} vs Control Performance Profile"
        ax.set_title(fig_title, fontsize=13, pad=15)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        # Clean up chart borders (spines) for a modern minimalist look
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        png_file = f"{fig_title.replace(' ', '_')}.png"
        plt.savefig(f"{PNG_DIR}/{png_file}", bbox_inches='tight')
        logging.debug(f"....Saved {png_file}")
        if showbar: plt.show()
        plt.close()

    logger.debug("\t✔️ Finished generating tornado charts")

'''

def init_analyze():
    global logger
    logger = get_logger()
