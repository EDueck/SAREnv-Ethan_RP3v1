# sarenv/utils/plot.py
"""
Geospatial visualization library for rendering Search and Rescue (SAR) environment arrays, 
evaluating continuous UAV trajectories, and generating comparative statistical figures.
"""
import os
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib

matplotlib.use('Agg')  # Enforces a non-interactive backend for headless scientific rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import stats
from shapely.geometry import Point

from sarenv.utils.logging_setup import get_logger
from sarenv.core.loading import SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.utils.geo import get_utm_epsg
from sarenv.utils.lost_person_behavior import get_environment_radius

log = get_logger()

# Standardized topological color mapping for environmental features
FEATURE_COLOR_MAP = {
    "structure": '#636363',  
    "road": '#bdbdbd',       
    "linear": '#8B4513',     
    "water": '#3182bd',      
    "drainage": '#9ecae1',   
    "woodland": '#31a354',   
    "scrub": '#78c679',      
    "brush": '#c2e699',      
    "field": '#fee08b',      
    "rock": '#969696',       
}
DEFAULT_COLOR = '#f0f0f0'

# Sequential palette for multi-agent trajectory rendering
COLORS_BLUE = [
    '#08519c',
    '#3182bd',
    '#6baed6',
    '#9ecae1',
    '#c6dbef',
]


def plot_heatmap(item, generated_paths, name, x_min, x_max, y_min, y_max, output_file):
    """
    Renders generated UAV trajectories overlaying a spatial probability distribution matrix.

    Args:
        item (SARDatasetItem): The encapsulated dataset containing the localized probability arrays.
        generated_paths (list): Evaluated geometric LineStrings representing agent paths.
        name (str): Title string for the generated figure.
        x_min, x_max, y_min, y_max (float): Cartesian boundary constraints for the visualization.
        output_file (str): Target serialization path for the rendered PDF.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    if item.heatmap is not None:
        ax.imshow(
            item.heatmap,
            extent=[x_min, x_max, y_min, y_max],
            cmap='YlOrRd',
            alpha=0.7,
            origin='lower'
        )

    colors = COLORS_BLUE
    line_width = 3.0
    for idx, path in enumerate(generated_paths):
        color = colors[idx % len(colors)]
        if path.geom_type == 'MultiLineString':
            for line in path.geoms:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=line_width, zorder=10)
        else:
            x, y = path.xy
            ax.plot(x, y, color=color, linewidth=line_width, zorder=10)

    # Strips Cartesian axis indicators to emphasize spatial data
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_file, format='pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Serialized trajectory overlay to {output_file}")


def plot_aggregate_bars(summary_df, evaluation_size, output_dir="graphs/aggregate"):
    """
    Generates aggregate bar charts to visualize metric distributions across all topographical datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = [
        ("Mean_Likelihood_Score", "CI_Likelihood_Score", "Likelihood Score"),
        ("Mean_Time_Discounted", "CI_Time_Discounted", "Time-Discounted Score"),
        ("Mean_Victims_Found", "CI_Victims_Found", "Victims Found (%)"),
        ("Mean_Triage_Score", "CI_Triage_Score", "Triage Score"),
        ("Mean_Area_Covered", "CI_Area_Covered", "Area Covered (km²)"),
        ("Mean_Path_Length", "CI_Path_Length", "Total Path Length (km)"),
    ]
    for mean_col, ci_col, label in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df["Algorithm"], summary_df[mean_col], yerr=summary_df[ci_col], capsize=5, alpha=0.7)
        plt.ylabel(label)
        plt.title(f"Algorithmic Performance: {label} (Aggregated Data)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f"aggregate_{label.replace(' ', '_').replace('(%)', '').replace('(km²)', '').replace('(km)', '').lower()}_{evaluation_size}.png"))
        plt.close()


def plot_combined_normalized_bars(summary_df, evaluation_size, output_dir="graphs/aggregate"):
    """
    Compiles a grouped bar chart comparing algorithmic performance across normalized statistical bounds (0.0 to 1.0).
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = [
        ("Mean_Likelihood_Score", "CI_Likelihood_Score", "Likelihood Score"),
        ("Mean_Time_Discounted", "CI_Time_Discounted", "Time-Discounted Score"),
        ("Mean_Victims_Found", "CI_Victims_Found", "Victims Found Score"),
        ("Mean_Triage_Score", "CI_Triage_Score", "Triage Score"),
    ]
    algorithms = summary_df["Algorithm"].tolist()
    n_algorithms = len(algorithms)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_algorithms

    normalized_means = {}
    for metric in metrics:
        values = summary_df[metric[0]].values
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val == 0:
            normalized_means[metric[0]] = np.ones_like(values)
        else:
            normalized_means[metric[0]] = (values - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10', n_algorithms)

    for i, alg in enumerate(algorithms):
        means = [normalized_means[metric[0]][summary_df["Algorithm"] == alg][0] for metric in metrics]
        cis = [summary_df.loc[summary_df["Algorithm"] == alg, metric[1]].values[0] for metric in metrics]
        positions = x - 0.4 + i * width + width / 2
        ax.bar(positions, means, width, yerr=cis, capsize=5, label=alg, color=colors(i), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in metrics])
    ax.set_ylabel('Normalized Score (0.0 to 1.0)')
    ax.set_title('Algorithmic Efficacy Across Normalized Evaluation Metrics')
    ax.legend(title='Algorithm')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregate_normalized_metrics_{evaluation_size}.png'))
    plt.close()


def plot_time_series_with_ci(time_series_results, evaluation_size, output_dir="graphs/plots"):
    """
    Renders longitudinal time-series metrics with 95% confidence intervals for individual algorithmic evaluations.
    """
    os.makedirs(output_dir, exist_ok=True)

    def mean_ci(arrays):
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), mode='edge') if len(a) < max_len else a for a in arrays]
        data = np.vstack(padded)
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)
        h = sem * stats.t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
        return mean, mean - h, mean + h

    for name, results_list in time_series_results.items():
        if not results_list:
            continue
        combined_likelihoods = [r['combined_cumulative_likelihood'] for r in results_list]
        combined_victims = [r['combined_cumulative_victims'] for r in results_list]

        mean_likelihood, ci_low_likelihood, ci_high_likelihood = mean_ci(combined_likelihoods)
        mean_victims, ci_low_victims, ci_high_victims = mean_ci(combined_victims)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_likelihood = 'tab:blue'
        ax1.set_xlabel('Evaluation Timestep')
        ax1.set_ylabel('Mean Spatial Likelihood Acquisition (%)', color=color_likelihood)
        ax1.plot(100 * mean_likelihood, color=color_likelihood, label='Mean Likelihood')
        ax1.fill_between(
            range(len(mean_likelihood)),
            100 * ci_low_likelihood, 100 * ci_high_likelihood,
            color=color_likelihood, alpha=0.3
        )
        ax1.tick_params(axis='y', labelcolor=color_likelihood)

        ax2 = ax1.twinx()
        color_victims = 'tab:red'
        ax2.set_ylabel('Mean Casualties Discovered', color=color_victims)
        ax2.plot(mean_victims, color=color_victims, label='Mean Casualties')
        ax2.fill_between(range(len(mean_victims)), ci_low_victims, ci_high_victims, color=color_victims, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color_victims)

        plt.title(f'Longitudinal Metric Evaluation (95% CI): {name}')
        fig.tight_layout()
        filename = os.path.join(output_dir, f'{name}_{evaluation_size}_average_combined_metrics.pdf')
        plt.savefig(filename)
        plt.close()


def plot_combined_time_series_with_ci(time_series_results, evaluation_size, output_dir="graphs/plots"):
    """
    Aggregates longitudinal time-series metrics across all evaluated algorithms into a vertically stacked comparative figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    def mean_ci(arrays):
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), mode='edge') if len(a) < max_len else a for a in arrays]
        data = np.vstack(padded)
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)
        h = sem * stats.t.ppf((1 + 0.95) / 2., data.shape[0] - 1)
        return mean, mean - h, mean + h

    valid_algorithms = {name: results for name, results in time_series_results.items() if results}

    if not valid_algorithms:
        log.warning("Time-series tracking absent. Comparative compilation aborted.")
        return

    max_time_length = 0
    processed_data = {}

    for name, results_list in valid_algorithms.items():
        combined_likelihoods = [r['combined_cumulative_likelihood'] for r in results_list]
        combined_victims = [r['combined_cumulative_victims'] for r in results_list]

        mean_likelihood, ci_low_likelihood, ci_high_likelihood = mean_ci(combined_likelihoods)
        mean_victims, ci_low_victims, ci_high_victims = mean_ci(combined_victims)

        processed_data[name] = {
            'mean_likelihood': mean_likelihood,
            'ci_low_likelihood': ci_low_likelihood,
            'ci_high_likelihood': ci_high_likelihood,
            'mean_victims': mean_victims,
            'ci_low_victims': ci_low_victims,
            'ci_high_victims': ci_high_victims
        }
        max_time_length = max(max_time_length, len(mean_likelihood))

    n_algorithms = len(valid_algorithms)
    fig, axes = plt.subplots(n_algorithms, 1, figsize=(12, 4 * n_algorithms), sharex=True)

    if n_algorithms == 1:
        axes = [axes]

    x_axis = np.arange(max_time_length)

    for i, (name, data) in enumerate(processed_data.items()):
        ax1 = axes[i]
        current_length = len(data['mean_likelihood'])
        
        if current_length < max_time_length:
            pad_width = max_time_length - current_length
            for key in data:
                data[key] = np.pad(data[key], (0, pad_width), mode='edge')

        color_likelihood = 'tab:blue'
        ax1.set_ylabel('Acquired Likelihood (%)', color=color_likelihood)
        ax1.plot(x_axis, 100 * data['mean_likelihood'], color=color_likelihood,
                 label='Mean Likelihood', linewidth=2)
        ax1.fill_between(
            x_axis,
            100 * data['ci_low_likelihood'],
            100 * data['ci_high_likelihood'],
            color=color_likelihood, alpha=0.3
        )
        ax1.tick_params(axis='y', labelcolor=color_likelihood)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color_victims = 'tab:red'
        ax2.set_ylabel('Casualties Discovered', color=color_victims)
        ax2.plot(x_axis, data['mean_victims'], color=color_victims,
                 label='Mean Casualties', linewidth=2)
        ax2.fill_between(x_axis, data['ci_low_victims'], data['ci_high_victims'],
                         color=color_victims, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color_victims)

        ax1.set_title(f'{name} - Longitudinal Progress (95% CI)', fontsize=14, fontweight='bold')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    axes[-1].set_xlabel('Evaluation Timestep', fontsize=12)
    fig.suptitle(f'Cross-Algorithmic Longitudinal Analysis ({evaluation_size})',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    filename = os.path.join(output_dir, f'combined_time_series_{evaluation_size}.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    log.info(f"Serialized aggregate time-series plot to {filename}")
    plt.close()


def plot_single_evaluation_results(results_df, evaluation_sizes, output_dir="graphs"):
    """
    Generates comparative bar charts for distinct evaluation metrics across spatial constraints.
    """
    if results_df is None:
        log.error("Results matrix empty. Evaluation sequence must precede rendering.")
        return

    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = [
        "Likelihood Score",
        "Time-Discounted Score",
        "Victims Found (%)",
        "Triage Score",
        "Area Covered (km²)",
        "Total Path Length (km)",
    ]

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=results_df,
            x="Dataset",
            y=metric,
            hue="Algorithm",
            order=evaluation_sizes,
        )
        plt.title(f"Comparative Efficacy: {metric}", fontsize=16)
        plt.ylabel(metric)
        plt.xlabel("Topographical Scope")
        plt.legend(title="Algorithm")
        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir, f"plot_{metric.replace(' ', '_').replace('(%)', '').replace('(m)', '').lower()}.png"
        )
        plt.savefig(plot_filename)
        log.info(f"Serialized evaluation metric to {plot_filename}")
        plt.close()


def create_individual_metric_plots(df_or_files, environment_size, output_dir="plots", budget_labels=None):
    """
    Generates isolated metric visualizations categorized by hardware budget constraints.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")  

    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
    except Exception:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

    def calculate_ci(data, confidence=0.95):
        n = len(data)
        if n < 2:
            return 0
        sem = stats.sem(data)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        return t_val * sem

    if isinstance(df_or_files, pd.DataFrame):
        combined_df = df_or_files
    else:
        if isinstance(df_or_files, list | tuple):
            file_paths = df_or_files
        else:
            file_paths = [df_or_files]

        dataframes = []
        for i, file_path in enumerate(file_paths):
            try:
                data_df = pd.read_csv(file_path)

                if budget_labels and i < len(budget_labels):
                    data_df['Budget Condition'] = budget_labels[i]
                elif len(file_paths) > 1:
                    data_df['Budget Condition'] = f'Condition {i + 1}'
                else:
                    data_df['Budget Condition'] = 'Default'

                dataframes.append(data_df)
            except FileNotFoundError:
                log.warning(f"Data file inaccessible: {file_path}")
                continue
            except Exception as e:
                log.error(f"Execution failure loading {file_path}: {e}")
                continue

        if not dataframes:
            log.error("Metric compilation aborted: Valid data arrays absent.")
            return

        combined_df = pd.concat(dataframes, ignore_index=True)

    size_data = combined_df[combined_df['Environment Size'] == environment_size].copy()

    if size_data.empty:
        log.warning(f"Topographical data absent for scope: {environment_size}")
        return

    size_data['Algorithm'] = size_data['Algorithm'].replace('RandomWalk', 'Random')

    metrics = {
        'Likelihood Score': {'column': 'Likelihood Score', 'unit': r'$\mathcal{L}(\pi)$'},
        'Time-Discounted Score': {'column': 'Time-Discounted Score', 'unit': r'$\mathcal{I}(\pi)$'},
        'Victims Found (%)': {'column': 'Victims Found (%)', 'unit': r'$\mathcal{D}(\pi)$'},
        'Triage Score': {'column': 'Triage Score', 'unit': r'$\mathcal{T}(\pi)$'}
    }

    algorithms = sorted([alg for alg in size_data['Algorithm'].unique() if alg != 'Spiral'])
    budgets = sorted(size_data['Budget Condition'].unique())

    for metric_name, metric_info in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 8))

        n_algorithms = len(algorithms)
        n_budgets = len(budgets)

        bar_width = 0.35
        x_positions = np.arange(n_algorithms)
        colors = sns.color_palette("Set2", n_budgets)

        for i, budget in enumerate(budgets):
            means = []
            cis = []

            for algorithm in algorithms:
                alg_budget_data = size_data[
                    (size_data['Algorithm'] == algorithm) &
                    (size_data['Budget Condition'] == budget)
                ]
                values = alg_budget_data[metric_info['column']].to_numpy()

                if len(values) > 0:
                    mean_val = np.mean(values)
                    ci_val = calculate_ci(values)
                else:
                    mean_val = 0
                    ci_val = 0

                means.append(mean_val)
                cis.append(ci_val)

            bar_positions = x_positions + i * bar_width
            bars = ax.bar(bar_positions, means, bar_width,
                          label=f'{budget}', color=colors[i], alpha=0.8,
                          yerr=cis, capsize=5, error_kw={'linewidth': 1.5})

            for bar, mean, ci in zip(bars, means, cis, strict=True):
                if mean > 0:
                    height = bar.get_height()
                    text_y = height + ci
                    ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=17)

        ax.set_ylabel(metric_info['unit'], fontsize=36, fontweight='bold')
        ax.set_xticks(x_positions + bar_width / 2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=26)
        ax.tick_params(axis='y', labelsize=26)
        ax.grid(True, alpha=0.3, axis='y')

        legend = ax.legend(fontsize=20, title="Operational Budget", frameon=True, fancybox=True, edgecolor='black')
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(20)

        current_ylim = ax.get_ylim()
        all_means = []
        all_cis = []
        
        for budget in budgets:
            for algorithm in algorithms:
                if algorithm == 'Spiral':
                    continue
                alg_budget_data = size_data[
                    (size_data['Algorithm'] == algorithm) &
                    (size_data['Budget Condition'] == budget)
                ]

                values = alg_budget_data[metric_info['column']].to_numpy()
                if len(values) > 0:
                    all_means.append(np.mean(values))
                    all_cis.append(calculate_ci(values))

        if all_means:
            max_value = max(all_means)
            max_ci = max(all_cis)
            new_ylim_top = (max_value + max_ci) * 1.15
            ax.set_ylim(current_ylim[0], new_ylim_top)

        plt.tight_layout()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        clean_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')
        pdf_filename = f"{clean_metric_name}_{environment_size}_grouped.pdf"
        pdf_filepath = output_path / pdf_filename

        plt.savefig(pdf_filepath, bbox_inches='tight', dpi=300)
        log.info(f"Serialized isolated performance array to: {pdf_filepath}")
        plt.close()


def setup_algorithm_plot(ax, item, victims_gdf, crs, algorithm_name, algorithm_colors):
    """Initializes the foundational Cartesian plane for algorithmic trajectory animations."""
    try:
        x_min, y_min, x_max, y_max = item.bounds
        extent = [x_min, x_max, y_min, y_max]

        ax.imshow(
            item.heatmap,
            extent=extent,
            origin='lower',
            cmap='YlOrRd',
            alpha=0.8,
            aspect='equal'
        )

        if not victims_gdf.empty:
            victims_proj = victims_gdf.to_crs(crs)
            ax.scatter(
                victims_proj.geometry.x,
                victims_proj.geometry.y,
                c='black', s=30, marker='X', linewidths=1,
                zorder=10, edgecolors='darkred'
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        title_color = algorithm_colors.get(algorithm_name, 'black')
        ax.set_title(f'{algorithm_name}', fontsize=12, fontweight='bold', color=title_color)
        ax.grid(True, alpha=0.3)

        ax.set_xticks([])
        ax.set_yticks([])

    except Exception as e:
        log.warning(f"Failed to initialize Cartesian plane for {algorithm_name}: {e}")
        ax.set_title(f'{algorithm_name} (Initialization Error)', fontsize=12)


def plot_drone_paths(ax, animation_data, frame_idx, drone_colors):
    """Renders the continuous multi-agent flight trajectories up to the current evaluation frame."""
    try:
        num_drones = animation_data.get('num_drones', 1)
        path_coordinates = animation_data.get('path_coordinates', [])

        if path_coordinates and frame_idx < len(path_coordinates):
            current_frame_coords = path_coordinates[frame_idx]

            for drone_idx in range(min(num_drones, len(current_frame_coords))):
                drone_path_coords = current_frame_coords[drone_idx]

                if len(drone_path_coords) > 1:
                    drone_path_x = [coord[0] for coord in drone_path_coords]
                    drone_path_y = [coord[1] for coord in drone_path_coords]

                    drone_color = drone_colors[drone_idx % len(drone_colors)]
                    ax.plot(drone_path_x, drone_path_y, color=drone_color,
                            linewidth=2, alpha=0.8, zorder=5)
        else:
            for drone_idx in range(num_drones):
                drone_path_x = []
                drone_path_y = []

                for past_frame in range(min(frame_idx + 1, len(animation_data['drone_positions']))):
                    if drone_idx < len(animation_data['drone_positions'][past_frame]):
                        pos = animation_data['drone_positions'][past_frame][drone_idx]
                        if pos is not None and len(pos) >= 2:
                            drone_path_x.append(pos[0])
                            drone_path_y.append(pos[1])

                if len(drone_path_x) > 1:
                    drone_color = drone_colors[drone_idx % len(drone_colors)]
                    ax.plot(drone_path_x, drone_path_y, color=drone_color,
                            linewidth=2, alpha=0.8, zorder=5)
    except Exception as e:
        log.warning(f"Mathematical execution failure resolving agent geometries: {e}")


def plot_current_drone_positions(ax, current_drone_positions, drone_colors, detection_radius):
    """Overlays the active UAV spatial coordinates and their corresponding optical sensor footprints."""
    try:
        for drone_idx, drone_position in enumerate(current_drone_positions):
            if drone_position is not None and len(drone_position) >= 2:
                drone_color = drone_colors[drone_idx % len(drone_colors)]

                detection_circle = plt.Circle(
                    (drone_position[0], drone_position[1]),
                    detection_radius,
                    fill=False, color=drone_color, alpha=0.3,
                    linewidth=1, linestyle='--', zorder=8
                )
                ax.add_patch(detection_circle)

                ax.scatter(
                    drone_position[0], drone_position[1],
                    c=drone_color, s=100, marker='o',
                    edgecolors='white', linewidths=2, zorder=15
                )

    except Exception as e:
        log.warning(f"Mathematical execution failure rendering agent coordinates: {e}")


def create_time_series_graphs(frame_idx, all_animation_data, ax_area, ax_score, ax_victims, ax_triage, algorithm_colors,
                              interval_distance_km=2.5):
    """Constructs dynamically updating longitudinal performance graphs for the animation sequence."""
    try:
        for ax in [ax_area, ax_score, ax_victims, ax_triage]:
            ax.clear()

        max_distance = 0

        for alg_name, animation_data in all_animation_data.items():
            if frame_idx < len(animation_data['metrics']):
                color = algorithm_colors.get(alg_name, 'black')
                frames_so_far = min(frame_idx + 1, len(animation_data['metrics']))

                scores = [animation_data['metrics'][i]['likelihood_score'] for i in range(frames_so_far)]
                victims = [animation_data['metrics'][i]['victims_found_pct'] for i in range(frames_so_far)]
                areas = [animation_data['metrics'][i]['area_covered'] for i in range(frames_so_far)]
                triage = [animation_data['metrics'][i].get('triage_score', 0) for i in range(frames_so_far)]

                if 'interval_distances' in animation_data and len(
                        animation_data['interval_distances']) >= frames_so_far:
                    distances = animation_data['interval_distances'][:frames_so_far]
                else:
                    distances = [i * interval_distance_km for i in range(frames_so_far)]

                if distances:
                    max_distance = max(max_distance, max(distances))

                try:
                    ax_area.plot(distances, areas, color=color, linewidth=2, label=alg_name)
                    ax_score.plot(distances, scores, color=color, linewidth=2, label=alg_name)
                    ax_victims.plot(distances, victims, color=color, linewidth=2, label=alg_name)
                    ax_triage.plot(distances, triage, color=color, linewidth=2, label=alg_name)
                except Exception as e:
                    log.warning(f"Evaluation render failed for {alg_name}: {e}")

        max_distance = max(max_distance, 1)

        for ax, title, ylabel in [
            (ax_area, 'Area Covered', 'Area (km²)'),
            (ax_score, 'Likelihood Score', 'Score'),
            (ax_victims, 'Victims Found', 'Percentage (%)'),
            (ax_triage, 'Triage Score', 'Score')
        ]:
            try:
                ax.set_xlim(0, max_distance * 1.1)
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(True, alpha=0.3)

                if ax == ax_area:
                    ax.legend(fontsize=8, loc='best')

                if ax != ax_triage:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Distance Traversed (km)', fontsize=9)

                if ax == ax_victims:
                    ax.set_ylim(0, 100)
            except Exception as e:
                log.warning(f"Error configuring Cartesian axis {title}: {e}")

    except Exception as e:
        log.warning(f"Execution failure resolving longitudinal tracking graphs: {e}")


def visualize_heatmap_matplotlib(item: SARDatasetItem, data_crs: str, plot_basemap: bool = True):
    """Generates a foundational spatial probability density visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))
    minx, miny, maxx, maxy = item.bounds
    im = ax.imshow(item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probability Density")
    
    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        
    ax.set_title(f"Spatial Matrix Evaluation: Constraint '{item.size}'")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")
    plt.tight_layout()


def visualize_risk_map(item: SARDatasetItem, plot_basemap: bool = True, plot_inset: bool = True, plot_show=True):
    """
    Renders the topographical hazard matrix (Beta Array) incorporating a localized spatial magnification inset.
    """
    if not hasattr(item, 'risk_map') or item.risk_map is None:
        log.warning("Topographical hazard array uninitialized. Visualization bypassed.")
        return

    log.info(f"Rendering topographical hazard matrix for constraint: {item.size}...")

    fig, ax = plt.subplots(figsize=(15, 13))

    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    minx, miny, maxx, maxy = item.bounds

    im = ax.imshow(
        item.risk_map, extent=(minx, maxx, miny, maxy), origin="lower", cmap="Reds"
    )

    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5, zorder=0)

    radii = get_environment_radius(item.environment_type, item.environment_climate)
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    legend_handles = []
    colors = ["blue", "orange", "red", "green"]
    labels = ["Small", "Medium", "Large", "Extra Large"]

    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=2, alpha=1, zorder=2)
        label = f"{labels[idx]} ({r} km)"
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
        )
    ax.legend(handles=legend_handles, title="RoIs", loc="upper left", fontsize=16, title_fontsize=18)

    if plot_inset:
        medium_radius_idx = 1
        medium_radius_m = radii[medium_radius_idx] * 1000

        inset_width = 0.60
        inset_height = 0.60
        ax_inset = ax.inset_axes([1.05, 0.5 - (inset_height / 2), inset_width, inset_height])

        ax_inset.imshow(item.risk_map, extent=(minx, maxx, miny, maxy), origin="lower", cmap="Reds")
        if plot_basemap:
            cx.add_basemap(ax_inset, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5)

        for idx, r in enumerate(radii):
            circle_geom = center_point_proj.buffer(r * 1000).iloc[0]
            gpd.GeoSeries([circle_geom], crs=data_crs).boundary.plot(
                ax=ax_inset, edgecolor=colors[idx % len(colors)], linestyle="--", linewidth=2, alpha=1)

        center_x = center_point_proj.geometry.x.iloc[0]
        center_y = center_point_proj.geometry.y.iloc[0]
        ax_inset.set_xlim(center_x - medium_radius_m, center_x + medium_radius_m)
        ax_inset.set_ylim(center_y - medium_radius_m, center_y + medium_radius_m)

        # Enforces a circular clipping path on the spatial inset
        circle_patch = Circle((0.5, 0.5), 0.5, transform=ax_inset.transAxes, facecolor='none', edgecolor='black',
                              linewidth=1)
        ax_inset.set_clip_path(circle_patch)
        ax_inset.patch.set_alpha(0.0)

        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        pp1, c1, c2 = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp1.set_zorder(0)
        c1.set_zorder(0)
        c2.set_zorder(0)

        pp2, c3, c4 = mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp2.set_zorder(0)
        c3.set_zorder(0)
        c4.set_zorder(0)

        cbar_ax = fig.add_axes([0.88, 0.25, 0.03, 0.5])
        fig.colorbar(im, cax=cbar_ax, label="Topographical Beta (Hazard Array)")
    else:
        fig.colorbar(im, ax=ax, shrink=0.8, label="Topographical Beta (Hazard Array)")

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels([f"{x / 1000:.1f}" for x in x_ticks], fontsize=18)
    ax.set_yticklabels([f"{y / 1000:.1f}" for y in y_ticks], fontsize=18)
    ax.set_xlabel("Easting (km)", fontsize=18)
    ax.set_ylabel("Northing (km)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"riskmap_{item.size}_magnified.pdf", bbox_inches='tight')
    
    if plot_show:
        plt.show()


def visualize_heatmap(item: SARDatasetItem, plot_basemap: bool = True, plot_inset: bool = True, plot_show=True):
    """
    Renders the spatial probability density matrix incorporating a localized spatial magnification inset.
    """
    log.info(f"Rendering spatial probability distribution for constraint: {item.size}...")

    fig, ax = plt.subplots(figsize=(15, 13))

    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    minx, miny, maxx, maxy = item.bounds
    
    im = ax.imshow(
        item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="YlOrRd"
    )

    if plot_basemap:
        cx.add_basemap(ax, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7, zorder=0)

    radii = get_environment_radius(item.environment_type, item.environment_climate)
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    legend_handles = []
    colors = ["blue", "orange", "red", "green"]
    labels = ["Small", "Medium", "Large", "Extra Large"]
    
    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=2, alpha=1, zorder=2)
        label = f"{labels[idx]} ({r} km)"
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
        )
    ax.legend(handles=legend_handles, title="RoIs", loc="upper left", fontsize=16, title_fontsize=18)

    if plot_inset:
        medium_radius_idx = 1
        medium_radius_m = radii[medium_radius_idx] * 1000

        inset_width = 0.60
        inset_height = 0.60
        ax_inset = ax.inset_axes([1.05, 0.5 - (inset_height / 2), inset_width, inset_height])

        ax_inset.imshow(item.heatmap, extent=(minx, maxx, miny, maxy), origin="lower", cmap="YlOrRd")
        if plot_basemap:
            cx.add_basemap(ax_inset, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.7)

        for idx, r in enumerate(radii):
            circle_geom = center_point_proj.buffer(r * 1000).iloc[0]
            gpd.GeoSeries([circle_geom], crs=data_crs).boundary.plot(
                ax=ax_inset, edgecolor=colors[idx % len(colors)], linestyle="--", linewidth=2, alpha=1)

        center_x = center_point_proj.geometry.x.iloc[0]
        center_y = center_point_proj.geometry.y.iloc[0]
        ax_inset.set_xlim(center_x - medium_radius_m, center_x + medium_radius_m)
        ax_inset.set_ylim(center_y - medium_radius_m, center_y + medium_radius_m)

        circle_patch = Circle((0.5, 0.5), 0.5, transform=ax_inset.transAxes, facecolor='none', edgecolor='black',
                              linewidth=1)
        ax_inset.set_clip_path(circle_patch)
        ax_inset.patch.set_alpha(0.0)

        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        pp1, c1, c2 = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp1.set_zorder(0)
        c1.set_zorder(0)
        c2.set_zorder(0)

        pp2, c3, c4 = mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp2.set_zorder(0)
        c3.set_zorder(0)
        c4.set_zorder(0)

        cbar_ax = fig.add_axes([0.88, 0.25, 0.03, 0.5])
        fig.colorbar(im, cax=cbar_ax, label="Spatial Probability Mass")
    else:
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spatial Probability Mass")

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels([f"{x / 1000:.1f}" for x in x_ticks], fontsize=18)
    ax.set_yticklabels([f"{y / 1000:.1f}" for y in y_ticks], fontsize=18)
    ax.set_xlabel("Easting (km)", fontsize=18)
    ax.set_ylabel("Northing (km)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"heatmap_{item.size}_magnified.pdf", bbox_inches='tight')
    
    if plot_show:
        plt.show()


def visualize_features(item: SARDatasetItem, plot_basemap: bool = False, plot_inset: bool = False,
                       num_lost_persons: int = 0, plot_show=True):
    """
    Renders discrete topographical features utilizing a localized spatial magnification inset, 
    mapping generated casualty distributions if active.
    """
    if not item:
        log.warning("Spatial data absent. Rendering sequence bypassed.")
        return

    radii = get_environment_radius(item.environment_type, item.environment_climate)

    log.info(f"Executing topographical feature serialization for constraint '{item.size}'...")
    center_point_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    )
    data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
    center_point_proj = center_point_gdf.to_crs(crs=data_crs)
    fig, ax = plt.subplots(figsize=(18, 15))

    feature_legend_handles = []
    
    for feature_type, data in item.features.groupby("feature_type"):
        color = FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR)
        data.plot(ax=ax, color=color, label=feature_type.capitalize(), alpha=0.7, zorder=1)
        feature_legend_handles.append(Patch(color=color, label=feature_type.capitalize()))

    if plot_basemap:
        cx.add_basemap(ax, crs=item.features.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

    radii_legend_handles = []
    lost_person_gdf = None
    
    if num_lost_persons > 0:
        log.info(f"Mapping stochastic distribution for {num_lost_persons} simulated casualties...")
        lost_person_generator = LostPersonLocationGenerator(item)
        locations = lost_person_generator.generate_locations(num_lost_persons, 0)

        if not locations.empty:
            lost_person_gdf = locations
            lost_person_gdf.plot(ax=ax, marker='*', color='red', markersize=200, zorder=1, label="Lost Person")
            radii_legend_handles.append(
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Casualty Topology'))

    colors = ["blue", "orange", "red", "green"]
    labels = ["Small", "Medium", "Large", "Extra Large"]
    
    for idx, r in enumerate(radii):
        circle = center_point_proj.buffer(r * 1000).iloc[0]
        color = colors[idx % len(colors)]
        gpd.GeoSeries([circle], crs=data_crs).boundary.plot(
            ax=ax, edgecolor=color, linestyle="--", linewidth=2, alpha=1, zorder=2)
        label = f"{labels[idx]} ({r} km)"
        radii_legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5, linestyle="--", label=label)
        )

    features_legend = ax.legend(handles=feature_legend_handles, title="Topographical Features",
                                loc="upper left", fontsize=16, title_fontsize=18)

    radii_legend = ax.legend(handles=radii_legend_handles, title="RoIs",
                             loc="upper right", fontsize=16, title_fontsize=18)

    ax.add_artist(features_legend)

    if plot_inset:
        medium_radius_idx = 1
        medium_radius_m = radii[medium_radius_idx] * 1000

        inset_width = 0.60
        inset_height = 0.60
        ax_inset = ax.inset_axes([1.05, 0.5 - (inset_height / 2), inset_width, inset_height])

        center_x = center_point_proj.geometry.x.iloc[0]
        center_y = center_point_proj.geometry.y.iloc[0]
        clipping_circle = Point(center_x, center_y).buffer(medium_radius_m)
        clipping_gdf = gpd.GeoDataFrame([1], geometry=[clipping_circle], crs=data_crs)

        features_proj = item.features.to_crs(data_crs)
        clipped_features = gpd.clip(features_proj, clipping_gdf)

        if plot_basemap:
            cx.add_basemap(ax_inset, crs=data_crs, source=cx.providers.OpenStreetMap.Mapnik)

        if not clipped_features.empty:
            for feature_type, data in clipped_features.groupby("feature_type"):
                data.plot(ax=ax_inset, color=FEATURE_COLOR_MAP.get(feature_type, DEFAULT_COLOR), alpha=0.7)

        for idx, r in enumerate(radii):
            circle_geom = center_point_proj.buffer(r * 1000).iloc[0]
            gpd.GeoSeries([circle_geom], crs=data_crs).boundary.plot(
                ax=ax_inset, edgecolor=colors[idx % len(colors)], linestyle="--", linewidth=2, alpha=1
            )

        if num_lost_persons > 0 and lost_person_gdf is not None:
            lost_person_proj = lost_person_gdf.to_crs(data_crs)
            clipped_lost_persons = gpd.clip(lost_person_proj, clipping_gdf)

            if not clipped_lost_persons.empty:
                clipped_lost_persons.plot(ax=ax_inset, marker='*', color='red', markersize=250, zorder=3)

        ax_inset.set_xlim(center_x - medium_radius_m, center_x + medium_radius_m)
        ax_inset.set_ylim(center_y - medium_radius_m, center_y + medium_radius_m)

        circle = Circle((0.5, 0.5), 0.5, transform=ax_inset.transAxes, facecolor='none', edgecolor='black', linewidth=1)
        ax_inset.set_clip_path(circle)
        ax_inset.patch.set_alpha(0.0)

        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        pp1, c1, c2 = mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp1.set_zorder(0)
        c1.set_zorder(0)
        c2.set_zorder(0)

        pp2, c3, c4 = mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="black", lw=1.5, alpha=0.5)
        pp2.set_zorder(0)
        c3.set_zorder(0)
        c4.set_zorder(0)

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    # Establishes explicit Cartesian limits to circumvent rendering warnings
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels([f"{x / 1000:.1f}" for x in x_ticks], fontsize=18)
    ax.set_yticklabels([f"{y / 1000:.1f}" for y in y_ticks], fontsize=18)
    ax.set_xlabel("Easting (km)", fontsize=22)
    ax.set_ylabel("Northing (km)", fontsize=22)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"features_{item.size}_circular_magnified_final.pdf", bbox_inches='tight')
    
    if plot_show:
        plt.show()
