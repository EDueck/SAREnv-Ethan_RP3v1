# toolkit.py
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import sarenv
from sarenv.analytics import metrics, paths
from sarenv.utils import geo
from sarenv.utils.logging_setup import get_logger
from sarenv.utils.plot import plot_single_evaluation_results

log = get_logger()


class PathGeneratorConfig:
    """
    Configuration framework for standardising Unmanned Aerial Vehicle (UAV) path generation parameters.
    Provides a centralized interface to manage hardware constraints and pathing heuristics across simulated agents.
    """

    def __init__(self, num_drones: int, budget: float, **kwargs):
        """
        Initialises the operational constraints for the multi-agent path generation.

        Args:
            num_drones (int): Total number of independent UAV agents deployed in the search swarm.
            budget (float): Maximum operational distance limit (in meters) representing battery/flight constraints.
            **kwargs: Additional hardware and algorithmic variables.
                fov_degrees (float): Optical sensor field of view, determining the evaluation footprint. Defaults to 45.0.
                altitude_meters (float): Operational flight altitude. Defaults to 80.0.
                overlap_ratio (float): Required sensor footprint overlap for continuous coverage. Defaults to 0.25.
                path_point_spacing_m (float): Discretized resolution between decision waypoints. Defaults to 10.0.
                transition_distance_m (float): Spacing requirement for concentric search expansions. Defaults to 50.0.
                pizza_border_gap_m (float): Boundary tolerance for sector-based routing. Defaults to 15.0.
        """
        self.num_drones = num_drones
        self.budget = budget
        self.fov_degrees = kwargs.pop('fov_degrees', 45.0)
        self.altitude_meters = kwargs.pop('altitude_meters', 80.0)
        self.overlap_ratio = kwargs.pop('overlap_ratio', 0)
        self.path_point_spacing_m = kwargs.pop('path_point_spacing_m', 10.0)
        self.transition_distance_m = kwargs.pop('transition_distance_m', 50.0)
        self.pizza_border_gap_m = kwargs.pop('pizza_border_gap_m', 15.0)

        # Captures any additional experimental parameters provided dynamically
        self.additional_params = kwargs

    def get_params_dict(
            self,
            center_x: float,
            center_y: float,
            max_radius: float,
            probability_map: np.ndarray | None,
            bounds: tuple[float, float, float, float] | None,
            risk_map: np.ndarray | None = None,
    ) -> dict[str, float | int | np.ndarray | tuple | None]:
        """
        Compiles the complete spatial and operational parameter state for algorithmic path generation.

        Args:
            center_x: Initial deployment X coordinate (Last Known Point projection).
            center_y: Initial deployment Y coordinate.
            max_radius: Maximum geographical boundary for the search vector.
            probability_map: Spatial array modeling casualty location likelihood.
            bounds: Geographic limits of the evaluated terrain.
            risk_map: Topographical array representing the environmental hazard bias (beta).

        Returns:
            Dictionary payload encapsulating all constraints for the routing heuristic.
        """
        params = {
            'center_x': center_x,
            'center_y': center_y,
            'max_radius': max_radius,
            'probability_map': probability_map,
            'bounds': bounds,
            'risk_map': risk_map,
            'num_drones': self.num_drones,
            'budget': self.budget,
            'fov_deg': self.fov_degrees,
            'altitude': self.altitude_meters,
            'overlap': self.overlap_ratio,
            'path_point_spacing_m': self.path_point_spacing_m,
            'transition_distance_m': self.transition_distance_m,
            'border_gap_m': self.pizza_border_gap_m,
        }

        params.update(self.additional_params)
        return params


class PathGenerator:
    """
    Algorithmic wrapper class that enforces a uniform execution structure for diverse pathfinding heuristics.
    Ensures that both systematic patterns and risk-aware greedy models ingest spatial inputs consistently.
    """

    def __init__(self, name: str, func, path_generator_config: PathGeneratorConfig, description: str = ""):
        """
        Initialises a targeted path generation strategy.

        Args:
            name: Identifier for the specific pathfinding algorithm.
            func: The callable routing logic generating the geospatial trajectories.
            path_generator_config: The constraint configuration object.
            description: Contextual detailing of the heuristic's underlying mechanics.
        """
        self.name = name
        self.func = func
        self.description = description
        self.path_generator_config = path_generator_config

    def __call__(
            self,
            center_x: float,
            center_y: float,
            max_radius: float,
            probability_map: np.ndarray | None = None,
            bounds: tuple[float, float, float, float] | None = None,
            risk_map: np.ndarray | None = None,
    ) -> list[LineString]:
        """
        Executes the algorithm across the defined geospatial grid.

        Returns:
            A list of Shapely LineString geometries defining the calculated route for each active agent.
        """
        params = self.path_generator_config.get_params_dict(
            center_x=center_x,
            center_y=center_y,
            max_radius=max_radius,
            probability_map=probability_map,
            bounds=bounds,
            risk_map=risk_map,
        )
        
        # Ensures the topographical risk bias is explicitly forced into the execution state
        params['risk_map'] = risk_map
        return self.func(**params)


def get_default_path_generators(config: PathGeneratorConfig) -> dict[str, PathGenerator]:
    """
    Compiles the baseline suite of comparative pathfinding algorithms.

    Args:
        config: The standardized operational constraints to be applied across all baselines.

    Returns:
        Dictionary mapping algorithm names to their respective PathGenerator execution instances.
    """
    return {
        "RandomWalk": PathGenerator(
            name="RandomWalk",
            func=paths.generate_random_walk_path,
            path_generator_config=config,
            description="Stochastic navigation acting as the untrained statistical baseline."
        ),
        "Greedy": PathGenerator(
            name="Greedy",
            func=paths.generate_greedy_path,
            path_generator_config=config,
            description="Myopic heuristic optimising strictly for immediate adjacent spatial probability."
        ),
        "Spiral": PathGenerator(
            name="Spiral",
            func=paths.generate_spiral_path,
            path_generator_config=config,
            description="Exhaustive, deterministic global expansion pattern."
        ),
        "Concentric": PathGenerator(
            name="Concentric",
            func=paths.generate_concentric_circles_path,
            path_generator_config=config,
            description="Layered radial sweep designed for equidistant probability distributions."
        ),
        "Pizza": PathGenerator(
            name="Pizza",
            func=paths.generate_pizza_zigzag_path,
            path_generator_config=config,
            description="Sector-segmented sweeping model allowing parallel multi-agent allocation."
        )
    }


class ComparativeDatasetEvaluator:
    """
    Orchestrates large-scale batch evaluations across multiple terrain configurations.
    Aggregates topological intersections, performance metrics, and temporal decay records,
    streaming outputs dynamically to prevent memory overflow during extensive Monte Carlo simulations.
    """

    def __init__(self,
                 dataset_dirs,
                 evaluation_sizes,
                 num_drones,
                 budget,
                 num_lost_persons=100,
                 **kwargs):
        """
        Initialises the batch evaluation framework.

        Args:
            dataset_dirs (list): Target directories containing the geospatial environment data.
            evaluation_sizes (list): Array of resolution constraints to test against.
            num_drones (int): Total active agents per simulation instance.
            num_lost_persons (int): Target casualty count for generating statistical distributions.
            budget (int): Maximum travel limit per agent per simulation.
            **kwargs: Configurable constraints regarding sensors, spacing, and temporal discounting.
        """
        self.dataset_dirs = dataset_dirs or [f"sarenv_dataset/{i}" for i in range(1, 5)]
        self.evaluation_sizes = evaluation_sizes or ["medium"]
        self.num_drones = num_drones
        self.num_victims = num_lost_persons
        self.budget = budget
        self.discount_factor = kwargs.get("discount_factor", 0.999)

        path_generators = kwargs.get("path_generators")
        path_generator_config = kwargs.get("path_generator_config")

        if path_generator_config is None:
            self.path_generator_config = PathGeneratorConfig(
                num_drones=self.num_drones, 
                budget=self.budget,
                **kwargs.copy()
            )
        else:
            self.path_generator_config = path_generator_config

        if path_generators is None:
            self.path_generators = get_default_path_generators(self.path_generator_config)
        else:
            self.path_generators = {}
            for name, generator in path_generators.items():
                if isinstance(generator, PathGenerator):
                    self.path_generators[name] = generator
                else:
                    self.path_generators[name] = PathGenerator(
                        name=name,
                        func=generator,
                        path_generator_config=self.path_generator_config,
                        description=f"Custom generator implementation: {name}"
                    )

        # Volatile storage arrays for streaming compilation
        self.metrics_results = []
        self.time_series_results = []
        self.path_results = []

    def evaluate(self, output_dir):
        """
        Executes the exhaustive evaluation sequence across all loaded environments.
        Implements an immediate serialization strategy to disk to mitigate memory inflation.

        Args:
            output_dir (str): Destination path for CSV serialization.

        Returns:
            tuple: DataFrames containing the aggregated performance metrics and temporal search progressions.
        """
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            suffix = f"_n{self.path_generator_config.num_drones}_budget{self.path_generator_config.budget}"
            metrics_file = Path(output_dir) / f"comparative_metrics_results{suffix}.csv"
            time_series_file = Path(output_dir) / f"comparative_time_series_results{suffix}.csv"

            metrics_header_written = False
            time_series_header_written = False

        self.evaluators = []

        for dataset_dir in self.dataset_dirs:
            evaluator = ComparativeEvaluator(
                dataset_directory=dataset_dir,
                evaluation_sizes=self.evaluation_sizes,
                num_lost_persons=self.num_victims,
                budget=self.budget,
                path_generator_config=self.path_generator_config,
                path_generators=self.path_generators,
                num_drones=self.path_generator_config.num_drones
            )
            self.evaluators.append(evaluator)

            log.info(f"Initiating evaluation sequence for dataset topography: {dataset_dir}")
            results_df, time_series_data = evaluator.run_baseline_evaluations()

            if results_df.empty:
                continue

            # Parses the cross-sectional evaluation metrics
            dataset_metrics_results = []
            for _, row in results_df.iterrows():
                result_dict = row.to_dict()
                result_dict["Dataset"] = Path(dataset_dir).name
                dataset_metrics_results.append(result_dict)

            # Iterates through agent-specific longitudinal tracking to map time-to-rescue correlations
            dataset_time_series_results = []
            for algorithm_name, time_series_list in time_series_data.items():
                for i, time_series_data in enumerate(time_series_list):
                    individual_drone_data = time_series_data.get('individual_drone_data', [])

                    algorithm_results = results_df[results_df['Algorithm'] == algorithm_name]
                    environment_size = algorithm_results['Environment Size'].iloc[0] if not algorithm_results.empty else "unknown"

                    for drone_data in individual_drone_data:
                        drone_id = drone_data['drone_id']
                        cumulative_likelihood = drone_data['cumulative_likelihood']
                        positions = drone_data['positions']

                        # Models the temporal discovery rate by distributing total yield across operational agents
                        total_victims_found = results_df.iloc[-1]["Victims Found (%)"] / 100.0 * self.num_victims if not results_df.empty else 0
                        per_drone_victims = total_victims_found / len(individual_drone_data) if individual_drone_data else 0

                        for t, likelihood in enumerate(cumulative_likelihood):
                            path_x, path_y = None, None
                            if t < len(positions):
                                path_x, path_y = positions[t]

                            drone_victims = per_drone_victims * (t + 1) / len(cumulative_likelihood) if len(cumulative_likelihood) > 0 else 0

                            dataset_time_series_results.append({
                                "Algorithm": algorithm_name,
                                "Dataset": Path(dataset_dir).name,
                                "Environment Size": environment_size,
                                "Run": i,
                                "Agent_ID": drone_id,
                                "Time_Step": t,
                                "Cumulative_Likelihood": likelihood,
                                "Cumulative_Victims": drone_victims,
                                "Path_X": path_x,
                                "Path_Y": path_y,
                            })

            # Flushes operational data to disk to isolate memory overhead per terrain iteration
            if output_dir is not None and dataset_metrics_results:
                dataset_metrics_df = pd.DataFrame(dataset_metrics_results)
                dataset_time_series_df = pd.DataFrame(dataset_time_series_results)

                dataset_metrics_df.to_csv(metrics_file, mode='a', header=not metrics_header_written, index=False)
                metrics_header_written = True

                if not dataset_time_series_df.empty:
                    dataset_time_series_df.to_csv(time_series_file, mode='a', header=not time_series_header_written, index=False)
                    time_series_header_written = True

                log.info(f"Dataset {Path(dataset_dir).name} evaluation serialized to system storage.")

                self.metrics_results.extend(dataset_metrics_results)
                self.time_series_results.extend(dataset_time_series_results)

                del dataset_metrics_results
                del dataset_time_series_results
                del dataset_metrics_df
                del dataset_time_series_df

                self._clear_memory()
            else:
                self.metrics_results.extend(dataset_metrics_results)
                self.time_series_results.extend(dataset_time_series_results)

        metrics_df = pd.DataFrame(self.metrics_results)
        time_series_df = pd.DataFrame(self.time_series_results)

        if output_dir is not None:
            log.info("Batch simulation complete. Master files stored at:")
            log.info(f"  Metrics File: {metrics_file}")
            log.info(f"  Longitudinal Data: {time_series_file}")

        return metrics_df, time_series_df

    def save_results(self, metrics_df: pd.DataFrame, time_series_df: pd.DataFrame, output_dir: str = "results"):
        """
        Manually serializes DataFrames to CSV format. 
        Retained for architectural compatibility; primary streaming occurs natively within evaluate().
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        suffix = f"_n{self.path_generator_config.num_drones}_budget{self.path_generator_config.budget}"

        metrics_file = Path(output_dir) / f"comparative_metrics_results{suffix}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        log.info(f"Standard metrics explicitly written to: {metrics_file}")

        time_series_file = Path(output_dir) / f"comparative_time_series_results{suffix}.csv"
        time_series_df.to_csv(time_series_file, index=False)
        log.info(f"Longitudinal arrays explicitly written to: {time_series_file}")

    def get_metrics_results(self) -> pd.DataFrame:
        """Returns the in-memory aggregated metrics state."""
        return pd.DataFrame(self.metrics_results)

    def get_time_series_results(self) -> pd.DataFrame:
        """Returns the in-memory longitudinal performance state."""
        return pd.DataFrame(self.time_series_results)

    def get_paths_results(self) -> pd.DataFrame:
        """Returns the calculated geometry of all algorithmic paths."""
        return pd.DataFrame(self.path_results)

    def get_results_per_dataset(self) -> pd.DataFrame:
        """Groups the aggregated evaluations geographically."""
        return pd.DataFrame(self.metrics_results).groupby("Dataset").apply(lambda x: x.reset_index(drop=True))

    def summarize_results(self) -> pd.DataFrame:
        """
        Derives statistical summaries grouped by heuristic model.
        Calculates the 95% confidence intervals to evaluate algorithm stability.
        """
        results_df = pd.DataFrame(self.metrics_results)
        if results_df.empty:
            return pd.DataFrame()

        return results_df.groupby("Algorithm").agg(
            Mean_Likelihood_Score=('Likelihood Score', 'mean'),
            CI_Likelihood_Score=('Likelihood Score', lambda x: 1.96 * x.sem()),
            Mean_Time_Discounted=('Time-Discounted Score', 'mean'),
            CI_Time_Discounted=('Time-Discounted Score', lambda x: 1.96 * x.sem()),
            Mean_Victims_Found=('Victims Found (%)', 'mean'),
            CI_Victims_Found=('Victims Found (%)', lambda x: 1.96 * x.sem()),
            Mean_Triage_Score=('Triage Score', 'mean'),
            CI_Triage_Score=('Triage Score', lambda x: 1.96 * x.sem()),
            Mean_Area_Covered=('Area Covered (km²)', 'mean'),
            CI_Area_Covered=('Area Covered (km²)', lambda x: 1.96 * x.sem()),
            Mean_Path_Length=('Total Path Length (km)', 'mean'),
            CI_Path_Length=('Total Path Length (km)', lambda x: 1.96 * x.sem()),
        ).reset_index()

    def _clear_memory(self):
        """
        Garbage collection utility.
        Maintains a volatile buffer constraint to prevent the large-scale spatial arrays from overloading RAM.
        """
        if len(self.metrics_results) > 1000:
            self.metrics_results = self.metrics_results[-1000:]
        if len(self.time_series_results) > 10000:
            self.time_series_results = self.time_series_results[-10000:]


class ComparativeEvaluator:
    """
    Core intersection engine.
    Coordinates the alignment of the simulated UAV fleets, the algorithmic path models, and the
    geospatial hazard data to determine true performance against degrading casualty survival models.
    """

    def __init__(self,
                 dataset_directory="sarenv_dataset",
                 evaluation_sizes=None,
                 num_drones=1,
                 num_lost_persons=100,
                 budget=100_000,
                 **kwargs):
        """
        Initialises the operational environment and spatial evaluation parameters.

        Args:
            dataset_directory (str): Local path directing to the foundational map geometries.
            evaluation_sizes (list): Scope configurations for environment complexity mapping.
            num_drones (int): The scale of the multi-agent deployment.
            num_lost_persons (int): Sample size for the casualty distribution array.
            budget (int): Hardware constraint modeling flight distance.
            **kwargs: Dynamic overrides for path generation architecture.
        """
        self.dataset_directory = dataset_directory
        self.evaluation_sizes = evaluation_sizes or ["small", "medium", "large"]
        self.num_victims = num_lost_persons
        self.num_drones = num_drones
        self.budget = budget

        if "path_generator_config" in kwargs:
            self.path_generator_config = kwargs.pop("path_generator_config")
        else:
            self.path_generator_config = PathGeneratorConfig(
                num_drones=self.num_drones, 
                budget=self.budget,
                **kwargs.copy()
            )

        self.path_generators = kwargs.get("path_generators")
        self.loader = sarenv.DatasetLoader(dataset_directory=self.dataset_directory)
        self.environments = {}
        self.results = None
        self.time_series_data = {}

        if self.path_generators is None:
            self.path_generators = get_default_path_generators(self.path_generator_config)
        else:
            wrapped_generators = {}
            for name, generator in self.path_generators.items():
                if isinstance(generator, PathGenerator):
                    wrapped_generators[name] = generator
                else:
                    wrapped_generators[name] = PathGenerator(
                        name=name,
                        func=generator,
                        path_generator_config=self.path_generator_config,
                        description=f"Non-standard custom heuristic wrapper: {name}"
                    )
            self.path_generators = wrapped_generators

        self.load_datasets()

    def load_datasets(self):
        """
        Parses the geographical map data, extracting CRS matrices and probabilistically distributing
        casualties based on topographical severity to emulate real-world wilderness search conditions.
        """
        log.info(f"Fetching topography profiles for target complexities: {self.evaluation_sizes}")
        for size in self.evaluation_sizes:
            item = self.loader.load_environment(size)

            if not item:
                log.warning(f"Map resolution '{size}' invalid. Bypassing sequence.")
                continue

            data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
            victim_generator = sarenv.LostPersonLocationGenerator(item)

            # Embeds severity modifiers into the generated casualty grid to fuel the dynamic triage scoring
            victims_gdf = victim_generator.generate_locations(self.num_victims)

            if victims_gdf.empty:
                victims_gdf = gpd.GeoDataFrame(columns=["geometry", "severity"], crs=data_crs)

            self.environments[size] = {
                "item": item,
                "victims": victims_gdf,
                "crs": data_crs,
            }
        log.info("Geospatial topography and dynamic casualty coordinates fully resolved.")

    def run_baseline_evaluations(self) -> tuple[pd.DataFrame, dict]:
        """
        Executes the intersection models, evaluating how efficiently each algorithmic routing
        heuristic uncovers casualties before their temporal triage thresholds expire.

        Returns:
            tuple: Evaluated operational metrics and chronological pathing progressions.
        """
        if not self.environments:
            log.error("Execution failure: Topography data absent. Invoke load_datasets() prior to evaluation.")
            return pd.DataFrame()

        all_results = []
        self.time_series_data = {}

        for size, env_data in self.environments.items():
            item = env_data["item"]
            victims_gdf = env_data["victims"]

            log.info(f"--- Simulating Heuristic Performance on Topography: '{size}' ---")

            evaluator = metrics.PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.path_generator_config.fov_degrees,
                self.path_generator_config.altitude_meters,
                self.loader._meter_per_bin
            )
            center_proj = (
                gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
                .to_crs(env_data["crs"])
                .geometry.iloc[0]
            )

            import os
            import numpy as np

            # Dynamically binds the specific topographical risk map (beta bias) to the geometry
            # ensuring the agent evaluates environmental penalties alongside search probability.
            current_risk_map = getattr(item, 'risk_map', None)
            if current_risk_map is None:
                risk_map_path = os.path.join(self.dataset_directory, size, "risk_map.npy")
                if os.path.exists(risk_map_path):
                    current_risk_map = np.load(risk_map_path)
                    setattr(item, 'risk_map', current_risk_map)
                else:
                    log.warning(f"CRITICAL WARNING: Risk array missing at {risk_map_path}. Risk-aware pathing compromised.")

            for name, generator in self.path_generators.items():
                log.info(f"Executing mathematical routing via {name} on {size} environment.")
                generator: PathGenerator

                generated_paths = generator(
                    center_x=center_proj.x,
                    center_y=center_proj.y,
                    probability_map=item.heatmap,
                    bounds=item.bounds,
                    max_radius=item.radius_km * 1000,
                    risk_map=current_risk_map
                )

                all_metrics = evaluator.calculate_all_metrics(
                    generated_paths,
                    0.999
                )

                victim_metrics = all_metrics['victim_detection_metrics']

                result = {
                    "Dataset": size,
                    "Algorithm": name,
                    "Environment Type": item.environment_type,
                    "Climate": item.environment_climate,
                    "Environment Size": size,
                    "n_agents": self.path_generator_config.num_drones,
                    "Budget (m)": self.path_generator_config.budget,
                    "Likelihood Score": all_metrics['total_likelihood_score'],
                    "Time-Discounted Score": all_metrics['total_time_discounted_score'],
                    "Victims Found (%)": victim_metrics['percentage_found'],
                    "Triage Score": victim_metrics.get('triage_score', 0),
                    "Area Covered (km²)": all_metrics['area_covered'],
                    "Total Path Length (km)": all_metrics['total_path_length'],
                }

                all_results.append(result)

                if name not in self.time_series_data:
                    self.time_series_data[name] = []

                cumulative_likelihoods = all_metrics['cumulative_likelihoods']
                if cumulative_likelihoods:
                    individual_drone_data = []

                    for drone_idx, cum_lik in enumerate(cumulative_likelihoods):
                        if len(cum_lik) > 0 and drone_idx < len(generated_paths):
                            drone_path = generated_paths[drone_idx]

                            drone_positions = []
                            if not drone_path.is_empty and drone_path.length > 0:
                                # Discretizes the continuous LineString vector into discrete evaluation coordinates
                                # bound to the hardware optical sensor footprint resolution.
                                interpolation_resolution = int(np.ceil(self.loader._meter_per_bin / 2))
                                num_points = int(np.ceil(drone_path.length / interpolation_resolution)) + 1
                                distances = np.linspace(0, drone_path.length, num_points)

                                for d in distances:
                                    point = drone_path.interpolate(d)
                                    drone_positions.append((point.x, point.y))

                                while len(drone_positions) < len(cum_lik):
                                    if drone_positions:
                                        drone_positions.append(drone_positions[-1])
                                    else:
                                        drone_positions.append((0, 0))

                                drone_positions = drone_positions[:len(cum_lik)]
                            else:
                                drone_positions = [(0, 0)] * len(cum_lik)

                            individual
