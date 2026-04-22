# sarenv/core/lost_person.py
"""
Generates stochastic casualty distributions across the operational environment, 
calculating topological hazard exposure and assigning medical triage states 
based on continuous biological decay models.
"""
import math
import random
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sarenv.core.loading import SARDatasetItem
from sarenv.utils.logging_setup import get_logger

log = get_logger()


class LostPersonLocationGenerator:
    """
    Simulates the chaotic movement vectors of a lost individual, distributing 
    casualty coordinates according to empirically derived spatial probability matrices.
    """

    def __init__(self, dataset_item: SARDatasetItem):
        self.dataset_item = dataset_item
        self.features = dataset_item.features.copy()
        self.type_probabilities = {}
        self._calculate_weights()

    def _calculate_weights(self):
        """
        Derives the normalized probability mass for each distinct topological feature class.
        """
        if self.features.empty or 'area_probability' not in self.features.columns:
            log.warning("Spatial probability matrices absent. Topological weights cannot be calculated.")
            return

        type_weights = self.features.groupby('feature_type')['area_probability'].sum()
        self.type_probabilities = type_weights.to_dict()
        log.info(f"Topological feature weights established: {self.type_probabilities}")

    def _generate_random_point_in_polygon(self, poly: Polygon) -> Point:
        """
        Executes a stochastic coordinate generation within the explicit bounds 
        of a targeted topographical geometry.
        """
        min_x, min_y, max_x, max_y = poly.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if poly.contains(random_point):
                return random_point

    def generate_locations(self, n: int = 1, percent_random_samples=0) -> gpd.GeoDataFrame:
        """
        Generates a specified population of casualty distributions. Evaluates 
        the resultant topographical intersections to mathematically assign continuous 
        triage severities via a sigmoid probability function.

        Returns:
            gpd.GeoDataFrame: Geometries mapped against biological triage states.
        """
        if not self.type_probabilities:
            log.error("Topological probability distribution absent. Casualty generation aborted.")
            return gpd.GeoDataFrame()

        locations = []
        center_proj = gpd.GeoDataFrame(
            geometry=[Point(self.dataset_item.center_point)],
            crs="EPSG:4326"
        ).to_crs(self.features.crs).geometry.iloc[0]

        main_search_circle = center_proj.buffer(self.dataset_item.radius_km * 1000)

        # Phase 1: Stochastic Geographical Distribution
        while len(locations) < n:
            if random.random() < percent_random_samples:
                chosen_feature = self.features.sample(n=1).iloc[0]
                feature_buffer = chosen_feature.geometry.buffer(15)
                final_search_area = feature_buffer.intersection(main_search_circle)
            else:
                chosen_type = random.choices(
                    list(self.type_probabilities.keys()),
                    weights=list(self.type_probabilities.values()),
                    k=1
                )[0]
                type_gdf = self.features[self.features['feature_type'] == chosen_type]
                chosen_feature = type_gdf.sample(n=1, weights='area_probability').iloc[0]
                feature_buffer = chosen_feature.geometry.buffer(15)
                final_search_area = feature_buffer.intersection(main_search_circle)

            point = self._generate_random_point_in_polygon(final_search_area)
            if point:
                locations.append(point)

        if len(locations) < n:
            log.warning(f"Distribution constrained: Populated {len(locations)} of {n} target coordinates.")

        # Phase 2: Topological Hazard Evaluation
        points_gdf = gpd.GeoDataFrame(geometry=locations, crs=self.features.crs)

        # Executes a vectorized Boolean intersection to map casualties to environmental hazards
        joined = gpd.sjoin(points_gdf, self.features[['geometry', 'feature_type']], how="left", predicate="intersects")
        feature_matches = joined.groupby(joined.index)['feature_type'].apply(list).to_dict()

        # Defines environmental stress thresholds calibrated against medical survival literature
        risk_lookup = {
            "water": 10.0, "drainage": 8.5,
            "rock": 8.5, "linear": 3.0,
            "scrub": 2.0, "woodland": 1.5,
            "structure": 1.0, "road": 1.0, "brush": 1.0, "field": 1.0
        }

        severities = []
        for point_idx in points_gdf.index:
            possible_risks = [1.0]

            matched_types = feature_matches.get(point_idx, [])
            for ftype in matched_types:
                if isinstance(ftype, str):
                    for key, val in risk_lookup.items():
                        if key in ftype:
                            possible_risks.append(val)

            local_risk = max(possible_risks)

            # Phase 3: Biological Triage Modeling
            # Utilizes a continuous sigmoid (logistic) probability distribution to transition
            # casualties from 'Stable' to 'Critical' based on the cumulative environmental hazard.
            p_min = 0.05
            p_max = 0.85
            r_mid = 5.5  # Mathematical inflection point representing the critical hazard threshold
            k = 1.2      # Gradient steepness governing the acceleration of biological decay

            # Derives the absolute probability of critical trauma
            p_crit = p_min + (p_max - p_min) / (1.0 + math.exp(-k * (local_risk - r_mid)))

            status = 'Critical' if random.random() < p_crit else 'Stable'
            severities.append(status)

        points_gdf['severity'] = severities
        return points_gdf

    def generate_location(self) -> Point | None:
        """
        Maintains API integrity by executing a constrained, singular coordinate generation.
        """
        gdf = self.generate_locations(1)

        if not gdf.empty:
            return gdf.geometry.iloc[0]

        return None
