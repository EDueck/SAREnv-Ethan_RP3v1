# sarenv/core/environment.py
import concurrent.futures
import json
import os
from pathlib import Path

import elevation
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.stats import norm
from shapely.geometry import LineString, Point
from skimage.draw import (
    polygon as ski_polygon,
)

from ..io.osm_query import query_features
from ..utils import (
    logging_setup,
)
from ..utils.geo import image_to_world, world_to_image, get_utm_epsg
from ..utils.lost_person_behavior import (
    get_environment_radius_by_size,
    get_environment_radius,
)
from .geometries import GeoPolygon

log = logging_setup.get_logger()
EPS = 1e-9
from ..utils.lost_person_behavior import (
    FEATURE_PROBABILITIES,
    CLIMATE_TEMPERATE,
    CLIMATE_DRY,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
)


def process_feature_osm(key_val_pair, query_polygon_wgs84, projected_crs):
    """
    Extracts topological geometries from the OpenStreetMap API, resolving MultiGeometries 
    and executing coordinate reference system (CRS) reprojections for spatial consistency.
    Defined at the module level to maintain serialization for multiprocessing.
    """
    key, tag_dict = key_val_pair
    osm_geometries_dict = query_features(query_polygon_wgs84, tag_dict)

    if osm_geometries_dict is None:
        log.warning(f"No geometries returned from OSM query for features: {key}")
        return key, None

    all_geoms_for_key = []
    for geom in osm_geometries_dict.values():
        if geom is not None and not geom.is_empty:
            if hasattr(geom, "geoms"):
                all_geoms_for_key.extend(
                    g for g in geom.geoms if g is not None and not g.is_empty
                )
            else:
                all_geoms_for_key.append(geom)

    if not all_geoms_for_key:
        log.info(
            f"No valid geometries found for feature type '{key}' after filtering empty coordinate arrays."
        )
        return key, None

    gdf_wgs84 = gpd.GeoDataFrame(geometry=all_geoms_for_key, crs="EPSG:4326")
    gdf_projected = gdf_wgs84.to_crs(projected_crs)
    log.info(f"Processed {len(gdf_projected)} geometries for feature classification '{key}'")
    return key, gdf_projected


def interpolate_line(line, distance):
    """
    Discretizes continuous geometrical vectors into equidistant coordinate arrays 
    to facilitate spatial rasterization.
    """
    if distance <= 0:
        return [shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])]

    points = []
    for i in range(len(line.coords) - 1):
        segment = LineString([line.coords[i], line.coords[i + 1]])
        segment_length = segment.length
        num_points = max(1, int(segment_length / distance))
        points.extend(
            segment.interpolate(float(j) / num_points * segment_length)
            for j in range(num_points)
        )
    points.append(shapely.Point(line.coords[-1]))
    return points


def generate_heatmap_task(
        feature_key,
        geometry_series,
        sample_distance,
        xedges,
        yedges,
        meter_per_bin,
        minx,
        miny,
        buffer_val,
        infill_geometries=True,
):
    """
    Rasterizes continuous topological vectors into a discrete two-dimensional spatial array.
    """
    heatmap = np.zeros((len(yedges) - 1, len(xedges) - 1), dtype=float)
    skipped_points = 0

    for geometry in geometry_series:
        if geometry is None or geometry.is_empty:
            continue

        current_geom_img_coords_x = []
        current_geom_img_coords_y = []

        if isinstance(geometry, LineString):
            points_on_line = interpolate_line(geometry, sample_distance)
            if points_on_line:
                world_x = [p.x for p in points_on_line]
                world_y = [p.y for p in points_on_line]
                img_x, img_y = world_to_image(
                    np.array(world_x),
                    np.array(world_y),
                    meter_per_bin,
                    minx,
                    miny,
                    buffer_val,
                )
                current_geom_img_coords_x.extend(img_x)
                current_geom_img_coords_y.extend(img_y)
        elif isinstance(geometry, shapely.geometry.Polygon):
            if infill_geometries:
                ext_coords_world = np.array(list(geometry.exterior.coords))
                ext_coords_img_x_arr, ext_coords_img_y_arr = world_to_image(
                    ext_coords_world[:, 0],
                    ext_coords_world[:, 1],
                    meter_per_bin,
                    minx,
                    miny,
                    buffer_val,
                )
                rr, cc = ski_polygon(
                    ext_coords_img_y_arr, ext_coords_img_x_arr, shape=heatmap.shape
                )
                current_geom_img_coords_y.extend(rr)
                current_geom_img_coords_x.extend(cc)
            else:
                points_on_exterior = interpolate_line(
                    geometry.exterior, sample_distance
                )
                if points_on_exterior:
                    world_x = [p.x for p in points_on_exterior]
                    world_y = [p.y for p in points_on_exterior]
                    img_x, img_y = world_to_image(
                        np.array(world_x),
                        np.array(world_y),
                        meter_per_bin,
                        minx,
                        miny,
                        buffer_val,
                    )
                    current_geom_img_coords_x.extend(img_x)
                    current_geom_img_coords_y.extend(img_y)

            for interior in geometry.interiors:
                interior_coords_world = np.array(list(interior.coords))
                interior_coords_img_x, interior_coords_img_y = world_to_image(
                    interior_coords_world[:, 0],
                    interior_coords_world[:, 1],
                    meter_per_bin,
                    minx,
                    miny,
                    buffer_val,
                )
                for ix, iy in zip(interior_coords_img_x, interior_coords_img_y):
                    if (
                            ix in current_geom_img_coords_x
                            and iy in current_geom_img_coords_y
                    ):
                        idx = current_geom_img_coords_x.index(ix)
                        if current_geom_img_coords_y[idx] == iy:
                            current_geom_img_coords_x.pop(idx)
                            current_geom_img_coords_y.pop(idx)
        elif isinstance(geometry, shapely.geometry.Point):
            skipped_points += 1
        else:
            log.warning(
                f"Unsupported geometry type for spatial array: {type(geometry)} for feature {feature_key}"
            )
            continue

        if current_geom_img_coords_x:
            valid_indices = [
                i
                for i, (x, y) in enumerate(
                    zip(current_geom_img_coords_x, current_geom_img_coords_y)
                )
                if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]
            ]
            if valid_indices:
                valid_x = np.array(current_geom_img_coords_x)[valid_indices]
                valid_y = np.array(current_geom_img_coords_y)[valid_indices]
                heatmap[valid_y, valid_x] = 1

    if skipped_points > 0:
        log.warning(
            f"Skipped {skipped_points} Point geometries for feature {feature_key} during spatial generation."
        )

    return heatmap


class EnvironmentBuilder:
    """
    Constructs the geospatial bounding constraints and resolution parameters for the operational terrain.
    """
    def __init__(self):
        self.polygon = None
        self.meter_per_bin = 3
        self.sample_distance = 1
        self.buffer = 0
        self.tags = {}
        self.projected_crs = None

    def set_polygon(self, polygon):
        self.polygon = polygon
        return self

    def set_sample_distance(self, sample_distance):
        self.sample_distance = sample_distance
        return self

    def set_meter_per_bin(self, meter_per_bin):
        self.meter_per_bin = meter_per_bin
        return self

    def set_projected_crs(self, crs: str):
        self.projected_crs = crs
        return self

    def set_buffer(self, buffer_val):
        self.buffer = buffer_val
        return self

    def set_features(self, features):
        if not isinstance(features, dict):
            raise ValueError("Features must be configured as a dictionary mapping.")
        self.tags.update(features)
        return self

    def set_feature(self, name, tags):
        self.tags[name] = tags
        return self

    def build(self):
        if self.polygon is None:
            raise ValueError("Spatial polygon constraint must be defined prior to environmental initialization.")
        if self.projected_crs is None:
            raise ValueError("Projected Coordinate Reference System (CRS) must be established.")

        return Environment(
            self.polygon,
            self.sample_distance,
            self.meter_per_bin,
            self.buffer,
            self.tags,
            self.projected_crs,
        )


class Environment:
    """
    Encapsulates the mathematical and topological state of the evaluation environment, 
    managing geospatial matrices, Digital Elevation Models (DEM), and distinct structural probability arrays.
    """
    def __init__(
            self,
            bounding_polygon,
            sample_distance,
            meter_per_bin,
            buffer_val,
            tags,
            projected_crs,
    ):
        self.tags = tags
        self.sample_distance = sample_distance
        self.meter_per_bin = meter_per_bin
        self.buffer_val = buffer_val
        self.projected_crs = projected_crs

        self.polygon: GeoPolygon | None = None
        self.xedges: np.ndarray | None = None
        self.yedges: np.ndarray | None = None
        self.heatmaps: dict[str, np.ndarray | None] = {}
        self.features: dict[str, gpd.GeoDataFrame | None] = {}
        self.heightmap: np.ndarray | None = None

        self.polygon = GeoPolygon(
            bounding_polygon, crs="EPSG:4326"
        )
        self.polygon.set_crs(
            self.projected_crs
        )
        log.info(f"Environment topological CRS established: {self.polygon.crs}")

        self.area = self.polygon.geometry.area
        log.info(
            "Calculated spatial area: %s m² (approx. %.2f km²)", self.area, self.area / 1e6
        )
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.geometry.bounds

        num_bins_x = int(
            abs(self.maxx - self.minx + 2 * self.buffer_val) / self.meter_per_bin
        )
        num_bins_y = int(
            abs(self.maxy - self.miny + 2 * self.buffer_val) / self.meter_per_bin
        )

        if num_bins_x <= 0:
            num_bins_x = 1
        if num_bins_y <= 0:
            num_bins_y = 1

        log.info("Matrix dimensional bounds - X: %i, Y: %i", num_bins_x, num_bins_y)

        self.xedges = np.linspace(
            self.minx - self.buffer_val, self.maxx + self.buffer_val, num_bins_x + 1
        )
        self.yedges = np.linspace(
            self.miny - self.buffer_val, self.maxy + self.buffer_val, num_bins_y + 1
        )

        self._load_features()

    def _load_features(self):
        """Dispatches asynchronous geometry extraction tasks across the OSM taxonomy."""
        query_polygon_wgs84 = GeoPolygon(self.polygon.geometry, crs=self.polygon.crs)
        query_polygon_wgs84.set_crs("EPSG:4326")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            tasks = [
                (item, query_polygon_wgs84, self.projected_crs)
                for item in self.tags.items()
            ]
            future_to_key = {
                executor.submit(process_feature_osm, *task): task[0][0]
                for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    _, feature_gdf = future.result()
                    self.features[key] = feature_gdf
                    if feature_gdf is not None:
                        log.info(
                            f"Archived {len(feature_gdf)} topological structures for '{key}' under CRS {feature_gdf.crs}"
                        )
                    else:
                        log.info(f"Topological features absent for classification '{key}'")
                except Exception as exc:
                    log.error(f"Execution failure processing topology {key}: {exc}", exc_info=True)
                    self.features[key] = None

    def interpolate_line(self, line, distance):
        if distance <= 0:
            return [shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])]

        points = []
        for i in range(len(line.coords) - 1):
            segment = LineString([line.coords[i], line.coords[i + 1]])
            segment_length = segment.length
            num_points = max(1, int(segment_length / distance))
            points.extend(
                segment.interpolate(float(j) / num_points * segment_length)
                for j in range(num_points)
            )
        points.append(
            shapely.Point(line.coords[-1])
        )
        return points

    def generate_heatmaps(self):
        """Orchestrates the parallel rasterization of topological features into spatial probability distributions."""
        log.info("Initiating spatial rasterization across all feature classes...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            tasks = [
                (
                    key,
                    feature_gdf.geometry,
                    self.sample_distance,
                    self.xedges,
                    self.yedges,
                    self.meter_per_bin,
                    self.minx,
                    self.miny,
                    self.buffer_val,
                )
                for key, feature_gdf in self.features.items()
                if feature_gdf is not None and not feature_gdf.empty
            ]
            future_to_key = {
                executor.submit(generate_heatmap_task, *task): task[0] for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    heatmap_result = future.result()
                    self.heatmaps[key] = heatmap_result
                    log.info(f"Spatial probability array established for '{key}'")
                except Exception as exc:
                    log.error(
                        f"Mathematical fault compiling array for '{key}': {exc}", exc_info=True
                    )
                    self.heatmaps[key] = None
        log.info("Spatial rasterization complete.")

    def get_combined_heatmap(self):
        """Aggregates isolated feature arrays into a singular global probability distribution matrix."""
        if not self.heatmaps:
            log.info("Isolated arrays uninitialized. Triggering primary rasterization sequence.")
            self.generate_heatmaps()
            
        if not any(h is not None for h in self.heatmaps.values()):
            log.warning(
                "Geospatial arrays devoid of structural data. Substituting uniform zero-matrix."
            )
            if self.xedges is None or self.yedges is None:
                return None
            return np.zeros((len(self.yedges) - 1, len(self.xedges) - 1), dtype=float)

        if self.xedges is None or self.yedges is None:
            log.error("Matrix compilation failed: Dimension bounds uninitialized.")
            return None

        combined_heatmap = np.zeros(
            (len(self.yedges) - 1, len(self.xedges) - 1), dtype=float
        )

        for key, individual_heatmap in self.heatmaps.items():
            if individual_heatmap is None:
                log.warning(
                    f"Bypassing classification '{key}' during array aggregation due to null state."
                )
                continue
            if individual_heatmap.shape != combined_heatmap.shape:
                log.error(
                    f"Dimensional discrepancy for '{key}': {individual_heatmap.shape} vs expected {combined_heatmap.shape}"
                )
                continue

            alpha = FEATURE_PROBABILITIES.get(key, 0)
            filtered_heatmap_part = individual_heatmap.astype(float) * alpha

            combined_heatmap = np.maximum(combined_heatmap, filtered_heatmap_part)
        return combined_heatmap

    def generate_heightmap(self, output_dir=".") -> np.ndarray | None:
        """
        Derives an operational heightmap matrix by extracting local Digital Elevation Model (DEM) data.
        """
        log.info("Constructing topographical heightmap from foundational DEM datasets...")
        if self.polygon is None:
            log.error("Heightmap compilation aborted: Spatial bounds uninitialized.")
            return None

        minx, miny, maxx, maxy = self.polygon.set_crs("EPSG:4326").geometry.bounds
        output_path = os.path.join(output_dir, "temp_dem.tif")
        log.info(
            f"Retrieving geospatial DEM telemetry for bounds: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}"
        )
        os.makedirs(output_dir, exist_ok=True)

        try:
            elevation.clip(bounds=(minx, miny, maxx, maxy), output=output_path)
            elevation.clean()

            log.info(f"DEM extraction serialized locally to {output_path}")

            with rasterio.open(output_path) as dem_dataset:
                x_centers = (self.xedges[:-1] + self.xedges[1:]) / 2
                y_centers = (self.yedges[:-1] + self.yedges[1:]) / 2
                xv, yv = np.meshgrid(x_centers, y_centers)

                points_proj = [Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
                grid_gdf_proj = gpd.GeoDataFrame(
                    geometry=points_proj, crs=self.projected_crs
                )

                grid_gdf_dem_crs = grid_gdf_proj.to_crs(dem_dataset.crs)
                coords = [(p.x, p.y) for p in grid_gdf_dem_crs.geometry]

                elevations = [val[0] for val in dem_dataset.sample(coords)]

                heightmap_grid = np.array(elevations).reshape(
                    len(y_centers), len(x_centers)
                )
                self.heightmap = heightmap_grid
                log.info(
                    f"Topographical heightmap array established with dimensions {self.heightmap.shape}"
                )
                return self.heightmap

        except Exception as e:
            log.error(
                f"Catastrophic failure compiling local DEM heightmap: {e}", exc_info=True
            )
            return None
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
                log.info(f"Purged volatile DEM cache at: {output_path}")


class DataGenerator:
    """
    Orchestrates the synthesis of empirical wilderness search data, generating spatial probability 
    gradients based on log-normal casualty distributions and overlaying topographical risk matrices.
    """

    def __init__(self):
        # Maps OSM classifications to foundational SAR terrain categories
        self.tags_mapping = {
            "structure": {
                "building": True,
                "man_made": True,
                "bridge": True,
                "tunnel": True,
            },
            "road": {"highway": True, "tracktype": True},
            "linear": {
                "railway": True,
                "barrier": True,
                "fence": True,
                "wall": True,
                "pipeline": True,
            },
            "drainage": {"waterway": ["drain", "ditch", "culvert", "canal"]},
            "water": {
                "natural": ["water", "wetland"],
                "water": True,
                "wetland": True,
                "reservoir": True,
            },
            "brush": {"landuse": ["grass"]},
            "scrub": {"natural": "scrub"},
            "woodland": {"landuse": ["forest", "wood"], "natural": "wood"},
            "field": {"landuse": ["farmland", "farm", "meadow"]},
            "rock": {"natural": ["rock", "bare_rock", "scree", "cliff"]},
        }

        # Empirically calibrated environmental hazard weights driving the operational risk bias (Beta)
        self.risk_scores = {
            "water": 10.0,       # Extreme Urgency: Rapid drowning risk and acute hypothermia onset.
            "drainage": 9.0,     # Extreme Urgency: Fast-moving currents and steep topographical embankments.
            "rock": 8.5,         # Very High Urgency: Severe mechanism of injury related to catastrophic blunt trauma.
            "woodland": 6.0,     # High Urgency: Elevated risk of isolation and heavy canopy obscuration.
            "scrub": 5.0,        # Moderate-High Urgency: General off-trail environmental exposure.
            "brush": 4.0,        # Moderate Urgency: Off-trail exposure with increased aerial visibility.
            "linear": 3.0,       # Moderate: Fall/trip hazards juxtaposed with easily locatable navigation handrails.
            "field": 1.5,        # Low-Moderate: High aerial visibility and low baseline mechanism of injury.
            "structure": 1.0,    # Minimum Urgency: High probability of shelter and survivability.
            "road": 1.0          # Minimum Urgency: Immediate localized access for emergency medical services.
        }

        self._builder = EnvironmentBuilder()
        for feature_category, osm_tags in self.tags_mapping.items():
            self._builder.set_feature(feature_category, osm_tags)

    def _lognormal_distribution_estimation(
            self, climate, environment
    ) -> tuple[float, float]:
        """
        Derives the parameters (mu, sigma) of a continuous log-normal spatial probability distribution 
        by executing a least-squares fit against empirically established wilderness search percentiles.
        """
        percentiles = np.array([0.25, 0.5, 0.75, 0.95])
        values = get_environment_radius(environment, climate)
        log_values = np.log(values)
        z_scores = norm.ppf(percentiles)

        A = np.vstack([np.ones_like(z_scores), z_scores]).T
        mu, sigma = np.linalg.lstsq(A, log_values, rcond=None)[0]

        model_log_values = mu + sigma * z_scores
        error = np.sqrt(np.mean((model_log_values - log_values) ** 2))
        log.info(f"Log-normal mathematical fit root mean squared error: {error:.4f}")
        log.info(f"Derived parameters - Log-normal mu: {mu:.4f}, sigma: {sigma:.4f}")
        return mu, abs(sigma)

    def _create_circular_polygon(
            self, center_lon: float, center_lat: float, radius_km: float
    ) -> dict:
        """Constructs a projected geographical bounding radius relative to the Last Known Point (LKP)."""
        point = shapely.Point(center_lon, center_lat)
        gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")

        projected_crs = get_utm_epsg(center_lon, center_lat)
        gdf_proj = gdf.to_crs(projected_crs)

        radius_meters = radius_km * 1000
        circle_proj = gdf_proj.buffer(radius_meters).iloc[0]

        circle_gdf_proj = gpd.GeoDataFrame(
            [1], geometry=[circle_proj], crs=projected_crs
        )
        circle_gdf_wgs84 = circle_gdf_proj.to_crs("EPSG:4326")

        return circle_gdf_wgs84.geometry.iloc[0]

    def generate_environment(
            self,
            center_point: tuple[float, float],
            size: str,
            environment_climate,
            environment_type,
            meter_per_bin: int = 30,
    ) -> "Environment | None":
        """Initializes a defined topological environment constrained by the calculated empirical search radius."""
        center_lon, center_lat = center_point
        radius_km = get_environment_radius_by_size(
            environment_type, environment_climate, size
        )
        projected_crs = get_utm_epsg(center_lon, center_lat)

        log.info(
            f"Compiling spatial data for LKP ({center_lon:.4f}, {center_lat:.4f}) utilizing constraint '{size}' (Radius: {radius_km} km)."
        )
        geojson_poly = self._create_circular_polygon(center_lon, center_lat, radius_km)

        try:
            env = (
                self._builder.set_polygon(geojson_poly)
                .set_projected_crs(projected_crs)
                .set_meter_per_bin(meter_per_bin)
                .build()
            )
            return env
        except Exception as e:
            log.error(
                f"Catastrophic failure during environment topology generation: {e}", exc_info=True
            )
            return None

    def generate_environment_from_polygon(
            self,
            polygon: shapely.geometry.Polygon | dict,
            meter_per_bin: int = 30,
    ) -> "Environment | None":
        """
        Initializes an operational environment explicitly bound by a provided geospatial polygon.
        """
        try:
            if isinstance(polygon, dict):
                if "coordinates" in polygon:
                    coords = polygon["coordinates"][0] if polygon["coordinates"] else []
                    geom_polygon = shapely.geometry.Polygon(coords)
                else:
                    msg = "Geospatial dictionary format mathematically invalid."
                    raise ValueError(msg)
            elif isinstance(polygon, shapely.geometry.Polygon):
                geom_polygon = polygon
            else:
                msg = "Polygon parameters mandate a Shapely object or strictly formatted GeoJSON dictionary."
                raise ValueError(msg)

            centroid = geom_polygon.centroid
            center_lon, center_lat = centroid.x, centroid.y
            projected_crs = get_utm_epsg(center_lon, center_lat)

            log.info(
                f"Generating terrain topology originating from geometric centroid ({center_lon:.4f}, {center_lat:.4f})"
            )

            return (
                self._builder.set_polygon(geom_polygon)
                .set_projected_crs(projected_crs)
                .set_meter_per_bin(meter_per_bin)
                .build()
            )

        except Exception as e:
            log.error(
                f"Catastrophic failure compiling explicit polygon boundaries: {e}",
                exc_info=True
            )
            return None

    def export_dataset(
            self,
            center_point: tuple[float, float],
            output_directory: str,
            environment_type,
            environment_climate,
            meter_per_bin: int = 30,
    ):
        """
        Executes the final matrix synthesis, combining the log-normal distance probability gradient 
        with localized topological features. Serializes the resulting structural GeoJSON, spatial 
        probability matrix, and explicit topographical risk map to disk.
        """
        log.info(f"--- Initiating comprehensive topology serialization to '{output_directory}' ---")
        os.makedirs(output_directory, exist_ok=True)

        log.info("--- Establishing foundational boundary constraints via 'xlarge' parameter ---")
        master_env = self.generate_environment(
            center_point, "xlarge", environment_climate, environment_type, meter_per_bin
        )
        if not master_env:
            log.error("Primary environment construction failed. Halting serialization.")
            return

        master_features_list = []
        for key, gdf in master_env.features.items():
            if gdf is not None and not gdf.empty:
                temp_gdf = gdf[~gdf.geometry.type.isin(["Point"])].copy()
                temp_gdf["feature_type"] = key
                master_features_list.append(temp_gdf)

        if not master_features_list:
            log.error("Topological features absent. Feature arrays will be omitted from export.")
            master_features_gdf = gpd.GeoDataFrame(
                columns=["geometry", "feature_type", "environemnt_type", "climate", " center_point", "meter_per_bin",
                         "radius_km", "bounds", ],
                geometry="geometry",
                crs=master_env.projected_crs,
            )
        else:
            master_features_gdf = pd.concat(master_features_list, ignore_index=True)

        log.info("--- Executing spatial probability gradient calculations ---")
        feature_heatmap = master_env.get_combined_heatmap()

        if feature_heatmap is None:
            log.error("Spatial feature array compilation failed. Halting serialization.")
            return

        log.info("Deriving log-normal distance gradient relative to LKP...")
        mu, sigma = self._lognormal_distribution_estimation(
            environment_climate, environment_type
        )
        h, w = feature_heatmap.shape
        y_indices, x_indices = np.mgrid[0:h, 0:w]

        center_lon, center_lat = center_point
        center_point_gdf = gpd.GeoDataFrame(
            geometry=[shapely.Point(center_lon, center_lat)], crs="EPSG:4326"
        )
        center_point_proj = center_point_gdf.to_crs(master_env.projected_crs)
        center_x_world = center_point_proj.geometry.x.iloc[0]
        center_y_world = center_point_proj.geometry.y.iloc[0]

        x_coords_world = master_env.xedges[x_indices]
        y_coords_world = master_env.yedges[y_indices]
        dist = np.sqrt(
            (x_coords_world - center_x_world) ** 2
            + (y_coords_world - center_y_world) ** 2
        )
        dist_km = dist / 1000.0
        dist_km = np.clip(dist_km, 1e-6, None)

        bell_curve_map = (1 / (dist_km * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(dist_km) - mu) ** 2) / (2 * sigma ** 2)
        )
        spatial_probability_map = bell_curve_map / bell_curve_map.sum()

        feature_heatmap_sum = np.sum(feature_heatmap)
        if feature_heatmap_sum == 0:
            log.warning(
                "Feature matrix registered zero density. Restricting gradient entirely to log-normal distribution."
            )
            feature_prob_map = np.ones_like(feature_heatmap)
        else:
            feature_prob_map = feature_heatmap / feature_heatmap_sum

        log.info("Blending topological features with baseline log-normal gradient...")
        combined_map_unnormalized = spatial_probability_map * feature_prob_map
        total_sum = np.sum(combined_map_unnormalized)
        if total_sum > 0:
            final_probability_map = combined_map_unnormalized / total_sum
        else:
            log.warning("Blended probability matrix sums to zero. Exporting null array.")
            final_probability_map = combined_map_unnormalized

        assert (
                np.isclose(np.sum(final_probability_map), 1.0, atol=1e-6)
                or np.sum(final_probability_map) == 0
        )

        log.info("Executing area-weighted probability calculations for metadata tracking...")

        def calculate_distance_probability(geom, center_x, center_y, mu, sigma):
            centroid = geom.centroid
            dist_meters = np.sqrt(
                (centroid.x - center_x) ** 2 + (centroid.y - center_y) ** 2
            )
            dist_km = max(dist_meters / 1000.0, 1e-6)

            probability = (1 / (dist_km * sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(dist_km) - mu) ** 2) / (2 * sigma ** 2)
            )
            return probability

        distance_prob = master_features_gdf.geometry.apply(
            calculate_distance_probability,
            args=(center_x_world, center_y_world, mu, sigma),
        )

        def get_area(geom):
            if isinstance(geom, shapely.Polygon):
                return geom.area
            elif isinstance(geom, LineString):
                return geom.buffer(15).area
            return 0

        master_features_gdf["area"] = master_features_gdf.geometry.apply(get_area)
        area_influence = master_features_gdf["area"] * master_features_gdf["feature_type"].map(FEATURE_PROBABILITIES)

        combined_probability = area_influence * distance_prob

        total_prob_sum = combined_probability.sum()
        if total_prob_sum > 0:
            master_features_gdf["area_probability"] = (
                    combined_probability / total_prob_sum
            )
        else:
            master_features_gdf["area_probability"] = 0

        geojson_path = os.path.join(output_directory, "features.geojson")
        master_features_gdf.to_crs("EPSG:4326", inplace=True)
        geojson_dict = master_features_gdf.__geo_interface__

        geojson_dict["environment_type"] = environment_type
        geojson_dict["climate"] = environment_climate
        geojson_dict["center_point"] = center_point
        geojson_dict["meter_per_bin"] = meter_per_bin
        geojson_dict["radius_km"] = get_environment_radius_by_size(
            environment_type, environment_climate, "xlarge"
        )
        geojson_dict["bounds"] = [
            master_env.minx,
            master_env.miny,
            master_env.maxx,
            master_env.maxy,
        ]

        with open(geojson_path, "w") as f:
            json.dump(geojson_dict, f)
        log.info(f"Serialized {len(master_features_gdf)} topological structures to {geojson_path}")

        heatmap_path = os.path.join(output_directory, "heatmap.npy")
        np.save(heatmap_path, final_probability_map)
        log.info(
            f"Serialized final probability matrix (Dimensions: {final_probability_map.shape}) to {heatmap_path}"
        )

        log.info("Constructing explicit topographical hazard matrix (Beta Array)...")
        risk_map = np.ones_like(final_probability_map)

        for key, individual_heatmap in master_env.heatmaps.items():
            if individual_heatmap is not None:
                risk_val = self.risk_scores.get(key, 1.0)
                feature_mask = individual_heatmap > 0
                risk_map[feature_mask] = np.maximum(risk_map[feature_mask], risk_val)

        risk_map_path = os.path.join(output_directory, "risk_map.npy")
        np.save(risk_map_path, risk_map)
        log.info(f"Serialized localized hazard matrix (Maximum Beta: {np.max(risk_map)}) to {risk_map_path}")

        log.info("--- Structural environment serialization successfully concluded. ---")

    def export_dataset_from_polygon(
            self,
            polygon: shapely.geometry.Polygon | dict,
            output_directory: str,
            environment_type,
            environment_climate,
            meter_per_bin: int = 30,
    ):
        """
        Generates and serializes a localized topographical dataset rigidly bound by a provided geospatial geometry.
        """
        log.info(f"--- Initiating geometrically bound topology serialization to '{output_directory}' ---")
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        log.info("--- Deriving boundary matrices from provided geometrical limits ---")
        master_env = self.generate_environment_from_polygon(polygon, meter_per_bin)
        if not master_env:
            log.error("Geometric boundary resolution failed. Halting serialization.")
            return

        if isinstance(polygon, dict):
            coords = polygon["coordinates"][0] if polygon["coordinates"] else []
            geom_polygon = shapely.geometry.Polygon(coords)
        else:
            geom_polygon = polygon

        centroid = geom_polygon.centroid
        center_point = (centroid.x, centroid.y)

        master_features_list = []
        for key, gdf in master_env.features.items():
            if gdf is not None and not gdf.empty:
                temp_gdf = gdf[~gdf.geometry.type.isin(["Point"])].copy()
                temp_gdf["feature_type"] = key
                master_features_list.append(temp_gdf)

        if not master_features_list:
            log.error("Topological features absent. Feature arrays will be omitted from export.")
            master_features_gdf = gpd.GeoDataFrame(
                columns=["geometry", "feature_type", "environment_type", "climate", "center_point", "meter_per_bin",
                         "bounds"],
                geometry="geometry",
                crs=master_env.projected_crs,
            )
        else:
            master_features_gdf = pd.concat(master_features_list, ignore_index=True)

        log.info("--- Executing spatial probability gradient calculations ---")
        feature_heatmap = master_env.get_combined_heatmap()

        if feature_heatmap is None:
            log.error("Spatial feature array compilation failed. Halting serialization.")
            return

        log.info("Deriving log-normal distance gradient relative to localized centroid...")
        mu, sigma = self._lognormal_distribution_estimation(
            environment_climate, environment_type
        )
        h, w = feature_heatmap.shape
        y_indices, x_indices = np.mgrid[0:h, 0:w]

        center_lon, center_lat = center_point
        center_point_gdf = gpd.GeoDataFrame(
            geometry=[shapely.Point(center_lon, center_lat)], crs="EPSG:4326"
        )
        center_point_proj = center_point_gdf.to_crs(master_env.projected_crs)
        center_x_world = center_point_proj.geometry.x.iloc[0]
        center_y_world = center_point_proj.geometry.y.iloc[0]

        x_coords_world = master_env.xedges[x_indices]
        y_coords_world = master_env.yedges[y_indices]
        dist = np.sqrt(
            (x_coords_world - center_x_world) ** 2
            + (y_coords_world - center_y_world) ** 2
        )
        dist_km = dist / 1000.0
        dist_km = np.clip(dist_km, 1e-6, None)

        bell_curve_map = (1 / (dist_km * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(dist_km) - mu) ** 2) / (2 * sigma ** 2)
        )
        spatial_probability_map = bell_curve_map / bell_curve_map.sum()

        feature_heatmap_sum = np.sum(feature_heatmap)
        if feature_heatmap_sum == 0:
            log.warning(
                "Feature matrix registered zero density. Restricting gradient entirely to log-normal distribution."
            )
            feature_prob_map = np.ones_like(feature_heatmap)
        else:
            feature_prob_map = feature_heatmap / feature_heatmap_sum

        log.info("Blending topological features with baseline log-normal gradient...")
        combined_map_unnormalized = spatial_probability_map * feature_prob_map
        total_sum = np.sum(combined_map_unnormalized)
        if total_sum > 0:
            final_probability_map = combined_map_unnormalized / total_sum
        else:
            log.warning("Blended probability matrix sums to zero. Exporting null array.")
            final_probability_map = combined_map_unnormalized

        assert (
                np.isclose(np.sum(final_probability_map), 1.0, atol=1e-6)
                or np.sum(final_probability_map) == 0
        )

        log.info("Executing area-weighted probability calculations for metadata tracking...")

        def calculate_distance_probability(geom, center_x, center_y, mu, sigma):
            centroid = geom.centroid
            dist_meters = np.sqrt(
                (centroid.x - center_x) ** 2 + (centroid.y - center_y) ** 2
            )
            dist_km = max(dist_meters / 1000.0, 1e-6)

            return (1 / (dist_km * sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(dist_km) - mu) ** 2) / (2 * sigma ** 2)
            )

        if not master_features_gdf.empty:
            distance_prob = master_features_gdf.geometry.apply(
                calculate_distance_probability,
                args=(center_x_world, center_y_world, mu, sigma),
            )

            def get_area(geom):
                if isinstance(geom, shapely.Polygon):
                    return geom.area
                if isinstance(geom, LineString):
                    return geom.buffer(15).area
                return 0

            master_features_gdf["area"] = master_features_gdf.geometry.apply(get_area)
            area_influence = master_features_gdf["area"] * master_features_gdf["feature_type"].map(
                FEATURE_PROBABILITIES)

            combined_probability = area_influence * distance_prob

            total_prob_sum = combined_probability.sum()
            if total_prob_sum > 0:
                master_features_gdf["area_probability"] = (
                        combined_probability / total_prob_sum
                )
            else:
                master_features_gdf["area_probability"] = 0

        geojson_path = Path(output_directory) / "features.geojson"
        master_features_gdf.to_crs("EPSG:4326", inplace=True)
        geojson_dict = master_features_gdf.__geo_interface__

        geojson_dict["environment_type"] = environment_type
        geojson_dict["climate"] = environment_climate
        geojson_dict["center_point"] = center_point
        geojson_dict["meter_per_bin"] = meter_per_bin
        geojson_dict["bounds"] = [
            master_env.minx,
            master_env.miny,
            master_env.maxx,
            master_env.maxy,
        ]

        with geojson_path.open("w") as f:
            json.dump(geojson_dict, f)
        log.info(f"Serialized {len(master_features_gdf)} topological structures to {geojson_path}")

        heatmap_path = Path(output_directory) / "heatmap.npy"
        np.save(heatmap_path, final_probability_map)

        log.info(
            f"Serialized final probability matrix (Dimensions: {final_probability_map.shape}) to {heatmap_path}"
        )

        log.info("Constructing explicit topographical hazard matrix (Beta Array)...")
        risk_map = np.ones_like(final_probability_map)

        for key, individual_heatmap in master_env.heatmaps.items():
            if individual_heatmap is not None:
                risk_val = self.risk_scores.get(key, 1.0)
                feature_mask = individual_heatmap > 0
                risk_map[feature_mask] = np.maximum(risk_map[feature_mask], risk_val)

        risk_map_path = os.path.join(output_directory, "risk_map.npy")
        np.save(risk_map_path, risk_map)
        log.info(f"Serialized localized hazard matrix (Maximum Beta: {np.max(risk_map)}) to {risk_map_path}")

        log.info("--- Explicit geometric serialization successfully concluded. ---")
