# sarenv/io/loader.py
"""
Provides data management frameworks for deserializing and dynamically cropping 
large-scale topographical environment data for targeted agent simulations.
"""

import json
import os
from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
import numpy as np
import shapely

from sarenv.utils.logging_setup import get_logger
from sarenv.utils.lost_person_behavior import get_environment_radius_by_size, get_available_sizes
log = get_logger()


@dataclass
class SARDatasetItem:
    """
    Data structure encapsulating a localized, geometrically cropped search environment.

    Attributes:
        size (str): Classification of the operational boundary scale (e.g., 'small', 'medium').
        center_point (tuple[float, float]): The operational Last Known Point (longitude, latitude).
        radius_km (float): The geographical perimeter distance utilized for coordinate cropping.
        bounds (tuple[float, float, float, float]): The projected Cartesian limits (minx, miny, maxx, maxy) of the active matrix.
        features (gpd.GeoDataFrame): Topological classification array.
        heatmap (np.ndarray): A 2D spatial array representing casualty location probability. Retains absolute un-normalized probability mass.
        environment_climate (str): The empirical climate classification utilized for log-normal variance.
        environment_type (str): The empirical terrain classification utilized for log-normal variance.
        risk_map (np.ndarray | None): A 2D spatial array representing topographical hazard values (Beta).
    """
    size: str
    center_point: tuple[float, float]
    radius_km: float
    bounds: Tuple[float, float, float, float]
    features: gpd.GeoDataFrame
    heatmap: np.ndarray
    environment_climate: str
    environment_type: str
    risk_map: np.ndarray | None = None


class DatasetLoader:
    """
    Manages the deserialization of comprehensive master environment files, executing dynamic 
    geometric cropping and matrix masking to allocate localized terrain data on-demand.
    """

    def __init__(self, dataset_directory: str):
        """
        Initialises the dynamic data allocation framework.

        Args:
            dataset_directory (str): Absolute or relative path to the serialized master datasets.
        """
        if not os.path.isdir(dataset_directory):
            log.error(f"Dataset directory mapping failed at path: {dataset_directory}")
            raise FileNotFoundError(
                f"The specified directory is inaccessible or does not exist: {dataset_directory}"
            )

        self.dataset_directory = dataset_directory
        self.master_features_path = os.path.join(
            self.dataset_directory, "features.geojson"
        )
        self.master_heatmap_path = os.path.join(self.dataset_directory, "heatmap.npy")
        self.master_risk_map_path = os.path.join(self.dataset_directory, "risk_map.npy")

        if not os.path.exists(self.master_features_path):
            raise FileNotFoundError(
                f"Master topology JSON absent: {self.master_features_path}"
            )
        if not os.path.exists(self.master_heatmap_path):
            raise FileNotFoundError(
                f"Master probability matrix absent: {self.master_heatmap_path}"
            )

        # Volatile cache allocation for on-demand memory management
        self._master_features_gdf = None
        self._master_features_gdf_proj = None
        self._master_probability_map = None
        self._center_point = None
        self._projected_crs = None
        self._meter_per_bin = None
        self._bounds = None
        self._climate = None
        self._environment_type = None
        self._master_risk_map = None

    def _get_utm_epsg(self, lon: float, lat: float) -> str:
        """Derives the required Universal Transverse Mercator (UTM) EPSG zone for planar projection."""
        zone = int((lon + 180) / 6) + 1
        return f"326{zone}" if lat >= 0 else f"327{zone}"

    def _load_master_data_if_needed(self):
        """Executes disk I/O to populate the volatile cache if active arrays are uninitialized."""
        if self._master_probability_map is not None:
            return

        log.info("Populating memory cache from serialized master data...")
        try:
            with open(self.master_features_path, "r") as f:
                geojson_data = json.load(f)

            self._center_point = tuple(geojson_data["center_point"])
            self._meter_per_bin = geojson_data["meter_per_bin"]
            self._bounds = tuple(geojson_data["bounds"])
            self._climate = geojson_data["climate"]
            self._environment_type = geojson_data["environment_type"]
            
            log.info(
                f"Metadata extracted: LKP={self._center_point}, Matrix Resolution={self._meter_per_bin} m/bin"
            )

            self._projected_crs = f"EPSG:{self._get_utm_epsg(self._center_point[0], self._center_point[1])}"

            self._master_features_gdf = gpd.GeoDataFrame.from_features(
                geojson_data["features"], crs="EPSG:4326"
            )
            self._master_features_gdf_proj = self._master_features_gdf.to_crs(
                self._projected_crs
            )
            
            self._master_probability_map = np.load(self.master_heatmap_path)
            log.info(
                f"Master probability array populated. Dimensions: {self._master_probability_map.shape}"
            )

            if os.path.exists(self.master_risk_map_path):
                self._master_risk_map = np.load(self.master_risk_map_path)
                log.info(f"Master topographical hazard array populated. Dimensions: {self._master_risk_map.shape}")
            else:
                self._master_risk_map = None
                log.warning("Hazard array absent. Environmental risk biases (\u03b2) disabled for this sequence.")

        except KeyError as e:
            log.error(
                f"Structural fault in {self.master_features_path}: Key '{e}' missing. JSON may be corrupt."
            )
            raise
        except Exception as e:
            log.error(
                f"Catastrophic failure during master deserialization: {e}",
                exc_info=True,
            )
            raise


    def _world_to_image_for_master(
        self, x_world: np.ndarray, y_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Translates continuous Cartesian coordinates into discrete matrix indices."""
        master_minx, master_miny, _, _ = self._bounds
        x_img = (x_world - master_minx) / self._meter_per_bin
        y_img = (y_world - master_miny) / self._meter_per_bin
        return x_img.astype(int), y_img.astype(int)

    def load_environment(self, size: str) -> SARDatasetItem | None:
        """
        Dynamically crops the master operational terrain to satisfy a specific empirical radius constraint.
        Masks extraneous matrix indices to enforce a strict circular search perimeter.

        Args:
            size (str): Classification of the required operational scale.

        Returns:
            SARDatasetItem | None: A localized data structure isolating the requested operational zone.
        """
        log.info(f"Deriving localized data structure for operational constraint: '{size}'")
        self._load_master_data_if_needed()

        if self._master_probability_map is None:
            log.error(
                "Cropping sequence aborted: Master probability array failed to initialize."
            )
            return None

        radius_km = get_environment_radius_by_size(
            self._environment_type, self._climate, size
        )

        clipping_point_wgs84 = gpd.GeoDataFrame(
            geometry=[shapely.Point(self._center_point)], crs="EPSG:4326"
        )
        clipping_point_proj = clipping_point_wgs84.to_crs(self._projected_crs)
        clipping_circle_proj = clipping_point_proj.buffer(radius_km * 1000).iloc[0]
        clipped_bounds = clipping_circle_proj.bounds

        clipped_features_proj = gpd.clip(
            self._master_features_gdf_proj, clipping_circle_proj
        )
        log.info(
            f"Topological features cropped to {len(clipped_features_proj)} entities for boundary '{size}'."
        )

        min_x_w, min_y_w, max_x_w, max_y_w = clipped_bounds
        img_min_x, img_min_y = self._world_to_image_for_master(
            np.array([min_x_w]), np.array([min_y_w])
        )
        img_max_x, img_max_y = self._world_to_image_for_master(
            np.array([max_x_w]), np.array([max_y_w])
        )

        h, w = self._master_probability_map.shape
        img_min_y, img_max_y = np.clip([img_min_y[0], img_max_y[0]], 0, h)
        img_min_x, img_max_x = np.clip([img_min_x[0], img_max_x[0]], 0, w)

        cropped_map = self._master_probability_map[
            img_min_y:img_max_y, img_min_x:img_max_x
        ]

        # Generates a strict circular mask to eliminate the corner indices of the bounding square
        yy, xx = np.mgrid[img_min_y:img_max_y, img_min_x:img_max_x]
        center_x_master_px = (
            clipping_point_proj.geometry.x.values[0] - self._bounds[0]
        ) / self._meter_per_bin
        center_y_master_px = (
            clipping_point_proj.geometry.y.values[0] - self._bounds[1]
        ) / self._meter_per_bin
        radius_px = (radius_km * 1000) / self._meter_per_bin
        mask = (
            (xx - center_x_master_px) ** 2 + (yy - center_y_master_px) ** 2
        ) <= radius_px**2

        # Eliminates data outside the radial threshold without re-normalizing the core distribution
        final_heatmap = np.where(mask, cropped_map, 0)

        final_risk_map = None
        if self._master_risk_map is not None:
            cropped_risk = self._master_risk_map[
                img_min_y:img_max_y, img_min_x:img_max_x
            ]
            
            # Non-operational terrain defaults to a baseline beta of 1.0 rather than 0
            final_risk_map = np.where(mask, cropped_risk, 1.0)


        log.info(
            f"Cropped probability matrix for constraint '{size}' retains {np.sum(final_heatmap):.4f} total density mass."
        )

        return SARDatasetItem(
            size=size,
            center_point=self._center_point,
            radius_km=radius_km,
            bounds=clipped_bounds,
            features=clipped_features_proj,
            heatmap=final_heatmap,
            environment_climate=self._climate,
            environment_type=self._environment_type,
            risk_map=final_risk_map
        )

    def load_all(self) -> dict[str, SARDatasetItem]:
        """
        Executes sequential cropping sequences across all empirically defined search perimeters.

        Returns:
            dict[str, SARDatasetItem]: Dictionary mapping size classifications to their localized data structures.
        """
        log.info("Initiating sequential matrix generation for all established operational boundaries.")
        all_data = {}
        for size in get_available_sizes():
            item = self.load_environment(size)
            if item:
                all_data[size] = item
        return all_data
