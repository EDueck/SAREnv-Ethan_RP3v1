# examples/07_export_results_visuals.py
"""
Orchestrates the batch generation of topological visualizations and operational 
flight trajectories. Serializes spatial matrices into standardized PDF figures 
and exports continuous vector geometries into GIS-compatible GeoJSON formats.
"""
import os
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

import sarenv
from sarenv.core.loading import DatasetLoader
from sarenv.utils.plot import visualize_features, visualize_heatmap, visualize_risk_map, plot_heatmap
from sarenv.analytics.paths import generate_greedy_path
from sarenv.utils.geo import get_utm_epsg

log = sarenv.get_logger()

def main():
    """
    Executes the comprehensive spatial visualization and topological serialization sequence
    across all established topographical hazard biases (\u03b2).
    """
    log.info("Initiating comprehensive visual rendering and GeoJSON export protocol.")
    
    base_dir = Path(os.getcwd())
    dataset_base = base_dir / "sarenv_dataset"
    output_base = base_dir / "results_visuals"
    output_base.mkdir(exist_ok=True)

    # Defines the discrete topographical hazard bias (\u03b2) matrix for trajectory evaluation
    biases_to_test = [
        0.0, 0.025, 0.05, 0.075,
        0.1, 0.125, 0.15, 0.175,
        0.2, 0.225, 0.25, 0.275,
        0.3, 0.325, 0.35, 0.375,
        0.4, 0.425, 0.45, 0.475,
        0.5, 0.525, 0.55, 0.575,
        0.6, 0.625, 0.65, 0.675,
        0.7, 0.725, 0.75, 0.775,
        0.8, 0.825, 0.85, 0.875,
        0.9, 0.925, 0.95, 0.975,
        1.0
    ] 

    for map_id in range(1, 6):
        map_dataset_dir = dataset_base / str(map_id)
        map_output_dir = output_base / f"Map_{map_id}"
        map_output_dir.mkdir(exist_ok=True)

        log.info(f"--- Commencing rendering sequence for Topography {map_id} ---")

        # Deserializes the comprehensive 'xlarge' topological dataset for spatial rendering
        loader = DatasetLoader(str(map_dataset_dir))
        item = loader.load_environment("xlarge")

        if not item:
            log.error(f"Execution failure: Topography {map_id} uninitialized. Bypassing sequence.")
            continue

        # Isolates the active working directory to prevent file namespace collisions during sequential rendering
        os.chdir(map_output_dir)

        log.info("Rendering foundational topological, probability, and hazard matrices...")
        # Executes headless rendering to facilitate uninterrupted batch processing
        visualize_features(item, plot_basemap=False, plot_inset=True, plot_show=False)
        visualize_heatmap(item, plot_basemap=False, plot_inset=True, plot_show=False)
        visualize_risk_map(item, plot_basemap=False, plot_inset=True, plot_show=False)

        # Derives the planar Coordinate Reference System (CRS) for accurate geometric distance calculations
        data_crs = get_utm_epsg(item.center_point[0], item.center_point[1])
        center_proj = gpd.GeoDataFrame(
            geometry=[Point(item.center_point)], crs="EPSG:4326"
        ).to_crs(data_crs).geometry.iloc[0]

        for bias in biases_to_test:
            log.info(f"Resolving continuous spatial trajectory for Topographical Beta (\u03b2) {bias}...")

            # Executes an isolated deterministic heuristic evaluation to capture continuous flight geometries
            paths = generate_greedy_path(
                center_x=center_proj.x,
                center_y=center_proj.y,
                num_drones=3, 
                probability_map=item.heatmap,
                bounds=item.bounds,
                max_radius=item.radius_km * 1000,
                risk_map=item.risk_map,
                risk_bias=bias,
                fov_deg=45.0,      
                altitude=80.0,     
                budget=50000       
            )

            pdf_filename = f"travel_history_bias_{bias}.pdf"
            plot_heatmap(
                item=item,
                generated_paths=paths,
                name=f"Map {map_id} - Risk-Aware Heuristic (Beta: {bias})",
                x_min=item.bounds[0],
                x_max=item.bounds[2],
                y_min=item.bounds[1],
                y_max=item.bounds[3],
                output_file=pdf_filename
            )

            # Serializes the evaluated flight paths into discrete geospatial vectors
            geojson_filename = f"flight_path_bias_{bias}.geojson"
            gdf = gpd.GeoDataFrame(
                {'drone_id': range(len(paths)), 'bias': [bias]*len(paths), 'map_id': [map_id]*len(paths)},
                geometry=paths,
                crs=data_crs
            )
            
            # Transforms planar geometries into the global WGS84 projection standard (EPSG:4326) 
            # to ensure strict compatibility with external GIS applications.
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            gdf_wgs84.to_file(geojson_filename, driver="GeoJSON")

        # Restores the original execution directory prior to the next topographical iteration
        os.chdir(base_dir)

    log.info("--- Global Visualization and GeoJSON Serialization Protocol Successfully Concluded ---")

if __name__ == "__main__":
    main()
