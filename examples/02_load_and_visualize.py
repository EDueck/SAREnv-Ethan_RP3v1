"""
Execution script demonstrating the deserialization of operational terrain data,
followed by the spatial rendering of probability, topology, and hazard matrices.
"""
from sarenv import (
    DatasetLoader,
    get_logger,
    visualize_heatmap,
    visualize_features,
)
from sarenv.utils.plot import visualize_risk_map

log = get_logger()


def run_loading_example():
    """
    Executes the data management framework to allocate a localized environment constraint,
    subsequently triggering the visualization suite to render the operational arrays.
    """
    log.info("Initiating dataset deserialization and spatial rendering sequence.")

    dataset_dir = "sarenv_dataset/1"
    size_to_load = "medium"

    try:
        loader = DatasetLoader(dataset_directory=dataset_dir)
        log.info(f"Extracting spatial matrices for operational constraint: '{size_to_load}'")
        item = loader.load_environment(size_to_load)

        if item:
            # Executes the spatial visualization suite against the populated dataset
            visualize_heatmap(item, plot_basemap=False, plot_inset=True)
            visualize_features(item, plot_basemap=False, plot_inset=True, num_lost_persons=300)
            visualize_risk_map(item, plot_basemap=False, plot_inset=True)
        else:
            log.error(f"Execution failure: Spatial constraint '{size_to_load}' could not be resolved.")

    except FileNotFoundError:
        log.error(
            f"Directory mapping failed: Target dataset '{dataset_dir}' is inaccessible."
        )
        log.error(
            "Master topology serialization via DataGenerator.export_dataset() must precede this execution."
        )
    except Exception as e:
        log.error(f"Catastrophic failure during environment deserialization: {e}", exc_info=True)


if __name__ == "__main__":
    run_loading_example()
