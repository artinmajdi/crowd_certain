"""
Example: Saving and loading simulation results

This example demonstrates how to save simulation results to disk and load them later.
"""

import os
from pathlib import Path
import pickle
from datetime import datetime

from src.crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.parameters.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques
from crowd_certain.utilities.parameters.settings import Settings, OutputModes


def save_results_example():
    """Run a simulation and save the results to disk."""
    print("Running simulation and saving results...")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"crowd_certain/results/example_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Create configuration
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            datasetNames=[DatasetNames.IONOSPHERE],
            path_all_datasets=Path("crowd_certain/datasets")
        ),
        simulation=dict(
            n_workers_min_max=[3, 8],
            low_dis=0.4,
            high_dis=1.0,
            num_seeds=3,
        ),
        technique=dict(
            uncertainty_techniques=[UncertaintyTechniques.STD],
            consistency_techniques=[ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
        ),
        output=dict(
            mode=OutputModes.SAVE,  # Save results to disk
            save=True,  # Enable saving
            path=str(output_dir),  # Output directory
        )
    )

    # Run simulation
    try:
        results = AIM1_3.calculate_one_dataset(config=config)
        print("Simulation completed successfully!")

        # The results are automatically saved by the framework when using OutputModes.SAVE
        # But we can also manually save them using pickle

        # Save results using pickle
        pickle_path = output_dir / "results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Results manually saved to: {pickle_path}")

        return output_dir

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_results_example(results_dir):
    """Load simulation results from disk."""
    print(f"Loading results from: {results_dir}")

    # Load results using pickle
    pickle_path = results_dir / "results.pkl"

    if not pickle_path.exists():
        print(f"Error: Results file not found at {pickle_path}")
        return None

    try:
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)

        print("Results loaded successfully!")

        # Display some basic information from the loaded results
        first_nl_key = list(results.outputs.keys())[0]
        first_result = results.outputs[first_nl_key][0]

        print("\nProposed Methods Metrics:")
        print(first_result.proposed.metrics)

        return results

    except Exception as e:
        print(f"Error loading results: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_example():
    """Run the save and load example."""
    # First, run a simulation and save the results
    results_dir = save_results_example()

    if results_dir is None:
        print("Failed to save results. Exiting.")
        return

    print("\n" + "="*50 + "\n")

    # Then, load the results from disk
    loaded_results = load_results_example(results_dir)

    if loaded_results is None:
        print("Failed to load results. Exiting.")
        return

    print("\nSuccessfully demonstrated saving and loading results!")


if __name__ == "__main__":
    run_example()
