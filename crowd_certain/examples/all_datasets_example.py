"""
Example: Running a simulation on all datasets

This example demonstrates how to run a simulation on all available datasets.
"""

from pathlib import Path
from crowd_certain.utilities.components.aim1_3 import AIM1_3
from crowd_certain.utilities.config.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques
from crowd_certain.utilities.config.settings import Settings, OutputModes


def run_all_datasets_example():
    """Run a simulation on all available datasets."""
    print("Running simulation on all datasets...")

    # Create configuration
    config = Settings(
        dataset=dict(
            # Include all available datasets
            datasetNames=list(DatasetNames),
            path_all_datasets=Path("crowd_certain/datasets")  # Path to datasets
        ),
        simulation=dict(
            n_workers_min_max=[3, 5],  # Min and max number of workers (using a smaller range for faster execution)
            low_dis=0.4,  # Minimum worker quality
            high_dis=1.0,  # Maximum worker quality
            num_seeds=2,  # Number of random seeds (using fewer seeds for faster execution)
        ),
        technique=dict(
            # Using only standard deviation for uncertainty to speed up execution
            uncertainty_techniques=[UncertaintyTechniques.STD],
            # Using only one minus uncertainty for consistency to speed up execution
            consistency_techniques=[ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
        ),
        output=dict(
            mode=OutputModes.CALCULATE,  # Calculate results but don't save
            save=False,  # Don't save results
        )
    )

    # Run simulation
    try:
        results = AIM1_3.calculate_all_datasets(config=config)
        print("Simulation completed successfully for all datasets!")

        # Display summary of results for each dataset
        for dataset_name, dataset_results in results.items():
            print(f"\n\n=== Results for dataset: {dataset_name} ===")

            first_nl_key = list(dataset_results.outputs.keys())[0]
            first_result = dataset_results.outputs[first_nl_key][0]

            print("\nAccuracy:")
            print(first_result.proposed.metrics.loc["accuracy"])

            print("\nF1 Score:")
            if "f1" in first_result.proposed.metrics.index:
                print(first_result.proposed.metrics.loc["f1"])
            else:
                print("F1 score not available")

        return results

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_specific_datasets_example():
    """Run a simulation on a specific subset of datasets."""
    print("Running simulation on a subset of datasets (Ionosphere, Chess, and Mushroom)...")

    # Create configuration
    config = Settings(
        dataset=dict(
            # Include only specific datasets
            datasetNames=[
                DatasetNames.IONOSPHERE,
                DatasetNames.CHESS,
                DatasetNames.MUSHROOM
            ],
            path_all_datasets=Path("crowd_certain/datasets")  # Path to datasets
        ),
        simulation=dict(
            n_workers_min_max=[3, 8],  # Min and max number of workers
            low_dis=0.4,  # Minimum worker quality
            high_dis=1.0,  # Maximum worker quality
            num_seeds=3,  # Number of random seeds
        ),
        technique=dict(
            uncertainty_techniques=[UncertaintyTechniques.STD],
            consistency_techniques=[ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
        ),
        output=dict(
            mode=OutputModes.CALCULATE,  # Calculate results but don't save
            save=False,  # Don't save results
        )
    )

    # Run simulation
    try:
        results = AIM1_3.calculate_all_datasets(config=config)
        print("Simulation completed successfully for selected datasets!")

        # Display summary of results for each dataset
        for dataset_name, dataset_results in results.items():
            print(f"\n\n=== Results for dataset: {dataset_name} ===")

            first_nl_key = list(dataset_results.outputs.keys())[0]
            first_result = dataset_results.outputs[first_nl_key][0]

            print("\nProposed Methods Metrics:")
            print(first_result.proposed.metrics)

        return results

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Uncomment one of the following lines to run the desired example
    # run_all_datasets_example()  # This may take a long time to run
    run_specific_datasets_example()  # This runs faster with only 3 datasets
