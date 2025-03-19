"""
Example: Running a simulation on a single dataset

This example demonstrates how to run a simulation on a single dataset with custom parameters.
"""

from pathlib import Path
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.parameters.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques
from crowd_certain.utilities.parameters.settings import Settings, OutputModes


def run_single_dataset_example():
    """Run a simulation on a single dataset with custom parameters."""
    print("Running simulation on the Ionosphere dataset...")

    # Create configuration
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,  # Specify the dataset
            datasetNames=[DatasetNames.IONOSPHERE],
            path_all_datasets=Path("crowd_certain/datasets")  # Path to datasets
        ),
        simulation=dict(
            n_workers_min_max=[3, 8],  # Min and max number of workers
            low_dis=0.4,  # Minimum worker quality
            high_dis=1.0,  # Maximum worker quality
            num_seeds=3,  # Number of random seeds
        ),
        technique=dict(
            # Specify uncertainty techniques
            uncertainty_techniques=[
                UncertaintyTechniques.STD,  # Standard deviation
                UncertaintyTechniques.ENTROPY  # Entropy
            ],
            # Specify consistency techniques
            consistency_techniques=[
                ConsistencyTechniques.ONE_MINUS_UNCERTAINTY,  # 1 - uncertainty
                ConsistencyTechniques.ONE_DIVIDED_BY_UNCERTAINTY  # 1 / uncertainty
            ],
        ),
        output=dict(
            mode=OutputModes.CALCULATE,  # Calculate results but don't save
            save=False,  # Don't save results
        )
    )

    # Run simulation
    try:
        results = AIM1_3.calculate_one_dataset(config=config)
        print("Simulation completed successfully!")

        # Access and display results
        first_nl_key = list(results.outputs.keys())[0]
        first_result = results.outputs[first_nl_key][0]

        print("\nProposed Methods Metrics:")
        print(first_result.proposed.metrics)

        print("\nBenchmark Methods Metrics:")
        print(first_result.benchmark.metrics)

        print("\nWorker Strength:")
        print(first_result.workers_strength)

        # Access confidence scores
        confidence_scores = first_result.proposed.confidence_scores
        if confidence_scores:
            print("\nConfidence Scores:")
            for method, scores in confidence_scores.items():
                print(f"\n{method} Confidence Scores:")
                print(scores.head())  # Show first few rows

        return results

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_single_dataset_example()
