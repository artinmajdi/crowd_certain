"""
Example: Using custom uncertainty and consistency techniques

This example demonstrates how to use different combinations of uncertainty and consistency techniques.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from crowd_certain.utilities import AIM1_3
from crowd_certain.utilities.config.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques
from crowd_certain.utilities.config.settings import Settings, OutputModes


def run_with_all_techniques():
    """Run a simulation using all available uncertainty and consistency techniques."""
    print("Running simulation with all uncertainty and consistency techniques...")

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
            # Use all available uncertainty techniques
            uncertainty_techniques=list(UncertaintyTechniques),
            # Use all available consistency techniques
            consistency_techniques=list(ConsistencyTechniques),
        ),
        output=dict(
            mode=OutputModes.CALCULATE,
            save=False,
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

        return results

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_uncertainty_techniques():
    """Compare different uncertainty techniques."""
    print("Comparing different uncertainty techniques...")

    # Dictionary to store results for each technique
    results_by_technique = {}

    # Run simulation for each uncertainty technique
    for technique in UncertaintyTechniques:
        print(f"\nRunning simulation with uncertainty technique: {technique.value}")

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
                # Use only one uncertainty technique
                uncertainty_techniques=[technique],
                # Use a fixed consistency technique
                consistency_techniques=[ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
            ),
            output=dict(
                mode=OutputModes.CALCULATE,
                save=False,
            )
        )

        # Run simulation
        try:
            results = AIM1_3.calculate_one_dataset(config=config)

            # Store accuracy for this technique
            first_nl_key = list(results.outputs.keys())[0]
            first_result = results.outputs[first_nl_key][0]
            accuracy = first_result.proposed.metrics.loc["accuracy"][0]

            results_by_technique[technique.value] = accuracy
            print(f"Accuracy: {accuracy}")

        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            results_by_technique[technique.value] = None

    # Plot comparison
    if results_by_technique:
        # Filter out None values
        results_by_technique = {k: v for k, v in results_by_technique.items() if v is not None}

        # Create DataFrame for plotting
        df = pd.DataFrame(list(results_by_technique.items()), columns=["Technique", "Accuracy"])

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Technique", y="Accuracy", data=df)
        plt.title("Comparison of Uncertainty Techniques")
        plt.xlabel("Uncertainty Technique")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        output_dir = Path("crowd_certain/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "uncertainty_techniques_comparison.png")

        print(f"\nComparison plot saved to: {output_dir / 'uncertainty_techniques_comparison.png'}")

    return results_by_technique


def compare_consistency_techniques():
    """Compare different consistency techniques."""
    print("Comparing different consistency techniques...")

    # Dictionary to store results for each technique
    results_by_technique = {}

    # Run simulation for each consistency technique
    for technique in ConsistencyTechniques:
        print(f"\nRunning simulation with consistency technique: {technique.value}")

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
                # Use a fixed uncertainty technique
                uncertainty_techniques=[UncertaintyTechniques.STD],
                # Use only one consistency technique
                consistency_techniques=[technique],
            ),
            output=dict(
                mode=OutputModes.CALCULATE,
                save=False,
            )
        )

        # Run simulation
        try:
            results = AIM1_3.calculate_one_dataset(config=config)

            # Store accuracy for this technique
            first_nl_key = list(results.outputs.keys())[0]
            first_result = results.outputs[first_nl_key][0]
            accuracy = first_result.proposed.metrics.loc["accuracy"][0]

            results_by_technique[technique.value] = accuracy
            print(f"Accuracy: {accuracy}")

        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            results_by_technique[technique.value] = None

    # Plot comparison
    if results_by_technique:
        # Filter out None values
        results_by_technique = {k: v for k, v in results_by_technique.items() if v is not None}

        # Create DataFrame for plotting
        df = pd.DataFrame(list(results_by_technique.items()), columns=["Technique", "Accuracy"])

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Technique", y="Accuracy", data=df)
        plt.title("Comparison of Consistency Techniques")
        plt.xlabel("Consistency Technique")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        output_dir = Path("crowd_certain/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "consistency_techniques_comparison.png")

        print(f"\nComparison plot saved to: {output_dir / 'consistency_techniques_comparison.png'}")

    return results_by_technique


if __name__ == "__main__":
    # Uncomment one of the following lines to run the desired example
    # run_with_all_techniques()
    compare_uncertainty_techniques()
    # compare_consistency_techniques()
