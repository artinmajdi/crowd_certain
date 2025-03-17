"""
Example: Visualizing Simulation Results

This example demonstrates how to visualize simulation results using matplotlib and seaborn.
"""

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Updated imports to use the new module structure
from crowd_certain.utilities.components.aim1_3 import AIM1_3
from crowd_certain.utilities.config.settings import Settings
from crowd_certain.utilities.config.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques


def run_simulation_for_visualization():
    """Run a simulation to generate results for visualization."""
    print("Running simulation to generate results for visualization...")

    # Configure settings
    config = Settings()
    config.dataset.dataset_name = DatasetNames.IONOSPHERE
    config.dataset.num_workers = 10
    config.dataset.distance_from_mean = 0.2
    config.dataset.distance_from_mean_std = 0.1

    # Set techniques
    config.techniques.uncertainty = UncertaintyTechniques.STD
    config.techniques.consistency = ConsistencyTechniques.WEIGHTED_MEAN

    # Run simulation
    try:
        results = AIM1_3.calculate_one_dataset(config=config)
        print("Simulation completed successfully!")
        return results
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        return None


def visualize_worker_weights(results, save_path=None):
    """Visualize worker weights as a heatmap."""
    print("Visualizing worker weights...")

    # Get the first result
    first_nl_key = list(results.outputs.keys())[0]
    first_result = results.outputs[first_nl_key][0]

    # Extract worker weights
    worker_weights = first_result.weights

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(worker_weights, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Worker Weights Heatmap")
    plt.xlabel("Workers")
    plt.ylabel("Samples")

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_confidence_distribution(results, save_path=None):
    """Visualize the distribution of confidence scores."""
    print("Visualizing confidence score distribution...")

    # Get the first result
    first_nl_key = list(results.outputs.keys())[0]
    first_result = results.outputs[first_nl_key][0]

    # Extract confidence scores
    confidence_scores = first_result.proposed.confidence_scores

    # Convert to a list of values
    confidence_values = []
    for key, scores in confidence_scores.items():
        confidence_values.extend(scores)

    # Create a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(confidence_values, kde=True, bins=20)
    plt.title("Distribution of Confidence Scores")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_metrics_comparison(results, save_path=None):
    """Visualize a comparison of metrics between proposed and benchmark methods."""
    print("Visualizing metrics comparison...")

    # Get the first result
    first_nl_key = list(results.outputs.keys())[0]
    first_result = results.outputs[first_nl_key][0]

    # Extract metrics
    proposed_metrics = first_result.proposed.metrics
    benchmark_metrics = first_result.benchmark.metrics

    # Create a DataFrame for comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    data = {
        'Metric': metrics * 2,
        'Value': [
            proposed_metrics.accuracy, proposed_metrics.precision,
            proposed_metrics.recall, proposed_metrics.f1,
            benchmark_metrics.accuracy, benchmark_metrics.precision,
            benchmark_metrics.recall, benchmark_metrics.f1
        ],
        'Method': ['Proposed'] * 4 + ['Benchmark'] * 4
    }
    df = pd.DataFrame(data)

    # Create a grouped bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Value', hue='Method', data=df)
    plt.title("Performance Metrics Comparison")
    plt.ylim(0, 1)

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Comparison chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_worker_strength(results, save_path=None):
    """Visualize worker strength as a bar chart."""
    print("Visualizing worker strength...")

    # Get the first result
    first_nl_key = list(results.outputs.keys())[0]
    first_result = results.outputs[first_nl_key][0]

    # Extract worker strength
    worker_strength = first_result.workers_strength

    # Create a bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(range(len(worker_strength))), y=worker_strength)
    plt.title("Worker Strength")
    plt.xlabel("Worker ID")
    plt.ylabel("Strength")

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Worker strength chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def run_example():
    """Run the visualization example."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "visualization_example"
    os.makedirs(output_dir, exist_ok=True)

    # Run simulation or load existing results
    results_path = output_dir / "simulation_results.pkl"

    if os.path.exists(results_path):
        print(f"Loading existing results from {results_path}")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = run_simulation_for_visualization()
        if results:
            print(f"Saving results to {results_path}")
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

    if not results:
        print("No results available for visualization.")
        return

    # Create visualizations
    visualize_worker_weights(results, save_path=output_dir / "worker_weights.png")
    visualize_confidence_distribution(results, save_path=output_dir / "confidence_distribution.png")
    visualize_metrics_comparison(results, save_path=output_dir / "metrics_comparison.png")
    visualize_worker_strength(results, save_path=output_dir / "worker_strength.png")

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    run_example()
