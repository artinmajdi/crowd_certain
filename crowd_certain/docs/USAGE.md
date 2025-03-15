# Usage Guide

[← Back to Main README](../README.md) | [← Back to Documentation Index](index.md) | [← Previous: Installation Guide](INSTALLATION.md) | [Next: API Reference →](API.md)

---

This guide provides examples of how to use the Crowd-Certain library for crowd-sourced label aggregation with uncertainty estimation.

## Basic Usage

### Importing the Library

```python
import crowd_certain
from crowd_certain.utilities.utils import AIM1_3, ResultType, WeightType, Result2Type
```

### Loading Data

Crowd-Certain provides utilities for loading datasets:

```python
from crowd_certain.utilities import load_data
from crowd_certain.utilities.utils import Settings, DatasetNames

# Configure settings
config = Settings()
config.dataset.dataset_name = DatasetNames.IONOSPHERE

# Load data
data, feature_columns = load_data.load_dataset(config=config)
```

### Creating an Instance

```python
# Create an AIM1_3 instance
aim1_3 = AIM1_3(data=data, feature_columns=feature_columns, config=config)
```

### Running the Core Measurements

```python
# Run core measurements for a specific number of workers and seed
n_workers = 5
seed = 0
results = AIM1_3.core_measurements(
    data=data,
    n_workers=n_workers,
    config=config,
    feature_columns=feature_columns,
    seed=seed
)

# Access the results
proposed_results = results.proposed
benchmark_results = results.benchmark
weights = results.weight
workers_strength = results.workers_strength
```

## Advanced Usage

### Calculating Uncertainties

```python
# Create a DataFrame of worker predictions
import pandas as pd
import numpy as np

# Example predictions from 3 workers for 10 items
predictions = pd.DataFrame({
    'worker_0': np.random.randint(0, 2, 10),
    'worker_1': np.random.randint(0, 2, 10),
    'worker_2': np.random.randint(0, 2, 10)
})

# Calculate uncertainties
uncertainties = aim1_3.calculate_uncertainties(predictions)
print(uncertainties)
```

### Converting Uncertainties to Consistency

```python
# Convert uncertainties to consistency scores
consistency = aim1_3.calculate_consistency(uncertainties)
print(consistency)
```

### Calculating Worker Weights

```python
# Calculate weights for the proposed technique
weights_proposed = aim1_3.aim1_3_measuring_proposed_weights(
    preds=predictions,
    uncertainties=uncertainties
)
print(weights_proposed)
```

### Calculating Confidence Scores

```python
from crowd_certain.utilities.utils import AIM1_3

# Binary worker responses (True/False)
delta = predictions > 0.5

# Worker weights (example)
weights = pd.Series([0.4, 0.3, 0.3], index=predictions.columns)

# Calculate confidence scores
confidence_scores = AIM1_3.calculate_confidence_scores(
    delta=delta,
    w=weights,
    n_workers=3
)
print(confidence_scores)
```

### Evaluating Aggregation Methods

```python
# True labels (example)
true_labels = pd.Series(np.random.randint(0, 2, 10))

# Aggregated labels (example)
aggregated_labels = pd.Series(np.random.random(10))

# Calculate evaluation metrics
metrics = AIM1_3.get_AUC_ACC_F1(
    aggregated_labels=aggregated_labels,
    truth=true_labels
)
print(metrics)
```

## Working with Multiple Datasets

```python
# Calculate results for all datasets
results_all = AIM1_3.calculate_all_datasets(config=config)

# Access results for a specific dataset
ionosphere_results = results_all[DatasetNames.IONOSPHERE]
```

## Visualization

Crowd-Certain provides visualization tools through the `AIM1_3_Plot` class:

```python
from crowd_certain.utilities.utils import AIM1_3_Plot

# Example data for plotting
plot_data = pd.DataFrame({
    'ProposedTechnique1': [0.8, 0.85, 0.9, 0.92, 0.95],
    'ProposedTechnique2': [0.75, 0.8, 0.85, 0.88, 0.9],
    'Benchmark1': [0.7, 0.75, 0.8, 0.82, 0.85]
}, index=[3, 5, 7, 9, 11])  # Number of workers

# Create a plot
plotter = AIM1_3_Plot(plot_data)
plotter.plot(
    xlabel='Number of Workers',
    ylabel='Accuracy',
    title='Accuracy vs Number of Workers',
    legend=['Proposed 1', 'Proposed 2', 'Benchmark'],
    smooth=True
)
```

## Analysis of Results

For comprehensive analysis of results, you can use the `Aim1_3_Data_Analysis_Results` class:

```python
from crowd_certain.utilities.utils import Aim1_3_Data_Analysis_Results

# Create an analysis instance
analysis = Aim1_3_Data_Analysis_Results(config=config)

# Update with latest results
analysis.update()

# Generate figures
analysis.figure_weight_quality_relation()
analysis.figure_metrics_mean_over_seeds_per_dataset_per_worker()
analysis.figure_metrics_all_datasets_workers()
analysis.figure_F_heatmap()
```

## Benchmarking Against Other Techniques

Crowd-Certain includes implementations of several benchmark techniques:

```python
from crowd_certain.utilities.utils import BenchmarkTechniques, OtherBenchmarkNames
import pandas as pd

# Example crowd labels and ground truth
crowd_labels = {
    'worker_0': pd.Series([0, 1, 0, 1, 0]),
    'worker_1': pd.Series([1, 1, 0, 0, 0]),
    'worker_2': pd.Series([0, 1, 0, 1, 1])
}
ground_truth = {'truth': pd.Series([0, 1, 0, 1, 0])}

# Create a benchmark instance
benchmark = BenchmarkTechniques(crowd_labels=crowd_labels, ground_truth=ground_truth)

# Apply all benchmark techniques
results = benchmark.apply(true_labels=ground_truth, use_parallelization_benchmarks=False)
print(results)
```

## Working with Real-World Data

For real-world applications, you might want to:

1. Load your own data
2. Configure the simulation parameters
3. Run the analysis
4. Visualize the results

Here's an example workflow:

```python
import pandas as pd
import numpy as np
from crowd_certain.utilities.utils import AIM1_3, Settings, SimulationMethods, UncertaintyTechniques, ConsistencyTechniques

# 1. Prepare your data
train_data = pd.DataFrame({
    'feature1': np.random.random(100),
    'feature2': np.random.random(100),
    'true': np.random.randint(0, 2, 100)
})

test_data = pd.DataFrame({
    'feature1': np.random.random(50),
    'feature2': np.random.random(50),
    'true': np.random.randint(0, 2, 50)
})

data = {
    'train': train_data,
    'test': test_data
}

feature_columns = ['feature1', 'feature2']

# 2. Configure settings
config = Settings()
config.simulation.simulation_methods = SimulationMethods.RANDOM_STATES
config.simulation.num_simulations = 5
config.simulation.workers_list = [3, 5, 7]
config.simulation.num_seeds = 3
config.technique.uncertainty_techniques = [UncertaintyTechniques.STD, UncertaintyTechniques.ENTROPY]
config.technique.consistency_techniques = [ConsistencyTechniques.ONE_MINUS_UNCERTAINTY]

# 3. Create an instance and run analysis
aim1_3 = AIM1_3(data=data, feature_columns=feature_columns, config=config)
outputs = aim1_3.get_outputs()

# 4. Analyze worker strength-weight relationship
weight_strength_relation = aim1_3.worker_weight_strength_relation(seed=0, n_workers=5)
print(weight_strength_relation)
```

## Additional Resources

For more detailed examples and use cases, refer to the Jupyter notebooks in the `crowd_certain/notebooks/` directory.

---
