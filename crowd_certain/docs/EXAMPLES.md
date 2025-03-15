# Examples

This document provides an overview of the example scripts available in the `crowd_certain/examples/` directory. These examples are designed to help you understand the library's capabilities and how to integrate it into your own projects.

## Available Examples

### 1. Running Simulations

- **[run_single_dataset_example.py](../examples/run_single_dataset_example.py)**: Demonstrates how to run a simulation on a single dataset with specific configuration parameters.
- **[run_all_datasets_example.py](../examples/run_all_datasets_example.py)**: Shows how to run simulations on multiple datasets, either all available datasets or a specific subset.

### 2. Saving and Loading Results

- **[save_load_results_example.py](../examples/save_load_results_example.py)**: Illustrates how to save simulation results to disk and load them later for analysis.

### 3. Customizing Techniques

- **[custom_techniques_example.py](../examples/custom_techniques_example.py)**: Demonstrates how to use different uncertainty and consistency techniques, and how to compare their performance.

### 4. Visualization

- **[visualize_results_example.py](../examples/visualize_results_example.py)**: Shows how to create various visualizations from simulation results, including worker weights, confidence distributions, and performance metrics.

### 5. Dashboard

- **[run_dashboard_example.py](../examples/run_dashboard_example.py)**: Shows how to programmatically launch the Streamlit dashboard for interactive exploration of results.

## How to Run Examples

You can run any example from the command line:

```bash
# From the project root directory
python -m crowd_certain.examples.run_single_dataset_example

# Or navigate to the examples directory first
cd crowd_certain/examples
python run_single_dataset_example.py
```

## Example Details

### Running Simulations

The simulation examples demonstrate how to configure and run simulations using the Crowd-Certain library:

```python
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.params import DatasetNames, ReadMode
from crowd_certain.utilities.settings import Settings

# Create configuration
config = Settings()
config.dataset.dataset_name = DatasetNames.IONOSPHERE
config.dataset.read_mode = ReadMode.AUTO

# Run simulation
results = AIM1_3.calculate_one_dataset(config=config)
```

### Saving and Loading Results

The save/load example shows how to persist simulation results for later analysis:

```python
import pickle
from pathlib import Path

# Save results
output_dir = Path("crowd_certain/results/example")
output_dir.mkdir(parents=True, exist_ok=True)
pickle_path = output_dir / "results.pkl"

with open(pickle_path, "wb") as f:
    pickle.dump(results, f)

# Load results
with open(pickle_path, "rb") as f:
    loaded_results = pickle.load(f)
```

### Customizing Techniques

The techniques example demonstrates how to use different uncertainty and consistency techniques:

```python
from crowd_certain.utilities.params import UncertaintyTechniques, ConsistencyTechniques

# Configure techniques
config.technique.uncertainty_techniques = [
    UncertaintyTechniques.STD,
    UncertaintyTechniques.ENTROPY
]
config.technique.consistency_techniques = [
    ConsistencyTechniques.ONE_MINUS_UNCERTAINTY,
    ConsistencyTechniques.WEIGHTED_MEAN
]
```

### Visualization

The visualization example shows how to create various plots from simulation results:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize worker weights
plt.figure(figsize=(10, 8))
sns.heatmap(worker_weights, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Worker Weights Heatmap")
plt.xlabel("Workers")
plt.ylabel("Samples")
plt.savefig("worker_weights.png")
```

### Dashboard

The dashboard example demonstrates how to programmatically launch the Streamlit dashboard:

```python
import streamlit.web.cli as stcli
from pathlib import Path

# Get the path to the dashboard.py file
dashboard_path = Path("crowd_certain/utilities/dashboard.py")

# Run the dashboard
sys.argv = ["streamlit", "run", str(dashboard_path)]
stcli.main()
```

## Creating Your Own Examples

Feel free to modify these examples or create your own based on them. The examples are designed to be simple and easy to understand, focusing on specific aspects of the library.

For more detailed information about the library's API, please refer to the [API Reference](API.md).
