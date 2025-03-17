# Crowd-Certain Dashboard

[← Back to Main README](../../README.md) | [← Back to Documentation Index](index.md) | [Next: API Reference →](API.md)

---

This is a Streamlit-based dashboard for the Crowd-Certain library, which provides tools for crowd-sourced label aggregation with uncertainty estimation and confidence scoring.

## Features

- Run simulations with different datasets and parameters
- Visualize worker strength and weight relationships
- Explore confidence scores and performance metrics
- Compare different aggregation methods

## Installation

### Option 1: Using the installation script (recommended)

Run the installation script from the project root:

```bash
# On Unix/Linux/macOS:
./crowd_certain/scripts/install.sh

# On Windows:
crowd_certain\scripts\install.bat
```

### Option 2: Manual installation

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the required dependencies:

   ```bash
   pip install -r crowd_certain/config/requirements.txt
   ```

## Running the Dashboard

### Option 1: Using the run script (recommended)

Run the dashboard script from the project root:

```bash
# On Unix/Linux/macOS:
./crowd_certain/scripts/run_dashboard.sh

# On Windows:
crowd_certain\scripts\run_dashboard.bat
```

### Option 2: Manual execution

1. Activate your environment if not already activated:

   ```bash
   source ./crowd_certain/config/activate_env.sh  # On Windows: crowd_certain\config\activate.bat
   ```

2. Run the dashboard:

   ```bash
   python -m crowd_certain.utilities.dashboard
   ```

## Configuration

The dashboard uses the configuration file located at `crowd_certain/config/config.json`. You can modify this file directly or use the dashboard interface to update settings.

### Default Configuration

A default configuration file is provided at `crowd_certain/config/config_default.json`. You can revert to this configuration using the "Revert to Default Config" button in the dashboard.

## Usage

1. Configure the simulation parameters in the sidebar:
   - Select a dataset
   - Set the number of workers
   - Adjust worker quality range
   - Choose uncertainty and consistency techniques
   - Set the number of random seeds

2. Click "Run Simulation" to start the simulation

3. Explore the results in the different tabs:
   - Simulation Results: View performance metrics, worker strength vs weight relationship, and confidence scores
   - Worker Analysis: Examine worker strength distribution and weight matrices
   - About: Learn more about the Crowd-Certain library

## Screenshots

![Simulation Results](https://via.placeholder.com/800x400?text=Simulation+Results)
![Worker Analysis](https://via.placeholder.com/800x400?text=Worker+Analysis)

---

[← Back to Main README](../../README.md) | [← Back to Documentation Index](index.md) | [Next: API Reference →](API.md)
