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

1. Make sure you have Python 3.10+ installed
2. Install the required dependencies:

```bash
# From the project root directory
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory, you can run:

```bash
# Option 1: Using the entry point script (recommended)
python crowd_certain/run_streamlit.py
# Or directly execute it
./crowd_certain/run_streamlit.py

# Option 2: Using the provided scripts
# On Unix/Linux/macOS
./crowd_certain/scripts/run_dashboard.sh

# On Windows
crowd_certain\scripts\run_dashboard.bat

# Option 3: Directly with Streamlit
streamlit run crowd_certain/utilities/dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

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
