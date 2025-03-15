# Crowd-Certain

A Python library for crowd-sourced label aggregation with uncertainty estimation and confidence scoring.

## Overview

Crowd-Certain is a comprehensive framework for aggregating labels from multiple annotators (crowd workers) while estimating the uncertainty and confidence in the aggregated labels.

## Documentation

Detailed documentation is available in the `crowd_certain/docs` folder:

- [Full README](crowd_certain/docs/README.md)
- [Installation Guide](crowd_certain/docs/INSTALLATION.md)
- [Usage Examples](crowd_certain/docs/USAGE.md)
- [API Reference](crowd_certain/docs/API.md)

## Quick Start

### Installation

```bash
# Using the installation script (recommended)
# For Unix/Linux/macOS:
./crowd_certain/scripts/install.sh

# For Windows:
crowd_certain\scripts\install.bat

# Or manually with pip
pip install -e .
```

The installation scripts will:
- Check if Python is installed (supports both 'python' and 'python3' commands)
- Verify that Python 3.10+ is available
- Check if a virtual environment or conda environment already exists
- Ask if you want to use the existing environment, create a new one, or abort
- Guide you through choosing an installation method
- Set up the appropriate environment
- Create an activation script that you can use to easily activate the environment

After installation, activate the environment with:
```bash
# On Unix/Linux/macOS:
source ./crowd_certain/activate_env.sh

# On Windows:
crowd_certain\activate_env.bat
```

### Basic Usage

```python
import crowd_certain
from crowd_certain.utilities.utils import AIM1_3, Settings, DatasetNames

# Configure settings
config = Settings()
config.dataset.dataset_name = DatasetNames.IONOSPHERE

# Run analysis
results = AIM1_3.calculate_one_dataset(config=config)
```

### Dashboard

The project includes a Streamlit-based dashboard that provides a user-friendly interface for running simulations and visualizing results.

To run the dashboard:

```bash
# Install all dependencies (if not already installed)
pip install -r requirements.txt

# Option 1: Using the entry point script (recommended)
python crowd_certain/run_streamlit.py
# Or directly execute it
./crowd_certain/run_streamlit.py

# Option 2: Using the provided scripts
# On Unix/Linux/macOS:
./crowd_certain/scripts/run_dashboard.sh

# On Windows:
crowd_certain\scripts\run_dashboard.bat

# Option 3: Directly with Streamlit
streamlit run crowd_certain/utilities/dashboard.py
```

The dashboard allows you to:
- Configure simulation parameters through an intuitive interface
- Run simulations on different datasets
- Visualize worker strength and weight relationships
- Explore confidence scores and performance metrics
- Compare different aggregation methods

For more details, see the [Dashboard Documentation](crowd_certain/docs/DASHBOARD.md).

## Project Structure

```
crowd-certain/
├── crowd_certain/       # Main package code
│   ├── datasets/        # Dataset handling
│   ├── docs/            # Documentation
│   ├── notebooks/       # Jupyter notebooks
│   ├── outputs/         # Output files
│   ├── scripts/         # Installation and utility scripts
│   │   ├── install.sh   # Unix installation script
│   │   ├── install.bat  # Windows installation script
│   │   ├── run_dashboard.sh # Unix dashboard script
│   │   └── run_dashboard.bat# Windows dashboard script
│   ├── utilities/       # Utility functions and classes
│   │   └── dashboard.py # Streamlit dashboard
│   ├── activate_env.sh  # Environment activation script
│   └── run_streamlit.py # Entry point for the dashboard
├── requirements.txt     # Python dependencies (pip)
└── requirements.yml     # Conda environment specification
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contact

Artin Majdi - msm2024@gmail.com
