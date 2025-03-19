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
source ./crowd_certain/config/activate_env.sh

# On Windows:
crowd_certain\config\activate.bat
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
# Option 1: Using the provided scripts (recommended)
# On Unix/Linux/macOS:
./crowd_certain/scripts/run_dashboard.sh

# On Windows:
crowd_certain\scripts\run_dashboard.bat

# Option 2: Directly with Streamlit
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

```bash
crowd-certain/
├── crowd_certain/       # Main package code
│   ├── config/          # Configuration files
│   │   ├── config.json  # Main configuration
│   │   ├── config_default.json # Default configuration
│   │   ├── requirements.txt # Python dependencies (pip)
│   │   ├── environment.yml # Conda environment specification
│   │   ├── activate_env.sh # Unix environment activation script
│   │   └── activate.bat # Windows environment activation script
│   ├── dashboard_components/ # Dashboard UI components
│   │   ├── aim1_3.py    # Aim 1.3 dashboard components
│   │   ├── confidence_scoring.py # Confidence scoring UI
│   │   ├── metrics.py   # Metrics visualization
│   │   ├── orchestrator.py # Dashboard orchestration
│   │   ├── uncertainty.py # Uncertainty visualization
│   │   ├── weighting_schemes.py # Weight schemes UI
│   │   └── worker_simulation.py # Worker simulation UI
│   ├── datasets/        # Dataset handling
│   │   ├── anneal/      # Anneal dataset
│   │   ├── biodeg/      # Biodeg dataset
│   │   ├── breast-cancer/ # Breast cancer dataset
│   │   ├── ionosphere/  # Ionosphere dataset
│   │   ├── iris/        # Iris dataset
│   │   └── [30+ more datasets] # Various UCI datasets
│   ├── docs/            # Documentation
│   │   ├── API.md       # API reference
│   │   ├── DASHBOARD.md # Dashboard documentation
│   │   ├── EXAMPLES.md  # Example scripts documentation
│   │   ├── INSTALLATION.md # Installation guide
│   │   ├── README.md    # Documentation overview
│   │   ├── STREAMLIT_TO_WEB.md # Streamlit deployment guide
│   │   ├── UCI_ML_REPO.md # UCI ML repository info
│   │   ├── USAGE.md     # Usage documentation
│   │   └── WEB_DEV.md   # Web development guide
│   ├── examples/        # Example scripts
│   │   ├── 1.3.3_paper_figures_per_dataset.py # Paper figure generation
│   │   ├── 1.3.4_paper_figures_final.py # Final paper figures
│   │   ├── all_datasets_example.py # Multiple dataset example
│   │   ├── custom_techniques_example.py # Custom techniques
│   │   ├── run_dashboard_example.py # Dashboard launch example
│   │   ├── save_load_results_example.py # Saving results
│   │   ├── single_dataset_example.py # Single dataset example
│   │   └── visualize_results_example.py # Visualization example
│   ├── notebooks/       # Jupyter notebooks
│   │   ├── 1.3.1 detailed-subversion.ipynb # Detailed analysis
│   │   ├── 1.3.2 weight_worker_strength_comparison.ipynb # Weight comparison
│   │   ├── 1.3.4_paper_figures_final.ipynb # Paper figures
│   │   └── old_experiments/ # Previous experiments
│   ├── outputs/         # Output files
│   │   ├── confidence_score/ # Confidence scoring outputs
│   │   ├── final_figures/ # Final figure outputs
│   │   ├── findings_comparisons/ # Comparative findings
│   │   ├── outputs/ # General outputs
│   │   └── weight_strength_relation/ # Weight-strength analysis
│   ├── scripts/         # Installation and utility scripts
│   │   ├── install.sh   # Unix installation script
│   │   ├── install.bat  # Windows installation script
│   │   ├── run_dashboard.sh # Unix dashboard script
│   │   ├── run_dashboard.bat # Windows dashboard script
│   │   └── test_reorganization.sh # Test organization script
│   └── utilities/       # Utility functions and classes
│       ├── dashboard.py # Streamlit dashboard
│       └── [additional utility files]
├── tests/               # Unit and integration tests
│   ├── conftest.py      # Test configuration
│   ├── test_config_functions.py # Configuration tests
│   ├── test_dataset_loader.py # Dataset loader tests
│   ├── test_multiple_datasets.py # Multiple dataset tests
│   ├── test_params.py  # Parameter tests
│   ├── test_settings.py # Settings tests
│   └── test_utils.py   # Utility tests
├── crowd_certain.egg-info/ # Package metadata
├── setup.py            # Package setup script
└── pyproject.toml      # Project configuration
```

For more information about the example scripts, see the [Examples Documentation](crowd_certain/docs/EXAMPLES.md).

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contact

Artin Majdi - <msm2024@gmail.com>
