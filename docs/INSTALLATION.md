# Installation Guide

[← Back to Main README](../README.md) | [← Back to Documentation Index](index.md) | [Next: Usage Guide →](USAGE.md)

---

Crowd-Certain can be installed using either pip or conda. The package provides installation scripts that will guide you through the process.

## Prerequisites

- Python 3.10 or higher
- pip or conda/mamba package manager

## Installation Methods

### Method 1: Using the Installation Scripts (Recommended)

The project provides convenient installation scripts for both Unix-based systems and Windows. These scripts will:

- Check if Python is installed (supports both 'python' and 'python3' commands)
- Verify that Python 3.10+ is available
- Check if a virtual environment or conda environment already exists
- Ask if you want to use the existing environment, create a new one, or abort
- Guide you through choosing an installation method
- Set up the appropriate environment
- Create an activation script that you can use to easily activate the environment

After installation, you'll need to activate the environment by running:

```bash
# On Unix/Linux/macOS:
source ./activate_env.sh

# On Windows:
activate_env.bat
```

#### For Unix/Linux/macOS

```bash
./scripts/install.sh
```

This script will prompt you to choose between:

1. Pip with virtual environment (.venv)
2. Pip (system-wide or in current environment)
3. Conda/Mamba (recommended for complex dependencies)

#### For Windows

```bash
scripts\install.bat
```

This batch file provides the same options as the Unix script.

### Method 2: Manual Installation

#### Option 1: Using pip (recommended for most users)

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the package in development mode:

   ```bash
   pip install -e .
   ```

   This will automatically install all required dependencies listed in `crowd_certain/config/requirements.txt`.

#### Option 2: Using conda/mamba (recommended for complex dependencies)

1. Create and activate a conda environment using the provided environment file:

   ```bash
   conda env create -f crowd_certain/config/environment.yml
   conda activate crowd-certain
   ```

2. Install the package in development mode:

   ```bash
   pip install -e .
   ```

## Activating the Environment

After installation, you can activate the environment using:

- For pip with virtual environment:

  ```bash
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```

- For conda:

  ```bash
  conda activate crowd-certain
  ```

- Using the provided activation script:

  ```bash
  source ./crowd_certain/config/activate_env.sh  # On Windows: crowd_certain\config\activate.bat
  ```

## Dependencies

The following dependencies will be automatically installed:

- matplotlib - For visualization
- numpy - For numerical operations
- pandas - For data manipulation
- scikit-learn - For machine learning algorithms
- scipy - For scientific computing
- seaborn - For statistical data visualization
- setuptools - For package setup
- tqdm - For progress bars
- wget - For downloading datasets
- crowd-kit - For crowd-sourcing algorithms
- jupyter - For notebook support
- ipywidgets - For interactive widgets in notebooks

## Verifying Installation

To verify that Crowd-Certain has been installed correctly, you can run:

```python
import crowd_certain
print(crowd_certain.__version__)
```

## Troubleshooting

### Common Issues

1. **Python Not Found**: If the installation script reports that Python is not found, make sure Python 3.10+ is installed and available in your PATH. The scripts check for both 'python' and 'python3' commands.

2. **Missing Dependencies**: If you encounter errors about missing dependencies, try reinstalling with:

   ```bash
   pip install -e . --force-reinstall
   ```

3. **Conda Environment Issues**: If you're using conda and encounter environment issues, try:

   ```bash
   conda clean --all
   conda env remove -n crowd-certain
   conda env create -f requirements.yml
   ```

4. **Version Conflicts**: If you encounter version conflicts with other packages, consider creating a dedicated virtual environment:

   ```bash
   python -m venv crowd_env
   source crowd_env/bin/activate  # On Windows: crowd_env\Scripts\activate
   pip install -e .
   ```

### Getting Help

If you encounter any issues during installation, please open an issue on the [GitHub repository](https://github.com/artinmajdi/taxonomy/issues) with details about your environment and the error messages you're seeing.

---
