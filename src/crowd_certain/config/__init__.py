"""
Configuration module for crowd-certain.

This module provides configuration files and settings for the crowd-certain package.
"""

from pathlib import Path

DEFAULT_CONFIG_DICT = {
  "dataset": {
    "data_mode": "train",
    "random_state": 42,
    "path_all_datasets": "./datasets",
    "dataset_name": "ionosphere",
    "datasetNames": [
      "ionosphere",
      "chess",
      "mushroom",
      "spambase",
      "breast-cancer",
      "banknote",
      "sonar"
    ],
    "non_null_samples": True,
    "train_test_ratio": 0.8,
    "shuffle": True,
    "augmentation_count": 1,
    "main_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
  },
  "simulation": {
    "n_workers_min_max": [3, 8],
    "high_dis": 1.0,
    "low_dis": 0.4,
    "num_simulations": 5,
    "num_seeds": 3,
    "use_parallelization": True,
    "max_parallel_workers": 8,
    "simulation_methods": "random_states"
  },
  "technique": {
    "uncertainty_techniques": [
      "standard_deviation",
      "entropy",
      "coefficient_of_variation",
      "predicted_interval",
      "confidence_interval"
    ],
    "consistency_techniques": [
      "one_minus_uncertainty",
      "one_divided_by_uncertainty"
    ]
  },
  "output": {
    "mode": "calculate",
    "save": True,
    "path": "./outputs"
  },
  "dashboard": {
    "port": 8501,
    "theme": "light",
    "default_dataset": "ionosphere",
    "auto_download": True,
    "cache_results": True,
    "max_cache_size": 100
  },
  "logging": {
    "level": "INFO",
    "file": "crowd_certain.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console_output": True
  },
  "development": {
    "debug": False,
    "profile_performance": False,
    "test_mode": False,
    "api_docs_url": "/docs"
  }
}


# Define the config directory path
CONFIG_PATH = Path(__file__).parent.absolute()

__all__ = ['CONFIG_PATH', 'DEFAULT_CONFIG_DICT']
