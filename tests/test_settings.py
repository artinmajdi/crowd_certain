"""
Test script for the Settings class that handles configuration.
"""

import os
import json
import tempfile
from pathlib import Path

from crowd_certain.utilities.parameters.settings import Settings
from crowd_certain.utilities.parameters.params import DatasetNames

def test_settings_initialization():
    """Test the initialization of the Settings class with different parameters."""
    # Test with minimal configuration
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
        ),
        simulation=dict(num_seeds=42),  # Using num_seeds instead of random_seed
        technique=dict(),
        output=dict(save=True)  # Using save instead of save_results
    )

    assert config.dataset.dataset_name == DatasetNames.IONOSPHERE
    assert config.simulation.num_seeds == 42
    assert config.output.save is True

def test_settings_with_dict():
    """Test creating Settings from a dictionary."""
    # Create a settings object using a dictionary
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            path_all_datasets=Path('crowd_certain/datasets')
        ),
        simulation=dict(num_simulations=123),  # Using num_simulations which exists in SimulationSettings
        technique=dict(),
        output=dict(save=True)
    )

    assert config.dataset.dataset_name == DatasetNames.IONOSPHERE
    assert isinstance(config.dataset.path_all_datasets, Path)
    assert config.simulation.num_simulations == 123

def test_settings_with_json_config():
    """Test creating Settings with config values that could come from a JSON file."""
    # This test simulates loading from a JSON but without actually reading a file
    # Create a Settings with values that would typically be provided in a JSON config
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            path_all_datasets=str(Path('crowd_certain/datasets'))
        ),
        simulation=dict(n_workers_min_max=[2, 5]),  # Using n_workers_min_max which exists
        technique=dict(),
        output=dict(save=False)
    )

    assert config.dataset.dataset_name == DatasetNames.IONOSPHERE
    assert config.simulation.n_workers_min_max == [2, 5]
    assert config.output.save is False

if __name__ == "__main__":
    test_settings_initialization()
    test_settings_with_dict()
    test_settings_with_json_config()
    print("All settings tests passed!")
