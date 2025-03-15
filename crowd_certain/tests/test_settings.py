"""
Test script for the Settings class that handles configuration.
"""

import os
import json
import tempfile
from pathlib import Path

from crowd_certain.utilities.settings import Settings
from crowd_certain.utilities.params import DatasetNames, ReadMode

def test_settings_initialization():
    """Test the initialization of the Settings class with different parameters."""
    # Test with minimal configuration
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            read_mode=ReadMode.AUTO,
        ),
        simulation=dict(random_seed=42),
        technique=dict(name="test_technique"),
        output=dict(save_results=True)
    )

    assert config.dataset.dataset_name == DatasetNames.IONOSPHERE
    assert config.dataset.read_mode == ReadMode.AUTO
    assert config.simulation.random_seed == 42
    assert config.technique.name == "test_technique"
    assert config.output.save_results is True

def test_settings_with_dict():
    """Test creating Settings from a dictionary."""
    # Create a settings object using a dictionary
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            read_mode=ReadMode.AUTO,
            path_all_datasets=Path('crowd_certain/datasets')
        ),
        simulation=dict(random_seed=123),
        technique=dict(name="test_technique"),
        output=dict(save_results=True)
    )

    assert config.dataset.dataset_name == DatasetNames.IONOSPHERE
    assert config.dataset.read_mode == ReadMode.AUTO
    assert isinstance(config.dataset.path_all_datasets, Path)
    assert config.simulation.random_seed == 123

def test_settings_with_json_config():
    """Test creating Settings with config values that could come from a JSON file."""
    # This test simulates loading from a JSON but without actually reading a file
    # Create a Settings with values that would typically be provided in a JSON config
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            read_mode=ReadMode.AUTO,
            path_all_datasets=str(Path('crowd_certain/datasets'))
        ),
        simulation=dict(random_seed=456),
        technique=dict(name="test_technique"),
        output=dict(save_results=False)
    )

    assert config.dataset.dataset_name == DatasetNames.IONOSPHERE
    assert config.simulation.random_seed == 456
    assert config.output.save_results is False

if __name__ == "__main__":
    test_settings_initialization()
    test_settings_with_dict()
    test_settings_with_json_config()
    print("All settings tests passed!")
