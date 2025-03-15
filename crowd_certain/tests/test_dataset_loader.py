"""
Test script for the updated DatasetNames class and load_dataset function.
"""

import pytest
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.settings import Settings
from crowd_certain.utilities.params import DatasetNames, ReadMode
from crowd_certain.utilities import dataset_loader
from pathlib import Path

def test_dataset_loader():
    """Test the dataset loader with the updated DatasetNames class."""
    print("Testing dataset loader with updated DatasetNames class...")

    # Create a configuration with required fields
    config = Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            read_mode=ReadMode.AUTO,
            path_all_datasets=Path('crowd_certain/datasets')
        ),
        simulation=dict(),
        technique=dict(),
        output=dict()
    )

    # Print dataset information
    print(f"Dataset name: {str(config.dataset.dataset_name)}")
    print(f"UCI ID: {config.dataset.dataset_name.uci_id}")

    try:
        print("Attempting to load dataset...")
        data, feature_columns = dataset_loader.load_dataset(config=config)

        # Add assertions to validate the data
        assert data is not None, "Dataset should not be None"
        assert "train" in data, "Dataset should contain training data"
        assert "test" in data, "Dataset should contain test data"
        assert feature_columns is not None, "Feature columns should not be None"
        assert len(feature_columns) > 0, "Feature columns should not be empty"

        print(f"Successfully loaded dataset with {len(data['train'])} training samples and {len(data['test'])} test samples")
        print(f"Feature columns: {feature_columns[:5]}...")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        pytest.fail(f"Failed to load dataset: {str(e)}")

if __name__ == "__main__":
    test_dataset_loader()
    print("Dataset loader test passed!")
