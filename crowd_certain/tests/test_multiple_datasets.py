"""
Test script for loading and processing multiple datasets.
"""

import pytest
from pathlib import Path

from crowd_certain.utilities.config.settings import Settings
from crowd_certain.utilities.config.params import DatasetNames
from crowd_certain.utilities.io import dataset_loader

# List of datasets to test, add or remove datasets as needed
DATASETS_TO_TEST = [
    DatasetNames.IONOSPHERE,
    # Add other dataset names that exist in your DatasetNames enum
    # For example: DatasetNames.IRIS, DatasetNames.MNIST, etc.
]

@pytest.mark.parametrize("dataset_name", DATASETS_TO_TEST)
def test_load_specific_dataset(dataset_name):
    """Test loading a specific dataset."""
    print(f"Testing loading of dataset: {dataset_name}")

    # Create configuration for this dataset
    config = Settings(
        dataset=dict(
            dataset_name=dataset_name,
            path_all_datasets=Path('crowd_certain/datasets')
        ),
        simulation=dict(),
        technique=dict(),
        output=dict()
    )

    # Print dataset information
    print(f"Dataset name: {str(config.dataset.dataset_name)}")
    if hasattr(config.dataset.dataset_name, 'uci_id'):
        print(f"UCI ID: {config.dataset.dataset_name.uci_id}")

    try:
        # Load the dataset
        data, feature_columns = dataset_loader.load_dataset(config=config)

        # Verify dataset was loaded successfully
        assert data is not None, "Dataset should not be None"
        assert "train" in data, "Dataset should contain training data"
        assert "test" in data, "Dataset should contain test data"

        # Print dataset details
        print(f"Successfully loaded dataset with {len(data['train'])} training samples and {len(data['test'])} test samples")
        print(f"Feature columns: {feature_columns[:5]}...")

        # Successful test, no need to return anything
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        pytest.fail(f"Failed to load dataset {dataset_name}: {str(e)}")

def test_invalid_dataset():
    """Test handling of invalid dataset names."""
    # Try with direct instantiation which should handle errors better
    try:
        # Try to create a Settings object with an invalid dataset name
        with pytest.raises(ValueError):
            # This should fail at instantiation time because "NONEXISTENT_DATASET"
            # is not a valid DatasetNames enum value
            config = Settings(
                dataset=dict(
                    dataset_name="NONEXISTENT_DATASET",  # This should cause a validation error
                    path_all_datasets=Path('crowd_certain/datasets')
                ),
                simulation=dict(),
                technique=dict(),
                output=dict()
            )
    except Exception as e:
        print(f"Error during invalid dataset test: {str(e)}")
        pytest.fail(f"Unexpected error in invalid dataset test: {str(e)}")

if __name__ == "__main__":
    for dataset in DATASETS_TO_TEST:
        test_load_specific_dataset(dataset)
    test_invalid_dataset()
    print("All multiple dataset tests completed!")
