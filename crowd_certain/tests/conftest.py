"""
Configuration file for pytest containing shared fixtures.
"""

import pytest
from pathlib import Path
from crowd_certain.utilities.settings import Settings
from crowd_certain.utilities.params import DatasetNames

@pytest.fixture
def base_config():
    """Provide a basic configuration object for tests."""
    return Settings(
        dataset=dict(
            dataset_name=DatasetNames.IONOSPHERE,
            path_all_datasets=Path('crowd_certain/datasets')
        ),
        simulation=dict(random_seed=42),
        technique=dict(name="test_technique"),
        output=dict(save_results=True)
    )

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    import numpy as np

    # Create synthetic data
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    n_features = 5

    # Generate random features
    X = np.random.rand(n_samples, n_features)

    # Generate random binary labels
    y = np.random.randint(0, 2, size=n_samples)

    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Feature names
    feature_columns = [f"feature_{i}" for i in range(n_features)]

    # Return structured data dict
    return {
        "data": {
            "train": {
                "X": X_train,
                "y": y_train
            },
            "test": {
                "X": X_test,
                "y": y_test
            }
        },
        "feature_columns": feature_columns
    }
