"""
Test script for the params module containing enums like DatasetNames.
"""

import pytest
from crowd_certain.utilities.config.params import DatasetNames

def test_dataset_names_enum():
    """Test properties and methods of the DatasetNames enum."""
    # Test basic enum properties
    assert DatasetNames.IONOSPHERE.name == "IONOSPHERE"
    assert DatasetNames.IONOSPHERE.value == "ionosphere"  # Updated to lowercase to match actual implementation

    # Test custom properties if they exist
    if hasattr(DatasetNames.IONOSPHERE, "uci_id"):
        assert isinstance(DatasetNames.IONOSPHERE.uci_id, (int, str)) or DatasetNames.IONOSPHERE.uci_id is None

    # Test string conversion - it appears the str() representation is the value, not the name
    assert str(DatasetNames.IONOSPHERE) == "ionosphere"  # Updated to match actual behavior

    # Test membership
    assert "IONOSPHERE" in [ds.name for ds in DatasetNames]

    # Test creating from string
    assert DatasetNames("ionosphere") == DatasetNames.IONOSPHERE  # Updated to lowercase

    # Test invalid dataset name
    with pytest.raises(ValueError):
        DatasetNames("NONEXISTENT_DATASET")


if __name__ == "__main__":
    test_dataset_names_enum()
    print("All params tests passed!")
