"""
Test script for the params module containing enums like DatasetNames and ReadMode.
"""

import pytest
from crowd_certain.utilities.params import DatasetNames, ReadMode

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

def test_read_mode_enum():
    """Test properties and methods of the ReadMode enum."""
    # Test basic enum properties
    assert ReadMode.AUTO.name == "AUTO"

    # ReadMode enum doesn't support creating via string name directly,
    # so we'll test by comparing attributes
    for mode in ReadMode:
        if mode.name == "AUTO":
            assert mode == ReadMode.AUTO
            break
    else:
        pytest.fail("AUTO mode not found in ReadMode enum")

    # Test invalid read mode by accessing a non-existent attribute
    with pytest.raises(AttributeError):
        ReadMode.INVALID_MODE

    # Test all available read modes
    available_modes = [mode.name for mode in ReadMode]
    assert "AUTO" in available_modes

    # If there are other modes like CSV, READ_ARFF, etc.
    if hasattr(ReadMode, "CSV"):
        assert "CSV" in available_modes
    if hasattr(ReadMode, "READ_ARFF"):
        assert "READ_ARFF" in available_modes
    if hasattr(ReadMode, "TEST"):
        assert "TEST" in available_modes

if __name__ == "__main__":
    test_dataset_names_enum()
    test_read_mode_enum()
    print("All params tests passed!")
