#!/bin/bash

# Test script for reorganized modules
echo "Testing reorganized module structure..."

# Test imports
python -c "from crowd_certain.utilities import AIM1_3, Aim1_3_Data_Analysis_Results, DatasetNames" && \
    echo "✓ Basic imports successful" || echo "✗ Basic imports failed"

# Test loading settings
python -c "from crowd_certain.utilities import get_settings; config = get_settings()" && \
    echo "✓ Settings loading successful" || echo "✗ Settings loading failed"

# Test example
python -c "import crowd_certain.examples.visualize_results_example" && \
    echo "✓ Example import successful" || echo "✗ Example import failed"

echo "Reorganization tests completed."
