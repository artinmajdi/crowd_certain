#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test script for reorganized modules
echo "Testing reorganized module structure..."

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

# Test imports
python -c "from crowd_certain.utilities.parameters import DatasetNames; from crowd_certain.dashboard_components import AIM1_3" && \
    echo -e "${GREEN}✓ Basic imports successful${NC}" || echo -e "${RED}✗ Basic imports failed${NC}"

# Test loading settings
python -c "from crowd_certain.utilities.parameters import Settings; config = Settings()" && \
    echo -e "${GREEN}✓ Settings loading successful${NC}" || echo -e "${RED}✗ Settings loading failed${NC}"

# Test dashboard
python -c "from crowd_certain.dashboard import main" && \
    echo -e "${GREEN}✓ Dashboard import successful${NC}" || echo -e "${RED}✗ Dashboard import failed${NC}"

echo "Reorganization tests completed."
