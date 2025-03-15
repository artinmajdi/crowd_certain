#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}          Starting Crowd-Certain Dashboard                 ${NC}"
echo -e "${GREEN}===========================================================${NC}"
echo ""

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CROWD_CERTAIN_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$CROWD_CERTAIN_ROOT/.." &> /dev/null && pwd )"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}Streamlit is not installed. Installing dependencies...${NC}"
    # Use the main project's requirements.txt file
    pip install -r "$PROJECT_ROOT/crowd_certain/config/requirements.txt"
    echo ""
fi

# Run the dashboard
echo -e "${GREEN}Launching Crowd-Certain Dashboard...${NC}"
cd "$PROJECT_ROOT"
streamlit run "$CROWD_CERTAIN_ROOT/utilities/dashboard.py"
