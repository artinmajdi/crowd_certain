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
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}Streamlit is not installed. Installing dependencies...${NC}"
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo ""
fi

# Run the dashboard
echo -e "${GREEN}Launching Crowd-Certain Dashboard...${NC}"
cd "$PROJECT_ROOT"
streamlit run "src/crowd_certain/dashboard.py"
