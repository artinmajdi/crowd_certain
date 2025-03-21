#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}          Crowd-Certain Installation Script                ${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo ""

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to determine which Python command to use
get_python_command() {
    if command_exists python; then
        echo "python"
    elif command_exists python3; then
        echo "python3"
    else
        echo ""
    fi
}

# Check if Python is installed
PYTHON_CMD=$(get_python_command)
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Neither 'python' nor 'python3' commands were found.${NC}"
    echo -e "${YELLOW}Please install Python 3.10 or higher before continuing.${NC}"
    echo -e "${YELLOW}Visit https://www.python.org/downloads/ for installation instructions.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")

if [ "$PYTHON_MAJOR_VERSION" -lt 3 ]; then
    echo -e "${RED}Error: Python 3.10 or higher is required, but Python $PYTHON_VERSION was found.${NC}"
    echo -e "${YELLOW}Please install Python 3.10 or higher before continuing.${NC}"
    echo -e "${YELLOW}Visit https://www.python.org/downloads/ for installation instructions.${NC}"
    exit 1
fi

PYTHON_MINOR_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MAJOR_VERSION" -eq 3 ] && [ "$PYTHON_MINOR_VERSION" -lt 10 ]; then
    echo -e "${YELLOW}Warning: Python 3.10 or higher is recommended, but Python $PYTHON_VERSION was found.${NC}"
    echo -e "${YELLOW}Some features may not work correctly.${NC}"
    read -p "Do you want to continue anyway? (y/n): " continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installation aborted.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"

# Function to install using pip
install_with_pip() {
    echo -e "${YELLOW}Installing with pip...${NC}"

    if [ "$1" == "venv" ]; then
        # Check if virtual environment already exists
        if [ -d ".venv" ]; then
            echo -e "${YELLOW}A virtual environment (.venv) already exists.${NC}"
            read -p "Do you want to use the existing environment? (y/n): " use_existing
            if [[ "$use_existing" =~ ^[Yy]$ ]]; then
                echo -e "${GREEN}Using existing virtual environment.${NC}"
            else
                read -p "Do you want to delete the existing environment and create a new one? (y/n): " delete_existing
                if [[ "$delete_existing" =~ ^[Yy]$ ]]; then
                    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
                    rm -rf .venv
                    echo -e "${YELLOW}Creating new virtual environment (.venv)...${NC}"
                    $PYTHON_CMD -m venv .venv
                else
                    echo -e "${YELLOW}Installation aborted.${NC}"
                    exit 1
                fi
            fi
        else
            echo -e "${YELLOW}Creating virtual environment (.venv)...${NC}"
            $PYTHON_CMD -m venv .venv
        fi

        # Activate virtual environment
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source .venv/Scripts/activate
        else
            source .venv/bin/activate
        fi

        echo -e "${GREEN}Virtual environment activated.${NC}"
    fi

    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -e .

    echo -e "${GREEN}Installation completed successfully!${NC}"

    if [ "$1" == "venv" ]; then
        echo -e "${YELLOW}To activate the virtual environment in the future, run:${NC}"
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            echo -e "  source .venv/Scripts/activate"
        else
            echo -e "  source .venv/bin/activate"
        fi
    fi
}

# Function to install using conda
install_with_conda() {
    echo -e "${YELLOW}Installing with conda...${NC}"

    # Check if conda exists
    if ! command_exists conda; then
        echo -e "${RED}Error: conda is not installed or not in PATH.${NC}"
        echo -e "${YELLOW}Please install conda first or choose the pip installation method.${NC}"
        exit 1
    fi

    # Check if mamba exists, install if not
    if ! command_exists mamba; then
        echo -e "${YELLOW}Mamba not found. Installing mamba using conda...${NC}"
        conda install -c conda-forge mamba -y
    fi

    # Check if conda environment already exists
    if conda env list | grep -q "crowd-certain"; then
        echo -e "${YELLOW}A conda environment 'crowd-certain' already exists.${NC}"
        read -p "Do you want to use the existing environment? (y/n): " use_existing
        if [[ "$use_existing" =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Using existing conda environment.${NC}"
        else
            read -p "Do you want to remove the existing environment and create a new one? (y/n): " delete_existing
            if [[ "$delete_existing" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Removing existing conda environment...${NC}"
                conda env remove -n crowd-certain
                echo -e "${YELLOW}Creating new conda environment from environment.yml...${NC}"
                mamba env update --file environment.yml
            else
                echo -e "${YELLOW}Installation aborted.${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${YELLOW}Creating conda environment from environment.yml...${NC}"
        mamba env update --file environment.yml
    fi

    # Activate the conda environment
    echo -e "${YELLOW}Activating conda environment...${NC}"
    # We can't directly activate the environment in the script, so we'll just inform the user
    echo -e "${YELLOW}Please activate the conda environment manually with:${NC}"
    echo -e "  conda activate crowd-certain"
    echo -e "${YELLOW}Then run:${NC}"
    echo -e "  pip install -e ."

    # Ask if user wants to continue with installation
    read -p "Have you activated the conda environment? (y/n): " activated
    if [[ "$activated" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installing package in development mode...${NC}"
        pip install -e .
        echo -e "${GREEN}Installation completed successfully!${NC}"
    else
        echo -e "${YELLOW}Please complete the installation by running:${NC}"
        echo -e "  conda activate crowd-certain"
        echo -e "  pip install -e ."
    fi
}

# Main installation process
echo -e "Please select installation method:"
echo -e "  1) Pip with virtual environment (.venv)"
echo -e "  2) Pip (system-wide or in current environment)"
echo -e "  3) Conda/Mamba (recommended for complex dependencies)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        install_with_pip "venv"
        ;;
    2)
        install_with_pip "system"
        ;;
    3)
        install_with_conda
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}===========================================================${NC}"
echo -e "${GREEN}Crowd-Certain installation process completed!${NC}"
echo -e "${YELLOW}See the documentation in the docs/ directory for usage examples.${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Create activation script
if [ "$choice" == "1" ]; then
    echo '#!/bin/bash' > scripts/activate_env.sh
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo 'source .venv/Scripts/activate' >> scripts/activate_env.sh
    else
        echo 'source .venv/bin/activate' >> scripts/activate_env.sh
    fi
    chmod +x scripts/activate_env.sh

    echo -e "${GREEN}Installation complete!${NC}"
    echo -e "${BLUE}To activate the environment, run:${NC}"
    echo -e "${BLUE}source ./scripts/activate_env.sh${NC}"
else
    # Create activation script for conda
    echo '#!/bin/bash' > scripts/activate_env.sh
    echo 'conda activate crowd-certain' >> scripts/activate_env.sh
    chmod +x scripts/activate_env.sh

    echo -e "${GREEN}Installation complete!${NC}"
    echo -e "${BLUE}To activate the environment, run:${NC}"
    echo -e "${BLUE}source ./scripts/activate_env.sh${NC}"
fi
