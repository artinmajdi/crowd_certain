#!/usr/bin/env python3
"""
Entry point script to run the Crowd-Certain Dashboard.

Usage:
    python crowd_certain/run_streamlit.py
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Crowd-Certain Dashboard."""
    # Get the path to the dashboard
    crowd_certain_root = Path(__file__).resolve().parent
    dashboard_path = crowd_certain_root / "utilities" / "dashboard.py"
    project_root = crowd_certain_root.parent

    # Check if the dashboard exists
    if not dashboard_path.exists():
        print(f"Error: Could not find dashboard at {dashboard_path}")
        sys.exit(1)

    # Check if streamlit is installed
    try:
        subprocess.run(["streamlit", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Streamlit is not installed. Installing dependencies...")
        # Use the requirements.txt in the project root directory
        requirements_path = project_root / "requirements.txt"
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)

    # Run the dashboard
    print("Launching Crowd-Certain Dashboard...")
    subprocess.run(["streamlit", "run", str(dashboard_path)])

if __name__ == "__main__":
    main()
