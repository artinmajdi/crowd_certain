#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for the wound analysis dashboard.
This file provides the entry points for the wound-dashboard command.
"""

import streamlit.cli_util as stcli
from pathlib import Path

def run_dashboard():
    """
    Run the Streamlit dashboard.
    This function is called when the user runs the wound-dashboard command.
    It uses Streamlit to run the dashboard.py file.
    """
    dashboard_path = Path(__file__).parent / "dashboard.py"
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    stcli.main()


if __name__ == "__main__":
    run_dashboard()
