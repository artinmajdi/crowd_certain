"""
Configuration module for crowd-certain.

This module provides configuration files and settings for the crowd-certain package.
"""

from pathlib import Path
import json

# Define the config directory path
CONFIG_DIR = Path(__file__).parent.absolute()

# Check if config_default.json exists and load it
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config_default.json"
DEFAULT_CONFIG_JSON = None

if DEFAULT_CONFIG_PATH.exists():
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        DEFAULT_CONFIG_JSON = json.load(f)

# Add DEFAULT_CONFIG to __all__ if it exists
if DEFAULT_CONFIG_JSON is not None:
    __all__ = ['CONFIG_DIR', 'DEFAULT_CONFIG_JSON']
else:
    __all__ = ['CONFIG_DIR']
