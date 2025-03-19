"""
Configuration module for crowd-certain.

This module provides configuration files and settings for the crowd-certain package.
"""

from pathlib import Path

# Define the config directory path
CONFIG_DIR = Path(__file__).parent.absolute()

__all__ = ['CONFIG_DIR']
