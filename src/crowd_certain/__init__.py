"""Crowd-Certain: A framework for crowd-sourcing with certainty estimation.

This package provides tools and utilities for crowd-sourcing with certainty estimation,
including simulation, uncertainty calculation, and evaluation metrics.
"""

from crowd_certain.utilities import parameters
from crowd_certain.utilities._components import *
from crowd_certain.utilities._visualization import *
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.benchmarks import BenchmarkTechniques

__version__ = "1.0.0"

__all__ = [
    'parameters',
    'AIM1_3',
    'BenchmarkTechniques',
]

