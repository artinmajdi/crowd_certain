"""
Main utilities module for crowd-certain.

This module imports and reuses functionality from the modular components.
This file is kept for backward compatibility with existing code.
"""

from crowd_certain.utilities.components.aim1_3 import AIM1_3
from crowd_certain.utilities.components.uncertainty import UncertaintyCalculator, calculate_uncertainties, calculate_consistency
from crowd_certain.utilities.components.worker_simulation import WorkerSimulator
from crowd_certain.utilities.components.weighting_schemes import WeightingSchemes
from crowd_certain.utilities.components.confidence_scoring import ConfidenceScorer
from crowd_certain.utilities.components.metrics import MetricsCalculator
from crowd_certain.utilities.components.orchestrator import Orchestrator
from crowd_certain.utilities.visualization.plots import Aim1_3_Data_Analysis_Results, AIM1_3_Plot

# Re-export the main classes and functions
__all__ = [
    'AIM1_3',
    'UncertaintyCalculator',
    'WorkerSimulator',
    'WeightingSchemes',
    'ConfidenceScorer',
    'MetricsCalculator',
    'Orchestrator',
    'calculate_uncertainties',
    'calculate_consistency',
    'Aim1_3_Data_Analysis_Results',
    'AIM1_3_Plot'
]

