"""
Configuration module for crowd-certain.

This module provides configuration classes, parameter definitions, and
settings handling for the crowd-certain package.
"""

from crowd_certain.utilities.config.params import *
from crowd_certain.utilities.config.settings import get_settings, Settings

__all__ = [
    'Settings',
    'get_settings',
    # All enum types from params
    'DatasetNames',
    'DataModes',
    'UncertaintyTechniques',
    'ConsistencyTechniques',
    'EvaluationMetricNames',
    'FindingNames',
    'OutputModes',
    'OtherBenchmarkNames',
    'MainBenchmarks',
    'ProposedTechniqueNames',
    'StrategyNames',
    'ConfidenceScoreNames',
    'SimulationMethods',
    'ResultType',
    'WeightType',
    'Result2Type',
    'ResultComparisonsType'
]
