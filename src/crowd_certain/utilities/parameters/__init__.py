"""
Configuration module for crowd-certain.

This module provides configuration classes, parameter definitions, and
settings handling for the crowd-certain package.
"""

# Import all classes from params module
from crowd_certain.utilities.parameters.params import (
    EnumWithHelpers,
    DatasetNames,
    DataModes,
    UncertaintyTechniques,
    ConsistencyTechniques,
    EvaluationMetricNames,
    FindingNames,
    OutputModes,
    OtherBenchmarkNames,
    MainBenchmarks,
    ProposedTechniqueNames,
)

# Import settings classes
from crowd_certain.utilities.parameters.settings import (
    Settings,
    ConfigManager,
    DatasetSettings,
    OutputSettings,
)

__all__ = [
    # Enums and parameter classes
    'EnumWithHelpers',
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
    # Settings classes
    'Settings',
    'ConfigManager',
    'DatasetSettings',
    'OutputSettings',
]
