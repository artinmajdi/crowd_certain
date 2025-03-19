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
    StrategyNames,
    ConfidenceScoreNames,
    SimulationMethods,
    ResultType,
    WeightType,
    Result2Type,
    ResultComparisonsType
)

# Import from settings module
from crowd_certain.utilities.parameters.settings import (
    Settings,
    ConfigManager,
    DatasetSettings,
    OutputSettings,
    TechniqueSettings,
    SimulationSettings
)

__all__ = [
    # Settings classes
    'Settings',
    'ConfigManager',
    'DatasetSettings',
    'OutputSettings',
    'TechniqueSettings',
    'SimulationSettings',

    # Base enum class
    'EnumWithHelpers',

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

    # Data classes
    'ResultType',
    'WeightType',
    'Result2Type',
    'ResultComparisonsType'
]
