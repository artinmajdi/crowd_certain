from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator, List, Literal, Tuple, Type, TypeAlias, TypeVar

import pandas as pd

T = TypeVar('T', bound='EnumWithHelpers')
class EnumWithHelpers(enum.Enum):
	"""
	EnumWithHelpers extends the basic Enum class with additional helper methods.

	This class adds utility methods for working with enum members, making it easier
	to access members as lists, iterate through them, and convert them to strings.

	Methods:
		members(): Returns a list of all enum members.
		all(): Returns a list of all enum values (same as members).
		values(): Returns a list of all enum member values.
		__iter__(): Makes the enum directly iterable, yielding member names.
		__str__(): Custom string representation of enum members. If the value is a tuple,
					returns the first element of the tuple, otherwise returns the value as a string.
	"""
	@classmethod
	def members(cls: Type[T]) -> List[T]:
		return list(cls.__members__.values())

	@classmethod
	def all(cls: Type[T]) -> List[T]:
		return list(cls)

	@classmethod
	def values(cls: Type[T]) -> List[str]:
		return [member.value for member in cls]

	def __iter__(self: T) -> Iterator[str]:
		return iter(self.__members__.keys())

	def __str__(self) -> str:
		if isinstance(self.value, tuple) and len(self.value) > 0:
			return self.value[0]
		return str(self.value)


# @members
class DatasetNames(EnumWithHelpers):
	CHESS         = "chess"           # Chess (King-Rook vs. King-Pawn)
	MUSHROOM      = "mushroom"        # Mushroom
	IRIS          = "iris"            # Iris
	SPAMBASE      = "spambase"        # Spambase
	TIC_TAC_TOE   = "tic-tac-toe"     # Tic-Tac-Toe Endgame
	HEART         = "heart"           # Heart Disease (replacement for SICK)
	WAVEFORM      = "waveform"        # Waveform Database Generator (Version 1)
	CAR           = "car"             # Car Evaluation
	VOTE          = "vote"            # Congressional Voting Records
	IONOSPHERE    = "ionosphere"      # Ionosphere
	BREAST_CANCER = "breast-cancer"   # Breast Cancer Wisconsin (Diagnostic)
	BANKNOTE      = "banknote"        # Banknote Authentication
	SONAR         = "sonar"           # Sonar, Mines vs. Rocks

	@property
	def uci_id(self) -> int:
		"""Get the UCI dataset ID."""
		uci_ids = {
			"chess"        : 22,
			"mushroom"     : 73,
			"iris"         : 53,
			"spambase"     : 94,
			"tic-tac-toe"  : 101,
			"heart"        : 45,
			"waveform"     : 107,
			"car"          : 19,
			"vote"         : 105,
			"ionosphere"   : 52,
			"breast-cancer": 17,
			"banknote"     : 267,
			"sonar"        : 151
		}
		return uci_ids[self.value]

# @members
class DataModes(EnumWithHelpers):
	TRAIN = 'train'
	TEST  = 'test'
	ALL   = 'all'


# @members
class UncertaintyTechniques(EnumWithHelpers):
	STD     = "standard_deviation"
	ENTROPY = "entropy"
	CV		= "coefficient_of_variation"
	PI      = "predicted_interval"
	CI 		= "confidence_interval"
	# BMA     = "bayesian_model_avaraging"
	# CBM     = "committee_based_method"
	# CP      = "conformal_prediction"


# @members
class ConsistencyTechniques(EnumWithHelpers):
	ONE_MINUS_UNCERTAINTY = "one_minus_uncertainty"
	ONE_DIVIDED_BY_UNCERTAINTY = "one_divided_by_uncertainty"


# @members
class EvaluationMetricNames(EnumWithHelpers):
	ACC = 'ACC'
	AUC = 'AUC'
	F1  = 'F1'
	THRESHOLD = 'THRESHOLD'


# @members
class FindingNames(EnumWithHelpers):
	TRUTH = 'ground_truth'
	PRED  = 'predicted_probabilities'


# @members
class OutputModes(EnumWithHelpers):
	CALCULATE   = "calculate"
	LOAD_LOCAL  = "load_local"

# @members
class OtherBenchmarkNames(EnumWithHelpers):
	# GLAD = 'GLAD'
	KOS    = 'KOS'
	MACE   = 'MACE'
	MMSR   = 'MMSR'
	WAWA   = 'Wawa'
	ZBS    = 'ZeroBasedSkill'
	MV     = 'MajorityVote'
	DS     = 'DawidSkene'


# @members
class MainBenchmarks(EnumWithHelpers):
	TAO	  = 'Tao'
	SHENG = 'Sheng'


# @members
class ProposedTechniqueNames(EnumWithHelpers):
	PROPOSED = 'Crowd-Certain Without Penalization'
	PROPOSED_PENALIZED = 'Crowd-Certain'


# @members
class StrategyNames(EnumWithHelpers):
	FREQ = 'freq'
	BETA = 'beta'

# @members
class ConfidenceScoreNames(EnumWithHelpers):
	ECE   = 'ece score'
	BRIER = 'brier score loss'


# @members
class SimulationMethods(EnumWithHelpers):
	RANDOM_STATES = "random_states"
	MULTIPLE_CLASSIFIERS = "multiple_classifiers"



@dataclass
class ResultType:
	confidence_scores: dict[str, pd.DataFrame]
	aggregated_labels: pd.DataFrame
	metrics 		 : pd.DataFrame


@dataclass
class WeightType:
	PROPOSED: pd.DataFrame
	TAO     : pd.DataFrame
	SHENG   : pd.DataFrame


@dataclass
class Result2Type:
	proposed        : ResultType
	benchmark       : ResultType
	weight          : WeightType
	workers_reliabilities: WorkerReliabilitiesSeriesType
	workers_accruacies: pd.DataFrame
	n_workers       : int
	true_label      : dict[str, pd.DataFrame]


@dataclass
class ResultComparisonsType:
	outputs                   : dict
	config                    : 'Settings'
	weight_strength_relation  : pd.DataFrame



WorkerID                      : TypeAlias = str  # Format: 'worker_0','worker_1', etc.
WorkerAndTruthID              : TypeAlias = Tuple[WorkerID, Literal['truth']]  # Format: ('worker_0', 'truth')
SimulationID                  : TypeAlias = str  # Format: 'simulation_0','simulation_1', etc.
DataMode                      : TypeAlias = Literal['train', 'test']
WorkerReliabilitiesSeriesType: TypeAlias  = pd.Series  # Index = WorkerID,  values = float reliability scores
WorkerLabelsDFType            : TypeAlias = pd.DataFrame  # DataFrames with columns = ['truth', WorkerID...]
ClassifierPredsDFType         : TypeAlias = pd.DataFrame  # DataFrames with columns = ['simulation_0', ...]

ConsistencyTechniqueType: TypeAlias = Literal[ *ConsistencyTechniques.values() ]
# ConsistencyTechniqueType: TypeAlias = Type[ConsistencyTechniques]

def main():
	print(EvaluationMetricNames.AUC in ['AUC', 'ACC'])
	print('something')

if __name__ == '__main__':
	main()

