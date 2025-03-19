from __future__ import annotations
import enum
from typing import Type, TypeVar, Callable, List, Iterator, Tuple

T = TypeVar('T', bound='EnumWithHelpers')
class EnumWithHelpers(enum.Enum):
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

# def members(cls):  # Untyped

# 	# Add the members class method
# 	cls.members: Callable[[], list[enum.Enum]] = classmethod(lambda cls2: list(cls2.__members__))

# 	# Adding all options in a list
# 	# This can be achieved by calling list(YourEnum)
# 	# cls.all = classmethod(lambda cls2: [cls2[n] for n in list(cls2.__members__)])

# 	cls.values: Callable[[], list[str]]  = classmethod(lambda cls2: [n.value for n in cls2])

# 	# Make the class iterable
# 	# cls.__iter__: Callable[[], Iterator[str]] = lambda self: iter(self.__members__.keys())

# 	# Overwrite the __str__ method, to output only the name of the member
# 	# cls.__str__: Callable[[], str] = lambda self: self.value
# 	return cls

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


def main():
	print(EvaluationMetricNames.AUC in ['AUC', 'ACC'])
	print('something')

if __name__ == '__main__':
	main()

