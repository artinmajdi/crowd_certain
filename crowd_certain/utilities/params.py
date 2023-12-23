from __future__ import annotations
import enum


def members(cls):

	# Add the members class method
	cls.members = classmethod(lambda cls2: list(cls2.__members__))

	# Adding all options in a list
	# This can be achieved by calling list(YourEnum)
	# cls.all = classmethod(lambda cls2: [cls2[n] for n in list(cls2.__members__)])
	cls.all = classmethod(lambda cls2: list(cls2))

	cls.values = classmethod(lambda cls2: [n.value for n in cls2])

	# Make the class iterable
	cls.__iter__ = lambda self: iter(self.__members__.keys())

	# Overwrite the __str__ method, to output only the name of the member
	cls.__str__ = lambda self: self.value
	return cls

@members
class ReadMode(enum.Enum):
	READ_ARFF = 'read_arff'
	READ      = 'read'
	DOWNLOAD  = 'download'


@members
class DatasetNames(enum.Enum):
	KR_VS_KP    = "kr-vs-kp"
	MUSHROOM    = "mushroom"
	IRIS        = "iris"
	SPAMBASE    = "spambase"
	TIC_TAC_TOE = "tic-tac-toe"
	SICK        = "sick"
	WAVEFORM    = "waveform"
	CAR         = "car"
	VOTE        = "vote"
	IONOSPHERE  = "ionosphere"

@members
class DataModes(enum.Enum):
	TRAIN = 'train'
	TEST  = 'test'
	ALL   = 'all'


@members
class UncertaintyTechniques(enum.Enum):
	STD     = "standard_deviation"
	ENTROPY = "entropy"
	PI      = "predicted_interval"
	BMA     = "bayesian_model_avaraging"
	CBM     = "committee_based_method"
	CP      = "conformal_prediction"


@members
class EvaluationMetricNames(enum.Enum):
	ACC = 'ACC'
	AUC = 'AUC'
	F1  = 'F1'
	THRESHOLD = 'THRESHOLD'


@members
class FindingNames(enum.Enum):
	TRUTH = 'ground_truth'
	PRED  = 'predicted_probabilities'


@members
class OutputModes(enum.Enum):
	CALCULATE   = "calculate"
	LOAD_LOCAL  = "load_local"

@members
class OtherBenchmarkNames(enum.Enum):
	# GLAD = 'GLAD'
	KOS    = 'KOS'
	MACE   = 'MACE'
	MMSR   = 'MMSR'
	WAWA   = 'Wawa'
	ZBS    = 'ZeroBasedSkill'
	MV     = 'MajorityVote'
	DS     = 'DawidSkene'


@members
class MainBenchmarks(enum.Enum):
	TAO	= 'Tao'
	SHENG = 'Sheng'


@members
class ProposedTechniqueNames(enum.Enum):
	PROPOSED = 'proposed'
	PROPOSED_PENALIZED = 'Crowd-Certain'


@members
class StrategyNames(enum.Enum):
	FREQ = 'freq'
	BETA = 'beta'

@members
class ConfidenceScoreNames(enum.Enum):
	ECE   = 'ece score'
	BRIER = 'brier score loss'

def main():
	print(EvaluationMetricNames.AUC in ['AUC', 'ACC'])
	print('something')

if __name__ == '__main__':
	main()

