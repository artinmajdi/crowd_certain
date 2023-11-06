from __future__ import annotations
import enum


def members(cls):

	# Add the members class method
	cls.members = classmethod(lambda cls2: list(cls2.__members__))

	# Adding all options in a list
	cls.all = classmethod(lambda cls2: [cls2[n] for n in list(cls2.__members__)])

	# cls.values = classmethod(lambda cls2: [n.value for n in cls2.__members__])

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
class EvaluationMetricNames(enum.Enum):
	ACC = 'acc'
	AUC = 'auc'
	F1  = 'f1'
	THRESHOLD = 'threshold'


@members
class FindingNames(enum.Enum):
	TRUTH = 'ground_truth'
	PRED  = 'predicted_probabilities'


@members
class OutputModes(enum.Enum):
	CALCULATE   = "calculate"
	LOAD_LOCAL  = "load_local"


def main():
	print(EvaluationMetricNames.AUC in ['AUC', 'ACC'])
	print('something')

if __name__ == '__main__':
	main()
