import argparse
import json
import pathlib
import sys
from typing import Any, TypeAlias, Union
import numpy as np
from typing_extensions import Annotated
from pydantic import BaseModel, confloat, conint, Field
from pydantic.functional_validators import field_validator
import sklearn

from crowd_certain.utilities.params import ConsistencyTechniques, DataModes, DatasetNames, OutputModes, ReadMode, SimulationMethods,UncertaintyTechniques

PathNoneType: TypeAlias = Union[pathlib.Path, None]


class DatasetSettings(BaseModel):
	data_mode         : DataModes          = DataModes.TRAIN
	path_all_datasets : pathlib.Path       = pathlib.Path('datasets')
	dataset_name      : DatasetNames       = DatasetNames.CHESS
	datasetNames      : list[DatasetNames] = Field(default=None)
	non_null_samples  : bool               = True
	random_state      : int                = 0
	read_mode         : ReadMode           = ReadMode.READ_ARFF
	shuffle           : bool               = False
	main_url          : str                = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
	train_test_ratio  : Annotated[float,Field(strict=True, ge=0.0, le=1.0)] = 0.7
	augmentation_count: Annotated[int,  Field(strict=True, ge=0)] 			= 1

	@field_validator('datasetNames', mode='before')
	def check_dataset_names(cls, v: Union[list[DatasetNames], str]):
		if isinstance(v, str):
			if v.lower() == 'all':
				return list(DatasetNames)
			else:
				return [DatasetNames(v.lower())]
		return v


	@field_validator('path_all_datasets', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		path = (pathlib.Path(__file__).parents[1] / v).resolve()
		print(f"------- path_all_datasets{path}")
		return path


# @add_path_validator
class OutputSettings(BaseModel):
	path: pathlib.Path = pathlib.Path('outputs')
	mode: OutputModes = OutputModes.CALCULATE
	save: bool = False

	@field_validator('path', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return (pathlib.Path(__file__).parents[1] / v).resolve()


class TechniqueSettings(BaseModel):
	uncertainty_techniques: list[UncertaintyTechniques] = [UncertaintyTechniques.STD]
	consistency_techniques: list[ConsistencyTechniques] = [ConsistencyTechniques.ONE_MINUS_UNCERTAINTY]

	@field_validator('uncertainty_techniques', mode='before')
	def check_uncertainty_techniques(cls, v: Union[list[UncertaintyTechniques], str]):
		if isinstance(v, str):
			if v.lower() == 'all':
				return list(UncertaintyTechniques)
			else:
				return [UncertaintyTechniques(v.lower())]
		return v

	@field_validator('consistency_techniques', mode='before')
	def check_consistency_techniques(cls, v: Union[list[ConsistencyTechniques], str]):
		if isinstance(v, str):
			if v.lower() == 'all':
				return list(ConsistencyTechniques)
			else:
				return [ConsistencyTechniques(v.lower())]
		return v


class SimulationSettings(BaseModel):
	n_workers_min_max   : list[int] = [3,8]
	high_dis            : float     = 1
	low_dis             : float     = 0.4
	num_simulations     : int       = 10
	num_seeds           : int       = 3
	use_parallelization: bool       = True
	max_parallel_workers: int = 10
	simulation_methods: SimulationMethods = SimulationMethods.RANDOM_STATES

	@property
	def workers_list(self):
		return list(range(*self.n_workers_min_max))

	@property
	def classifiers_list(self):
		return  [ sklearn.neighbors.KNeighborsClassifier(3),
					sklearn.svm.SVC(gamma=2, C=1),
					sklearn.tree.DecisionTreeClassifier(max_depth=5),
					sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
					sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000),
					sklearn.ensemble.AdaBoostClassifier(),
					sklearn.naive_bayes.GaussianNB(),
					sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),
						]


class Settings(BaseModel):

	dataset   : DatasetSettings
	simulation: SimulationSettings
	technique : TechniqueSettings
	output    : OutputSettings

	class Config:
		use_enum_values      = False
		case_sensitive       = False
		str_strip_whitespace = True


def get_settings(argv=None, jupyter=True) -> 'Settings':

	def parse_args() -> dict:
		"""	Getting the arguments from the command line
			Problem:	Jupyter Notebook automatically passes some command-line arguments to the kernel.
						When we run argparse.ArgumentParser.parse_args(), it tries to parse those arguments, which are not recognized by your argument parser.
			Solution:	To avoid this issue, you can modify your get_args() function to accept an optional list of command-line arguments, instead of always using sys.argv.
						When this list is provided, the function will parse the arguments from it instead of the command-line arguments. """

		# If argv is not provided, use sys.argv[1: ] to skip the script name
		args = [] if jupyter else (argv or sys.argv[1:])

		# Initializing the parser
		parser = argparse.ArgumentParser()
		parser.add_argument('--config', type=str, help='Path to config file')

		# Filter out any arguments starting with '-f'
		filtered_argv = [arg for arg in args if not (arg.startswith('-f') or 'jupyter/runtime' in arg.lower())]

		# Parsing the arguments
		parsed_args = parser.parse_args(args=filtered_argv)

		return {k: v for k, v in vars(parsed_args).items() if v is not None}

	def get_config(args_dict: dict[str, Any]) -> Settings:

		# Loading the config.json file
		if args_dict.get('config') is not None:
			config_dir = pathlib.Path(args_dict.get('config'))
		else:
			config_dir = pathlib.Path(__file__).parent.parent / 'config.json'

		if not config_dir.exists():
			raise FileNotFoundError(f'Config file not found at {config_dir}')

		with open(config_dir) as f:
			config_data = json.load(f)

		# Updating the config with the arguments as command line input
		def  update_config(model, config_key):
			for key in model.__fields__:
				if key in args_dict:
					config_data[config_key][key] = args_dict[key]

		# Updating the config with the arguments as command line input
		update_config(DatasetSettings    , 'dataset')
		update_config(SimulationSettings , 'simulation')
		update_config(TechniqueSettings  , 'technique')
		update_config(OutputSettings     , 'output')

		# Convert the dictionary to a Namespace
		config = Settings(**config_data)

		# # Updating the paths to their absolute path
		# PATH_BASE = pathlib.Path(__file__).parent.parent.parent.parent
		# args.DEFAULT_FINDING_FOLDER_NAME = f'{args.datasetName}-{args.modelName}'

		return config

	# Updating the config file
	return  get_config(args_dict=parse_args())


def main():
	config = get_settings()
	print(config.dataset.datasetInfoList)
	print('something')


if __name__ == '__main__':
	main()


