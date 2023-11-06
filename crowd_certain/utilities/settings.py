import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field, InitVar
from typing import Any, TypeAlias, Union

from pydantic import BaseModel, confloat, conint, Field, FieldValidationInfo
from pydantic.functional_validators import field_validator

from crowd_certain.utilities.params import DataModes, DatasetNames, SimulationOptions, ReadMode

PathNoneType: TypeAlias = Union[pathlib.Path, None]


class DatasetSettings(BaseModel):
	data_mode         : DataModes           = DataModes.TRAIN
	path_all_datasets: pathlib.Path         = pathlib.Path('../datasets')
	datasetNames      : list[DatasetNames]  = None
	non_null_samples  : bool                = True
	train_test_ratio  : confloat(ge=0,le=1) = 0.7
	random_state      : int                 = 0
	read_mode         : ReadMode            = ReadMode.READ_ARFF
	shuffle: bool = False
	augmentation_count: conint(ge   = 0) = 1


	@field_validator('path_all_datasets', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return v.resolve()

	# @field_validator('datasetInfoList', mode='before')
	# def post_process_info(cls, v: None, info: FieldValidationInfo) -> list[DatasetInfo]:
	# 	return [ DatasetInfo(   path_all_datasets = info.data['path_all_datasets'],
	# 							data_mode         = info.data['data_mode'],
	# 							views 			  = info.data['views'],
	# 							datasetName       = dt )
	# 			 for dt in info.data['datasetNames']]

class SimulationSettings(BaseModel):
	nlabelers_min_max   : list[float] = [3,8],
	high_dis            : int         = 1,
	low_dis             : int         = 0.4,
	num_simulations     : int         = 10,
	num_seeds           : int         = 3,
	use_parallelization: bool         = True

class OutputSettings(BaseModel):
	path: pathlib.Path = pathlib.Path('../outputs')
	mode: SimulationOptions = SimulationOptions.CALCULATE
	save: bool = False

	@field_validator('path', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return v.resolve()


class Settings(BaseModel):

	dataset                    : DatasetSettings
	simulation                 : SimulationSettings
	output                     : OutputSettings

	class Config:
		use_enum_values      = False
		case_sensitive       = False
		str_strip_whitespace = True


def get_settings(argv=None, jupyter=True, config_path='config.json') -> 'Settings':

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

	def get_config(args_dict: dict) -> Union[Settings,ValueError]:

		# Loading the config.json file
		config_dir = pathlib.Path( args_dict.get('config') or config_path ).resolve()

		if not config_dir.exists():
			return ValueError(f'Config file not found at {config_dir}')

		with open(config_dir) as f:
			config_data = json.load(f)

		# Updating the config with the arguments as command line input
		def  update_config(model, config_key):
			for key in model.__fields__:
				if key in args_dict:
					config_data[config_key][key] = args_dict[key]

		# Updating the config with the arguments as command line input
		update_config(DatasetSettings             , 'dataset')
		update_config(SimulationSettings          , 'simulation')
		update_config(OutputSettings              , 'output')

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
