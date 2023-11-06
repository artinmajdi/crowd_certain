import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field, InitVar
from typing import Any, TypeAlias, Union, Optional, Type

from pydantic import BaseModel, confloat, conint, Field, FieldValidationInfo, validator, root_validator
from pydantic.functional_validators import field_validator

from crowd_certain.utilities.params import DataModes, DatasetNames, OutputModes, ReadMode

PathNoneType: TypeAlias = Union[pathlib.Path, None]


# class CommonBaseModel(BaseModel):
#     # Define the absolute path adjustment in a root_validator
#     @root_validator(pre=True)
#     def make_path_absolute(cls, values):
#         for field_name, model_field in cls.__fields__.items():
#             # Check if the field type is pathlib.Path
#             if issubclass(model_field.type_, pathlib.Path):
#                 # Check if the field is in values and is a Path (to prevent overwriting non-Path values)
#                 if field_name in values and isinstance(values[field_name], pathlib.Path):
#                     # Adjust the path to be absolute
#                     values[field_name] = pathlib.Path(__file__).parents[1] / values[field_name]
#         return values


# def add_path_validator(model_class: Type[BaseModel]) -> Type[BaseModel]:
#     # Define a classmethod inside the decorator
#     @classmethod
#     def make_path_absolute(cls, v: pathlib.Path):
#         if not v.is_absolute():
#             return pathlib.Path(__file__).parents[1] / v
#         return v

#     # Dynamically add validators for all pathlib.Path fields
#     for field_name, field_type in model_class.__annotations__.items():
#         if field_type == pathlib.Path:
#             # The name of the validator is composed to be unique for each field
#             validator_name = f"validate_{field_name}"
#             setattr(
#                 model_class,
#                 validator_name,
#                 validator(field_name, allow_reuse=True)(make_path_absolute)
#             )

#     return model_class

# @add_path_validator
class DatasetSettings(BaseModel):
	data_mode         : DataModes           = DataModes.TRAIN
	path_all_datasets: pathlib.Path         = pathlib.Path('datasets')
	dataset_name      : DatasetNames = None
	datasetNames      : list[DatasetNames]  = None
	non_null_samples  : bool                = True
	train_test_ratio  : confloat(ge=0.0,le=1.0) = 0.7
	random_state      : int                 = 0
	read_mode         : ReadMode            = ReadMode.READ_ARFF
	shuffle: bool = False
	augmentation_count: conint(ge   = 0) = 1
	main_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

	@field_validator('path_all_datasets', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return (pathlib.Path(__file__).parents[1] / v).resolve()

# @add_path_validator
class OutputSettings(BaseModel):
	path: pathlib.Path = pathlib.Path('outputs')
	mode: OutputModes = OutputModes.CALCULATE
	save: bool = False

	@field_validator('path', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return (pathlib.Path(__file__).parents[1] / v).resolve()


class SimulationSettings(BaseModel):
	nlabelers_min_max   : list[int] = [3,8]
	high_dis            : float       = 1
	low_dis             : float       = 0.4
	num_simulations     : int         = 10
	num_seeds           : int         = 3
	use_parallelization: bool         = True

	@property
	def workers_list(self):
		return list(range(*self.nlabelers_min_max))




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
