import argparse
import json
import pathlib
import shutil
import sys
from typing import Any, TypeAlias, Union, Tuple
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator
import sklearn

from crowd_certain.utilities.parameters import params
from crowd_certain.config import CONFIG_DIR

PathNoneType: TypeAlias = Union[pathlib.Path, None]


class DatasetSettings(BaseModel):
	data_mode         : params.DataModes          = params.DataModes.TRAIN
	path_all_datasets : pathlib.Path       = pathlib.Path('datasets')
	dataset_name      : params.DatasetNames       = params.DatasetNames.CHESS
	datasetNames      : list[params.DatasetNames] = Field(default=None)
	non_null_samples  : bool               = True
	random_state      : int                = 0
	shuffle           : bool               = False
	main_url          : str                = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
	train_test_ratio  : Annotated[float,Field(strict=True, ge=0.0, le=1.0)] = 0.7
	augmentation_count: Annotated[int,  Field(strict=True, ge=0)] 			= 1

	@field_validator('datasetNames', mode='before')
	def check_dataset_names(cls, v: Union[list[params.DatasetNames], str]):
		if isinstance(v, str):
			if v.lower() == 'all':
				return list(params.DatasetNames)
			else:
				return [params.DatasetNames(v.lower())]
		return v


	@field_validator('path_all_datasets', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		path = (pathlib.Path(__file__).parents[1] / v).resolve()
		return path


class OutputSettings(BaseModel):
	path: pathlib.Path = pathlib.Path('outputs')
	mode: params.OutputModes = params.OutputModes.CALCULATE
	save: bool = False

	@field_validator('path', mode='after')
	def make_path_absolute(cls, v: pathlib.Path):
		return (pathlib.Path(__file__).parents[1] / v).resolve()


class TechniqueSettings(BaseModel):
	uncertainty_techniques: list[params.UncertaintyTechniques] = [params.UncertaintyTechniques.STD]
	consistency_techniques: list[params.ConsistencyTechniques] = [params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY]

	@field_validator('uncertainty_techniques', mode='before')
	def check_uncertainty_techniques(cls, v: Union[list[params.UncertaintyTechniques], str]):
		if isinstance(v, str):
			if v.lower() == 'all':
				return list(params.UncertaintyTechniques)
			else:
				return [params.UncertaintyTechniques(v.lower())]
		return v

	@field_validator('consistency_techniques', mode='before')
	def check_consistency_techniques(cls, v: Union[list[params.ConsistencyTechniques], str]):
		if isinstance(v, str):
			if v.lower() == 'all':
				return list(params.ConsistencyTechniques)
			else:
				return [params.ConsistencyTechniques(v.lower())]
		return v


class SimulationSettings(BaseModel):
	n_workers_min_max   : list[int] = [3,8]
	high_dis            : float     = 1
	low_dis             : float     = 0.4
	num_simulations     : int       = 10
	num_seeds           : int       = 3
	use_parallelization: bool       = True
	max_parallel_workers: int = 10
	simulation_methods: params.SimulationMethods = params.SimulationMethods.RANDOM_STATES

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
	"""Main settings class that contains all configuration settings for the application."""

	dataset   : DatasetSettings
	simulation: SimulationSettings
	technique : TechniqueSettings
	output    : OutputSettings

	class Config:
		use_enum_values      = False
		case_sensitive       = False
		str_strip_whitespace = True

	def save(self, file_path: Union[str, pathlib.Path]) -> None:
		"""
		Save the current configuration to a JSON file.

		Args:
			file_path: Path where the configuration will be saved

		Raises:
			IOError: If the file cannot be written
		"""
		# Convert to a dictionary
		config_dict = self.model_dump()

		# Convert Path objects to strings and handle enums
		def convert_values(d):
			for k, v in d.items():
				if isinstance(v, dict):
					convert_values(v)
				elif isinstance(v, pathlib.Path):
					d[k] = str(v)
				elif isinstance(v, list) and v and hasattr(v[0], 'value'):
					# Handle list of enums
					d[k] = [str(item) for item in v]
				elif hasattr(v, 'value'):
					# Handle enum values
					d[k] = str(v)

		convert_values(config_dict)

		# Ensure the directory exists
		file_path = pathlib.Path(file_path)
		file_path.parent.mkdir(parents=True, exist_ok=True)

		# Write the file
		with open(file_path, 'w') as f:
			json.dump(config_dict, f, indent=4)


class ConfigManager:

	@staticmethod
	def find_config_file(config_path: Union[str, pathlib.Path] = None,
						debug: bool = False,
						find_default_config: bool = False) -> pathlib.Path:
		"""
		Find the config.json file in the project.

		Args:
			config_path: Specific config path to check first (optional)
			debug: Whether to print debug information
			find_default_config: Whether to find the default config file instead of the regular config file

		Returns:
			Path to the config.json file (or config_default.json if find_default_config is True)
			Note: The returned path may not exist if no config file was found

		Raises:
			FileNotFoundError: If no config file is found and raise_error is True
		"""
		# For default config case, return the fixed location
		if find_default_config:
			default_config_path = CONFIG_DIR / "config_default.json"
			if debug:
				print(f"Looking for default config at fixed location: {default_config_path}")
			return default_config_path

		# For regular config.json, continue with the normal search process
		# If a specific config path is provided, check it first
		if config_path is not None:

			# If the provided path is a directory, append the config filename
			config_path = pathlib.Path(config_path)
			if config_path.is_dir():
				config_path = config_path / 'config.json'

			# If it's a file but not named correctly, replace the name
			elif config_path.name != 'config.json':
				config_path = config_path.with_name('config.json')

			if config_path.exists():
				if debug:
					print(f"Using specified config file at: {config_path}")
				return config_path

			elif debug:
				print(f"Specified config file not found at: {config_path}")

		# Check multiple possible locations for the config file
		possible_locations = [
				# Config directory using CONFIG_DIR
			CONFIG_DIR / 'config.json',

			# Current directory
			pathlib.Path.cwd() / 'config.json',

			# Main project directory (crowd_certain)
			pathlib.Path(__file__).parents[1] / 'config.json',

			# Utilities directory
			pathlib.Path(__file__).parent / 'config.json',

			# One level up from current directory
			pathlib.Path.cwd().parent / 'config.json',

			# Two levels up from current directory
			pathlib.Path.cwd().parent.parent / 'config.json'
		]

		# Check each location and print debug info
		for location in possible_locations:
			if location.exists():
				if debug:
					print(f"Found config file at: {location}")
				return location
			elif debug:
				print(f"Config file not found at: {location}")

		# If no config file is found, return a default path
		default_location = CONFIG_DIR / 'config.json'
		if debug:
			print(f"No config file found. Using default location: {default_location}")

		return default_location

	@staticmethod
	def revert_to_default_config(config_path: Union[str, pathlib.Path] = None,
								debug: bool = False) -> Tuple[bool, pathlib.Path]:
		"""
		Find the default config file and copy it to the config location.

		Args:
			config_path: Path where the config should be copied to (optional)
			debug: Whether to print debug information

		Returns:
			Tuple of (success, path):
				- success: Boolean indicating whether the operation was successful
				- path: Path where the config was copied to or should have been copied to
		"""
		# Get the default config file from the fixed location using CONFIG_DIR
		default_config_path = CONFIG_DIR / "config_default.json"
		if debug:
			print(f"Using default config from fixed location: {default_config_path}")

		# Determine the target path for the config file
		if config_path is None:
			# Use the CONFIG_DIR for the config file
			config_path = CONFIG_DIR / 'config.json'
		else:
			config_path = pathlib.Path(config_path)
			if config_path.is_dir():
				config_path = config_path / 'config.json'

		# Ensure the directory exists
		config_path.parent.mkdir(parents=True, exist_ok=True)

		# Check if the default config exists
		if not default_config_path.exists():
			if debug:
				print(f"Default config file not found at: {default_config_path}")
			return False, config_path

		# Copy the default config to the target location
		try:
			shutil.copy2(default_config_path, config_path)
			if debug:
				print(f"Successfully copied default config from {default_config_path} to {config_path}")
			return True, config_path
		except Exception as e:
			if debug:
				print(f"Error copying default config: {str(e)}")
			return False, config_path


	@staticmethod
	def get_settings(argv=None, jupyter=True, debug=False) -> Settings:
		"""
		Get application settings from command line arguments and/or config file.

		Args:
			argv: Command line arguments (optional)
			jupyter: Whether the function is being called from a Jupyter notebook
			debug: Whether to print debug information

		Returns:
			Settings object with all configuration parameters

		Raises:
			FileNotFoundError: If the config file is not found
		"""
		def parse_args() -> dict:
			"""
			Parse command line arguments, handling Jupyter notebook special cases.

			Returns:
				Dictionary of parsed arguments
			"""
			# If argv is not provided, use sys.argv[1:] to skip the script name
			# For Jupyter notebooks, use an empty list to avoid parsing Jupyter's arguments
			args = [] if jupyter else (argv or sys.argv[1:])

			# Print the arguments for debugging
			if args and debug:
				print(f"Command line arguments: {args}")

			# Initialize the parser
			parser = argparse.ArgumentParser()
			parser.add_argument('--config', type=str, help='Path to config file')

			# Filter out Jupyter-specific arguments
			filtered_argv = [arg for arg in args if not (arg.startswith('-f') or 'jupyter/runtime' in arg.lower())]

			# Parse the arguments
			parsed_args = parser.parse_args(args=filtered_argv)

			# Return only non-None values
			result = {k: v for k, v in vars(parsed_args).items() if v is not None}

			# Print the parsed arguments for debugging
			if result and debug:
				print(f"Parsed arguments: {result}")

			return result

		def get_config(args_dict: dict[str, Any]) -> Settings:
			"""
			Load and update configuration from file and command line arguments.

			Args:
				args_dict: Dictionary of command line arguments

			Returns:
				Settings object with all configuration parameters

			Raises:
				FileNotFoundError: If the config file is not found
				ValueError: If the config file contains invalid JSON
			"""
			# Load the config.json file
			if args_dict.get('config') is not None:
				config_path = args_dict.get('config')
			else:
				config_path = None

			# Find the config file
			config_path = ConfigManager.find_config_file(config_path=config_path, debug=debug)

			# If the config file doesn't exist, try to revert to the default config
			if not config_path.exists():
				success, config_path = ConfigManager.revert_to_default_config(config_path=config_path, debug=debug)
				if not success or not config_path.exists():
					raise FileNotFoundError(f'Config file not found at {config_path} and could not revert to default config')

			if debug:
				print(f"Loading configuration from: {config_path}")

			try:
				with open(config_path) as f:
					config_data = json.load(f)
			except json.JSONDecodeError as e:
				raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
			except Exception as e:
				raise IOError(f"Error reading config file {config_path}: {e}")

			# Validate the config data structure
			required_sections = ['dataset', 'simulation', 'technique', 'output']
			missing_sections = [section for section in required_sections if section not in config_data]
			if missing_sections:
				raise ValueError(f"Config file is missing required sections: {', '.join(missing_sections)}")

			# Update the config with command line arguments
			def update_config(model_class, config_key):
				"""Update config data with command line arguments for a specific section."""
				if config_key not in config_data:
					config_data[config_key] = {}

				for key in model_class.__annotations__:
					if key in args_dict:
						config_data[config_key][key] = args_dict[key]

			# Update each section of the config
			update_config(DatasetSettings    , 'dataset')
			update_config(SimulationSettings , 'simulation')
			update_config(TechniqueSettings  , 'technique')
			update_config(OutputSettings     , 'output')

			# Create the Settings object
			try:
				return Settings(**config_data)
			except Exception as e:
				raise ValueError(f"Error creating Settings object from config file {config_path}: {e}")

		# Get and return the configuration
		return get_config(args_dict=parse_args())


def main():
	config = ConfigManager.get_settings()
	print(config.dataset.datasetInfoList)
	print('something')


if __name__ == '__main__':
	main()


