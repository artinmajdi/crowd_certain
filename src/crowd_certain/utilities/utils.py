# Standard library imports
import functools
import multiprocessing
import stat
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple,
                    TypeAlias, TypedDict, Union, get_args)

# Third-party imports
import numpy as np
import pandas as pd
import scipy
from scipy.special import bdtrc
from sklearn import ensemble as sk_ensemble
from sklearn import metrics as sk_metrics

# Local imports
from crowd_certain.utilities.benchmarks import BenchmarkTechniques
from crowd_certain.utilities.io import dataset_loader
from crowd_certain.utilities.io.dataset_loader import LoadSaveFile
from crowd_certain.utilities.io.hdf5_storage import HDF5Storage
from crowd_certain.utilities.parameters import params
from crowd_certain.utilities.parameters.params import (
    ClassifierPredsDFType, ConsistencyTechniques, ConsistencyTechniqueType,
    DataMode, UncertaintyTechniques, WorkerID, WorkerLabelsDFType,
    WorkerReliabilitiesSeriesType)
from crowd_certain.utilities.parameters.settings import Settings

USE_HD5F_STORAGE = True


class ClassifierTraining:
	def __init__(self, config: 'Settings'):
		self.config = config
		# Pre-initialize the classifiers_list to avoid repeated imports
		self._classifiers_list = None

	def train_classifier(self, sim_num: int):
		pass

	def get_classifier(self, simulation_number: int, random_state: int=0):
		"""
		Returns a classifier based on the simulation method configuration.

		For params.SimulationMethods.RANDOM_STATES:
			Returns a RandomForestClassifier with fixed parameters but varying random_state
			for different simulation numbers.

		For params.SimulationMethods.MULTIPLE_CLASSIFIERS:
			Returns a classifier from the pre-configured list of classifiers based on
			the simulation number.

		Returns:
			sklearn estimator: A classifier instance configured according to the simulation method.
		"""
		if self.config.simulation.simulation_methods is params.SimulationMethods.RANDOM_STATES:
			return self.get_random_forest_ensemble(random_state=random_state, simulation_number=simulation_number)

		elif self.config.simulation.simulation_methods is params.SimulationMethods.MULTIPLE_CLASSIFIERS:
			return self.classifiers_list[simulation_number % self.n_classifiers]

		raise ValueError(f"Invalid simulation method: {self.config.simulation.simulation_methods}")

	@property
	def n_classifiers(self) -> int:
		return len(self.classifiers_list)

	@property
	def n_simulations(self) -> int:
		if self.config.simulation.simulation_methods is params.SimulationMethods.RANDOM_STATES:
			return self.config.simulation.num_simulations

		elif self.config.simulation.simulation_methods is params.SimulationMethods.MULTIPLE_CLASSIFIERS:
			return self.n_classifiers

		raise ValueError(f"Invalid simulation method: {self.config.simulation.simulation_methods}")

	def get_random_forest_ensemble(self, random_state: int, simulation_number: int):
		return sk_ensemble.RandomForestClassifier(
			n_estimators=4,
			max_depth=4,
			random_state=random_state * simulation_number,
			n_jobs=-1  # Use all available processors for parallel training
		)

	@property
	def classifiers_list(self):
		# Lazy initialization of classifiers to avoid repeated imports and improve startup time
		if self._classifiers_list is None:
			from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
			from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
			from sklearn.naive_bayes import GaussianNB
			from sklearn.neighbors import KNeighborsClassifier
			from sklearn.neural_network import MLPClassifier
			from sklearn.svm import SVC
			from sklearn.tree import DecisionTreeClassifier

			self._classifiers_list = {
				0: KNeighborsClassifier(3, n_jobs=-1),  # Add parallel processing
				1: SVC(gamma=2, C=1, probability=True),
				2: DecisionTreeClassifier(max_depth=5),
				3: RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
				4: MLPClassifier(alpha=1, max_iter=1000),
				5: AdaBoostClassifier(),
				6: GaussianNB(),
				7: QuadraticDiscriminantAnalysis()}

		return self._classifiers_list


@dataclass
class CrowdCertainOrchestrator:
	"""
	Main class for CrowdCertainOrchestrator implementation that handles the calculation of worker weights, uncertainties, and confidence scores
	for crowdsourced label aggregation.

	This class implements methods for:
	- Calculating uncertainties using different techniques (standard deviation, entropy, coefficient of variation, etc.)
	- Converting uncertainties to consistency scores
	- Calculating weights for proposed techniques and benchmark methods
	- Measuring accuracy metrics for different aggregation techniques
	- Simulating worker strengths and noisy labels
	- Generating confidence scores for aggregated labels

	Attributes:
		data (dict): Dictionary containing training and test data
		feature_columns (list): List of feature column names
		config (Settings): Configuration object containing simulation parameters
		n_workers (int): Number of workers in the simulation
		seed (int): Random seed for reproducibility

	Key Methods:
		measuring_nu_and_confidence_score(): Calculates confidence scores and aggregated labels
		core_measurements(): Main method that orchestrates the calculation pipeline
		calculate_one_dataset(): Entry point for running simulations on a single dataset
		calculate_all_datasets(): Entry point for running simulations on multiple datasets
	"""
	data               : dict
	feature_columns    : list
	config             : 'Settings'
	classifier_training: Optional[ClassifierTraining] = None
	n_workers          : int = 3
	seed               : int = 0

	def __post_init__(self):
		self.classifier_training = ClassifierTraining(config=self.config)
		np.random.seed(self.seed + 1)


	@staticmethod
	def measuring_nu_and_confidence_score(self, weights: params.WeightType, preds_all, true_labels, use_parallelization_benchmarks: bool=False) -> Tuple[params.ResultType, params.ResultType]:
		"""Calculates the confidence scores (nu) and evaluation metrics for proposed methods and benchmarks.

		This function computes the confidence scores and aggregated labels using the provided weights
		for both the proposed techniques and benchmark methods (Tao and Sheng). It also calculates
		evaluation metrics (AUC, Accuracy, and F1 score) for all methods.

			weights (params.WeightType): Object containing weights for proposed methods and benchmarks.
			preds_all: Dictionary containing workers' predictions for different datasets and methods.
			true_labels: Ground truth labels for evaluation.
			use_parallelization_benchmarks (bool, optional): Whether to use parallelization for benchmark calculation.
				Defaults to False.

			Tuple[params.ResultType, params.ResultType]: A tuple containing:
				- results_proposed: Results for proposed methods (aggregated labels, confidence scores, metrics)
				- results_benchmarks: Results for benchmark methods (aggregated labels, confidence scores, metrics)
		"""
		def get_results_proposed(preds: pd.DataFrame) -> params.ResultType:
			"""
			Returns:
				F: pd.DataFrame(columns = pd.MultiIndex.from_product([params.ConsistencyTechniques, UncertaintyTechniques,ProposedTechniqueNames]),
								index   = pd.MultiIndex.from_product([StrategyNames, data_indices])

				aggregated_labels: pd.DataFrame(columns = pd.MultiIndex.from_product([params.ConsistencyTechniques, UncertaintyTechniques,ProposedTechniqueNames])
												index   = data_indices)
			"""

			agg_labels = pd.DataFrame(columns=weights.PROPOSED.index, index=preds.index)

			index = pd.MultiIndex.from_product([ ['F', 'F_pos'], params.StrategyNames.values(), preds.index ], names=['F_F_pos', 'strategies', 'indices'])
			confidence_scores = {}

			metrics = pd.DataFrame(columns=weights.PROPOSED.index, index=params.EvaluationMetricNames.values())

			for cln in weights.PROPOSED.index:
				agg_labels[cln] = (preds * weights.PROPOSED.T[cln]).sum(axis=1)

				confidence_scores[cln] = CalculateConfidenceScores(delta=preds, w=weights.PROPOSED.T[cln], n_workers=self.n_workers ).calculate_confidence_scores()

				# Measuring the metrics
				metrics[cln] = CalculateMetrics.get_AUC_ACC_F1(aggregated_labels=agg_labels[cln], truth=true_labels['test'].truth)

			return params.ResultType(aggregated_labels=agg_labels, confidence_scores=confidence_scores, metrics=metrics)

		def get_results_tao_sheng(workers_labels) -> params.ResultType:
			"""Calculating the weights for Tao and Sheng methods

			Args:
				workers_labels (pd.DataFrame): pd.DataFrame(columns=[worker_0, ...], index=data_indices)

			Returns:
				Tuple[pd.DataFrame, pd.DataFrame]:
				F 							 = pd.DataFrame(columns = pd.MultiIndex.from_product([params.MainBenchmarks, params.StrategyNames] , index = workers_labels.index)
				aggregated_labels_benchmarks = pd.DataFrame(columns = params.MainBenchmarks 											 , index = workers_labels.index)
			"""

			def get_v_F_Tao_sheng() -> Tuple[pd.DataFrame, dict[str, pd.DataFrame]]:

				def initialize():
					nonlocal agg_labels, confidence_scores

					confidence_scores = {}

					agg_labels = pd.DataFrame(index=workers_labels.index, columns=params.MainBenchmarks.values())
					return agg_labels, confidence_scores

				agg_labels, confidence_scores = initialize()

				for m in params.MainBenchmarks:

					w = weights.TAO if m is params.MainBenchmarks.TAO else weights.SHENG

					agg_labels[m.value] = (workers_labels * w).sum(axis=1)

					confidence_scores[m.value] = CalculateConfidenceScores( delta=workers_labels, w=w, n_workers=self.n_workers ).calculate_confidence_scores()

				return agg_labels, confidence_scores

			v_benchmarks, F_benchmarks = get_v_F_Tao_sheng()

			v_other_benchmarks = BenchmarkTechniques.apply(true_labels=true_labels, use_parallelization_benchmarks=use_parallelization_benchmarks)

			v_benchmarks = pd.concat( [v_benchmarks, v_other_benchmarks], axis=1)

			# Measuring the metrics
			metrics_benchmarks = pd.DataFrame({cln: CalculateMetrics.get_AUC_ACC_F1(aggregated_labels=v_benchmarks[cln], truth=true_labels['test'].truth) for cln in v_benchmarks.columns})

			return params.ResultType(aggregated_labels=v_benchmarks, confidence_scores=F_benchmarks, metrics=metrics_benchmarks)

		# Getting confidence scores and aggregated labels for proposed techniques
		results_proposed   = get_results_proposed(  preds_all['test']['mv'] )
		results_benchmarks = get_results_tao_sheng( preds_all['test']['simulation_0'] )

		return results_proposed, results_benchmarks


	@classmethod
	def core_measurements(cls, data, n_workers, config, feature_columns, seed, use_parallelization_benchmarks=False) -> params.Result2Type:
		""" Final pred labels & uncertainties for proposed technique
				dataframe = preds[train, test] * [mv] <=> {rows: samples, columns: workers}
				dataframe = uncertainties[train, test]  {rows: samples, columns: workers}

			Final pred labels for proposed benchmarks
				dataframe = preds[train, test] * [simulation_0] <=> {rows: samples,  columns: workers}

			Note: A separate boolean flag is used for use_parallelization_benchmarks.
			This will ensure we only apply parallelization when called through worker_weight_strength_relation function """

		# TODO: change the data['train'] and data['test'] to have two elements: truth and feature_columns. instead of passing the feature_columns everytime

		aim1_3 = cls(data=data, config=config, feature_columns=feature_columns, n_workers=n_workers, seed=seed)

		# calculating uncertainty and predicted probability values
		workers_reliabilities, workers_labels, uncertainties, consistencies, classifiers_preds = ReliabilityCalculator.generate(config=config, n_workers=n_workers, seed=seed, data=data, feature_columns=feature_columns)


		workers_accruacies = CalculateMetrics.adding_accuracy_for_each_worker(n_workers=n_workers, preds=classifiers_preds, workers_labels=workers_labels, workers_assigned_reliabilities=workers_reliabilities)

		# Calculating weights for proposed techniques and TAO & Sheng benchmarks
		weights = CalculateWeights.get_weights( classifiers_preds = classifiers_preds,
												uncertainties 	  = uncertainties,
												consistencies 	  = consistencies,
												workers_labels	  = workers_labels,
												n_workers      	  = n_workers,
												config         	  = config)

		# weights = CalculateWeights.get_weights( classifiers_preds = classifiers_preds, # ['test']['simulation_0']
		# 										preds 			  = classifiers_preds['test']['mv'],
		# 										uncertainties 	  = uncertainties,
		# 										consistencies 	  = consistencies,
		# 										noisy_true_labels = workers_labels,
		# 										n_workers		  = n_workers,
		# 										config			  = config)

		# Calculating results for proposed techniques and benchmarks
		results_proposed, results_benchmarks = aim1_3.measuring_nu_and_confidence_score(weights     = weights,
																						preds_all   = classifiers_preds,
																						true_labels = workers_labels,
																						use_parallelization_benchmarks = use_parallelization_benchmarks)

		# merge workers_strength and weights
		# merge_worker_strengths_and_weights()

		return params.Result2Type(  proposed 	   	   = results_proposed,
									benchmark          = results_benchmarks,
									weight             = weights,
									workers_reliabilities  = workers_reliabilities,
									workers_accruacies = workers_accruacies,
									n_workers          = n_workers,
									true_label         = workers_labels )


	@staticmethod
	def wrapper(args: Dict, config: Settings, data, feature_columns) -> Tuple[Dict, params.Result2Type]:
		return args, CrowdCertainOrchestrator.core_measurements(data=data, n_workers=args['nl'], config=config, feature_columns=feature_columns, seed=args['seed'], use_parallelization_benchmarks=False)


	@staticmethod
	def objective_function(config: Settings, data, feature_columns) -> Callable[[Dict], Tuple[Dict, params.Result2Type]]:
		return functools.partial(CrowdCertainOrchestrator.wrapper, config=config, data=data, feature_columns=feature_columns)


	def get_outputs(self) -> Dict[str, List[params.ResultType]]:
		"""
		Retrieves the outputs for the calculation or loads them from file based on the current configuration.

		This method operates in two modes dictated by the configuration:

		1. CALCULATE mode:
			- Constructs an input list of parameter dictionaries, associating each worker ('nl') with a set of seeds.
			- Defines an objective function using the given configuration, data, and feature columns.
			- If parallelization is enabled, computes results in parallel using a Pool of workers; otherwise, computes results sequentially.
			- Aggregates the results into an outputs dictionary where each key (formatted as "NL{nl}")
				holds a list of results corresponding to each seed. Results are stored at the index of the seed
				in the list. Missing results are represented as NaN.
			- If output saving is enabled in the configuration, stores the outputs using HDF5Storage.
			- Returns the populated outputs dictionary.

		2. Non-CALCULATE mode:
			- Loads the outputs directly from the specified file path using HDF5Storage.

		Returns:
			Dict[str, List[params.ResultType]]:
				A dictionary mapping output keys to lists containing the results for each seed.
		"""
		if USE_HD5F_STORAGE:
			path = self.config.output.path / 'outputs' / f'{self.config.dataset.dataset_name}.h5'
			h5s = HDF5Storage(path)
		else:
			path = self.config.output.path / 'outputs' / f'{self.config.dataset.dataset_name}.pkl'

		def update_outputs() -> Dict[str, List[params.ResultType]]:
			"""
			Update and return the simulation outputs based on the current core results.

			This function constructs an output dictionary where each key follows the format 'NL{nl}'
			for each worker as specified in the simulation configuration. For each worker, it initializes
			a list of length `num_seeds` filled with NaN values using numpy's np.full. It then updates
			the list by replacing the NaN at the index corresponding to the seed (as provided in the
			core_results tuple) with the computed result value.

			If the configuration flag to save outputs is enabled (self.config.output.save), the
			outputs dictionary is stored using HDF5Storage.

			Returns:
				Dict[str, List[params.ResultType]]: A dictionary mapping worker identifiers to lists
				of results updated for each seed.
			"""
			nonlocal core_results
			# Initialize output structure with NaN values
			outputs = {f'NL{nl}': np.full(self.config.simulation.num_seeds, np.nan).tolist() for nl in self.config.simulation.workers_list}

			for args, values in core_results:
				outputs[f'NL{args["nl"]}'][args['seed']] = values

			if self.config.output.save:
				if USE_HD5F_STORAGE:
					h5s.save(data=outputs, group_path='/outputs')
				else:
					LoadSaveFile(path).dump(outputs)

			return outputs

		if self.config.output.mode is params.OutputModes.CALCULATE:

			input_list = [{'nl': nl, 'seed': seed} for nl in self.config.simulation.workers_list for seed in range(self.config.simulation.num_seeds)]

			function = CrowdCertainOrchestrator.objective_function(config=self.config, data=self.data, feature_columns=self.feature_columns)

			if self.config.simulation.use_parallelization:
				with multiprocessing.Pool(processes=min(self.config.simulation.max_parallel_workers, len(input_list))) as pool:
					core_results = pool.map(function, input_list)

			else:
				core_results = [function(args) for args in input_list]

			return update_outputs()

		if USE_HD5F_STORAGE:
			return h5s.load(f'/{self.config.dataset.dataset_name}/outputs')
		return LoadSaveFile(path).load()


	def worker_weight_strength_relation(self, seed=0, n_workers=10) -> pd.DataFrame:
		"""
		Calculate and retrieve the relationship between worker strength and weights.

		This method either calculates the relationship between worker strength and weights
		using the core_measurements function, or loads it from a saved file, depending on
		the configuration output mode.

		Parameters:
		-----------
		seed : int, default=0
			Random seed for reproducibility

		n_workers : int, default=10
			Number of workers to simulate

		Returns:
		--------
		pd.DataFrame
			DataFrame containing the worker strength-weight relationship
		"""
		metric_name = 'weight_strength_relation'
		path_main = self.config.output.path / metric_name / str(self.config.dataset.dataset_name)
		if USE_HD5F_STORAGE:
			h5s = HDF5Storage(path_main / f'{metric_name}.h5')
		else:
			lsf = LoadSaveFile(path_main / f'{metric_name}.xlsx')

		if self.config.output.mode is params.OutputModes.CALCULATE:
			params_dict = {
						'seed'                          : seed,
						'n_workers'                     : n_workers,
						'config'                        : self.config,
						'data'                          : self.data,
						'feature_columns'               : self.feature_columns,
						'use_parallelization_benchmarks': self.config.simulation.use_parallelization}

			df = CrowdCertainOrchestrator.core_measurements(**params_dict).workers_reliabilities.set_index('workers_strength').sort_index()

			if self.config.output.save:
				if USE_HD5F_STORAGE:
					h5s.save_dataframe(df, '/weight_strength_relation')
				else:
					lsf.dump(df, index=True)
			return df

		if USE_HD5F_STORAGE:
			return h5s.load_dataframe('/weight_strength_relation')
		return lsf.load(header=0)

	"""
	def get_confidence_scores(self, outputs) -> dict[str, dict[StrategyNames, dict[str, pd.DataFrame]]]:
		# TODO need to fix this function according to new changes.

		path_main = self.config.output.path / f'confidence_score/{self.config.dataset.dataset_name}'

		DICT_KEYS = ['F_all', 'F_pos_all', 'F_mean_over_seeds', 'F_pos_mean_over_seeds']

		def get_Fs_per_nl_per_strategy(strategy, n_workers) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

			def get(stuff: str) -> pd.DataFrame:
				seeds_list   = list(range(self.config.simulation.num_seeds))
				methods_list = list(ProposedTechniqueNames) + list(MainBenchmarks)
				columns      = pd.MultiIndex.from_product([methods_list, seeds_list], names=['method', 'seed_ix'])
				df = pd.DataFrame(columns=columns)

				for (m, sx) in columns:
					df[(m, sx)] = outputs[n_workers][sx].F[strategy][m][stuff].squeeze()

				return df

			inF     = get( 'F' )
			inF_pos = get( 'F_pos' )

			# inF_mean_over_seeds     = inF.T.groupby(level=0).mean().T
			# inF_pos_mean_over_seeds = inF_pos.T.groupby(level=0).mean().T

			techniques_list         = inF.columns.get_level_values(0)
			inF_mean_over_seeds     = pd.DataFrame({m: inF[m].mean(axis=1)     for m in techniques_list})
			inF_pos_mean_over_seeds = pd.DataFrame({m: inF_pos[m].mean(axis=1) for m in techniques_list})

			return inF, inF_pos, inF_mean_over_seeds, inF_pos_mean_over_seeds


		if self.config.output.mode is OutputModes.CALCULATE:

			F_dict = { key : {StrategyNames.FREQ:{}, StrategyNames.BETA:{}} for key in DICT_KEYS }

			for st in StrategyNames:
				for nl in [f'NL{x}' for x in self.config.simulation.workers_list]:

					F, F_pos, F_mean_over_seeds, F_pos_mean_over_seeds = get_Fs_per_nl_per_strategy( strategy=st, n_workers=nl )

					F_dict['F_all' 			 	  ][st][nl] = F.copy()
					F_dict['F_pos_all' 			  ][st][nl] = F_pos.copy()
					F_dict['F_mean_over_seeds'	  ][st][nl] = F_mean_over_seeds.copy()
					F_dict['F_pos_mean_over_seeds'][st][nl] = F_pos_mean_over_seeds.copy()


			if self.config.output.save:
				for name in DICT_KEYS:
					LoadSaveFile(path_main / f'{name}.pkl').dump(F_dict[name])

			return F_dict

		return { key: LoadSaveFile(path_main / f'{key}.pkl').load()  for key in DICT_KEYS }
	"""

	@classmethod
	def calculate_one_dataset(cls, config: Settings, dataset_name: params.DatasetNames=None) -> params.ResultComparisonsType:
		"""
		Calculates various output metrics and analyses for a given dataset using the provided configuration.

		This method updates the configuration to specify the desired dataset, loads the corresponding data,
		computes model outputs using the instance created with the input data and configuration, and then
		evaluates the worker weight strength relationship. The results, along with the outputs and configuration,
		are encapsulated in a params.ResultComparisonsType object that is returned to the caller.

		Parameters:
			config (Settings): Configuration settings that include parameters, dataset details, and other experiment configurations.
			dataset_name (params.DatasetNames, optional): The identifier for the dataset to be processed.
				Defaults to params.DatasetNames.IONOSPHERE.

		Returns:
			params.ResultComparisonsType: An object containing:
				- weight_strength_relation: The computed relationship between worker weights and strength.
				- outputs: The outputs produced by the model after running the dataset.
				- config: The configuration that was used for the computation.

		Note:
			The method internally loads the dataset using the UCI database reader,
			creates an instance for further analysis, and computes the required metrics.
		"""

		if dataset_name is not None:
			config.dataset.dataset_name = dataset_name

		# Check if results already exist in HDF5 storage and should be loaded
		if USE_HD5F_STORAGE and config.output.mode is params.OutputModes.LOAD_LOCAL:
			h5_path = config.output.path / 'results' / f'{config.dataset.dataset_name}.h5'
			h5s = HDF5Storage(h5_path)
			result = h5s.load_result_comparisons(config.dataset.dataset_name.value)
			if result is not None:
				return result

		# Loading the dataset
		data, feature_columns = dataset_loader.load_dataset(config=config)

		aim1_3 = cls(data=data, feature_columns=feature_columns, config=config)

		# Getting the outputs
		outputs = aim1_3.get_outputs()

		# measuring the confidence scores
		# findings_confidence_score = aim1_3.get_confidence_scores(outputs)

		# Measuring the worker strength weight relationship for proposed and Tao
		weight_strength_relation = aim1_3.worker_weight_strength_relation(seed=0, n_workers=10)

		if not USE_HD5F_STORAGE:
			return params.ResultComparisonsType(weight_strength_relation=weight_strength_relation, outputs=outputs, config=config)

		# Create result object
		result = params.ResultComparisonsType(
			weight_strength_relation=weight_strength_relation,
			outputs=outputs,
			config=config
		)

		# Save result to HDF5 if configured
		if config.output.save:
			h5_path = config.output.path / 'results' / f'{config.dataset.dataset_name}.h5'
			h5s = HDF5Storage(h5_path)
			h5s.save_result_comparisons(result, config.dataset.dataset_name.value)

		return result

	@classmethod
	def calculate_all_datasets(cls, config: 'Settings') -> Dict[params.DatasetNames, params.ResultComparisonsType]:
		"""
		Calculate results for all datasets specified in the configuration.

		This method processes each dataset in the configuration's dataset list,
		computing and returning results for each one. Results are stored in HDF5
		format for efficient storage and retrieval.

		Parameters:
		-----------
		config : Settings
			Configuration settings containing parameters, dataset details, and
			other experiment configurations.

		Returns:
		--------
		Dict[params.DatasetNames, params.ResultComparisonsType]
			A dictionary mapping dataset names to their respective result objects.

		Notes:
		------
		If config.output.mode is set to LOAD, the method will attempt to load
		existing results from HDF5 files. If set to CALCULATE, it will compute
		new results and save them if config.output.save is True.
		"""
		if not USE_HD5F_STORAGE:
			return {dt: cls.calculate_one_dataset(dataset_name=dt, config=config) for dt in config.dataset.datasetNames}

		# Check if we should load all results from a combined HDF5 file
		if config.output.mode is params.OutputModes.LOAD_LOCAL:
			h5_path = config.output.path / 'results' / 'all_datasets.h5'
			h5s = HDF5Storage(h5_path)
			results = h5s.load_all_datasets_results(config.dataset.datasetNames)
			if results and len(results) == len(config.dataset.datasetNames):
				return results

		# Calculate results for each dataset
		results = {}
		for dt in config.dataset.datasetNames:
			results[dt] = cls.calculate_one_dataset(dataset_name=dt, config=config)

		# Save all results together if configured
		if config.output.save:
			h5_path = config.output.path / 'results' / 'all_datasets.h5'
			h5s = HDF5Storage(h5_path)
			h5s.save_all_datasets_results(results)

		return results


class CalculateConfidenceScores:
	def __init__(self, delta: pd.DataFrame, w: Union[pd.DataFrame, pd.Series], n_workers: int):
		self.delta = delta
		self.w = w
		self.n_workers = n_workers

	def get_freq(self, P_pos: pd.Series, P_neg: pd.Series) -> pd.DataFrame:
		"""
		Calculates frequency metrics based on positive and negative probability values.

		This function creates a DataFrame with positive and negative probabilities,
		then returns a new DataFrame containing:
		- 'F': The maximum probability value between P_pos and P_neg for each entry
		- 'F_pos': The positive probability values

		Returns:
			pandas.DataFrame: A DataFrame with 'F' and 'F_pos' columns, where:
				- 'F' contains the maximum probability value for each entry
				- 'F_pos' contains the positive probability values
		"""
		out = pd.DataFrame({'P_pos':P_pos,'P_neg':P_neg})
		return pd.DataFrame({'F': out.max(axis=1), 'F_pos': P_pos})

	def get_beta(self, P_pos: pd.Series, P_neg: pd.Series, n_workers: int) -> pd.DataFrame:
		"""
		Calculate beta values based on positive and negative probabilities.

		This function computes various beta-related statistics from the positive and negative
		probability values (P_pos and P_neg). It calculates parameters for a beta distribution
		including alpha (l_alpha) and beta (u_beta), then computes the incomplete beta function
		values (I) using the bdtrc function.

		Returns:
			pandas.DataFrame: A DataFrame containing two columns:
				- F: Maximum value between I and (1-I) for each row
				- F_pos: The original I values

		Note:
			This function assumes that P_pos, P_neg, n_workers, and bdtrc are defined
			in the outer scope.
		"""
		out = pd.DataFrame({'P_pos':P_pos,'P_neg':P_neg})

		out['l_alpha'] = 1 + out['P_pos'] * n_workers
		out['u_beta']  = 1 + out['P_neg'] * n_workers

		out['k'] = out['l_alpha'] - 1

		# This seems to be equivalent to n_workers + 1
		out['n'] = ((out['l_alpha'] + out['u_beta']) - 1) #.astype(int)


		def get_I(row):
			return bdtrc(row['k'], row['n'], 0.5)
		out['I'] = out.apply(get_I, axis=1)

		def get_F_lambda(row):
			return max(row['I'], 1-row['I'])

		return pd.DataFrame({'F': out.apply(get_F_lambda, axis=1), 'F_pos': out['I']})

	def calculate_confidence_scores(self) -> pd.DataFrame:
		"""
		Calculate confidence scores for each item using multiple strategies.

		This function computes confidence scores for binary classifications based on worker responses
		and their weights. It implements two strategies: frequency-based and beta distribution-based.

		Parameters
		----------
		delta : pd.DataFrame
			Binary matrix of worker responses where each row represents an item and each column
			represents a worker's classification (True/False)
		w : Union[pd.DataFrame, pd.Series]
			Worker weights used to compute weighted sum of responses.
			Shape should match the columns of delta.
		n_workers : int
			Number of workers who provided classifications

		Returns
		-------
		pd.DataFrame
			A DataFrame with multi-level columns containing confidence scores:
			- First level: Strategy names (FREQ, BETA)
			- Second level: 'F' (overall confidence score) and 'F_pos' (confidence in positive class)

		Notes
		-----
		The frequency strategy simply calculates the weighted sum of positive and negative responses.
		The beta strategy models the confidence using a beta distribution and calculates the probability
		mass using the bdtrc function.
		"""

		P_pos: pd.Series = ( self.delta * self.w).sum( axis=1 )
		P_neg: pd.Series = (~self.delta * self.w).sum( axis=1 )


		columns = pd.MultiIndex.from_product([params.StrategyNames.values(), ['F', 'F_pos']], names=['strategies', 'F_F_pos'])
		confidence_scores = pd.DataFrame(columns=columns, index=self.delta.index)
		confidence_scores[params.StrategyNames.FREQ.value] = self.get_freq(P_pos=P_pos, P_neg=P_neg)
		confidence_scores[params.StrategyNames.BETA.value] = self.get_beta(P_pos=P_pos, P_neg=P_neg, n_workers=self.n_workers)

		return confidence_scores


class CalculateWeights:
	"""
	Class for calculating weights for different methods (proposed, TAO, and SHENG).

	Parameters
	----------
	config : Settings
		Configuration object containing uncertainty settings.
	n_workers : int
		Number of workers.
	"""
	def __init__(self, delta: pd.DataFrame, noisy_true_labels: pd.DataFrame, n_workers: int, config: 'Settings'):
		self.delta = delta
		self.noisy_true_labels = noisy_true_labels
		self.n_workers = n_workers
		self.config = config
		self.weights = None


	@staticmethod
	def measuring_Tao_weights_based_on_classifier_labels(delta: pd.DataFrame, noisy_true_labels: pd.DataFrame) -> pd.DataFrame:
		"""
		Calculates normalized Tao weights based on classifier labels.

		This function computes weights for worker responses in a crowdsourcing scenario
		using the agreement between worker responses and the classifier's estimated true labels.

		Parameters
		----------
		delta : pandas.DataFrame
			Worker responses with shape (num_samples, n_workers), where each cell contains
			a binary response (True/False) or (1/0) from a worker for a specific sample.

		noisy_true_labels : pandas.DataFrame or Series
			The estimated true labels from a classifier with shape (num_samples, 1).

		Returns
		-------
		pandas.DataFrame
			Normalized Tao weights with shape (num_samples, n_workers), where each value
			represents the weight assigned to a worker's response for a specific sample.
			Higher weights indicate higher estimated reliability for that worker-sample pair.

		Notes
		-----
		The algorithm calculates:
		- tau      : Average agreement between worker responses and estimated true labels
		- gamma    : Worker-specific quality metrics that account for agreement with other workers
		- W_hat_Tao: Raw Tao weights before normalization
		- z        : Normalizing factors
		- Final weights are W_hat_Tao normalized by z for each sample

			tau          : 1 thresh_technique 1
			weights_Tao  : num_samples thresh_technique n_workers
			W_hat_Tao    : num_samples thresh_technique n_workers
			z            : num_samples thresh_technique 1
			gamma        : num_samples thresh_technique 1
		"""

		tau = (delta == noisy_true_labels).mean(axis=0)

		# number of workers
		M = len(delta.columns)

		# number of true and false labels for each class and sample
		true_counts = delta.sum(axis=1)
		false_counts = M - true_counts

		# measuring the "specific quality of instanses"
		s = delta.multiply(true_counts - 1, axis=0) + (~delta).multiply(false_counts - 1, axis=0)
		gamma = (1 + s ** 2) * tau
		W_hat_Tao = gamma.apply(lambda x: 1 / (1 + np.exp(-x)))
		z = W_hat_Tao.mean(axis=1)

		return W_hat_Tao.divide(z, axis=0)


	@staticmethod
	def measuring_Tao_weights_based_on_actual_labels(workers_labels: pd.DataFrame, noisy_true_labels: pd.DataFrame, n_workers: int) -> pd.DataFrame:
		"""
		Calculate worker weights based on their accuracy relative to actual labels.

		This function evaluates how well each worker's labels align with the noisy true labels
		and computes weights that give more importance to workers who are more accurate.
		The weights are normalized so that they sum to 1 for each sample.

		Parameters:
		-----------
		workers_labels : pandas.DataFrame
			DataFrame where each row represents a sample and each column represents a worker's labels.
			Values should be boolean or binary (0/1).

		noisy_true_labels : pandas.DataFrame
			DataFrame containing the noisy ground truth labels for each sample.
			Should have the same row indices as workers_labels.

		n_workers : int
			Number of workers.

		Returns:
		--------
		pandas.DataFrame
			Normalized weights for each worker for each sample.
			Shape: (num_samples, n_workers)

		Notes:
		------
		The function computes:
		- tau: Worker accuracy (mean agreement with true labels)
		- gamma: Weighted worker quality considering sample difficulty
		- W_hat_Tao: Logistic activation of gamma
		- Final weights: Normalized W_hat_Tao divided by n_workers
			tau          : 1 thresh_technique 1
			weights_Tao  : num_samples thresh_technique n_workers
			W_hat_Tao    : num_samples thresh_technique n_workers
			z            : num_samples thresh_technique 1
			gamma        : num_samples thresh_technique 1
		"""

		tau = (workers_labels == noisy_true_labels).mean(axis=0)

		# number of workers
		M = len(noisy_true_labels.columns)

		# number of true and false labels for each class and sample
		true_counts = noisy_true_labels.sum(axis=1)
		false_counts = M - true_counts

		# measuring the "specific quality of instanses"
		s = noisy_true_labels.multiply(true_counts - 1, axis=0) + (~noisy_true_labels).multiply(false_counts - 1, axis=0)
		gamma = (1 + s ** 2) * tau
		W_hat_Tao = gamma.apply(lambda x: 1 / (1 + np.exp(-x)))
		z = W_hat_Tao.mean(axis=1)

		return W_hat_Tao.divide(z, axis=0) / n_workers


	@staticmethod
	def get_weights(config: Settings, n_workers:int,
					classifiers_preds: Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]],
					uncertainties    : Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]],
					consistencies    : Dict[WorkerID, Dict[DataMode, Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]]],
					workers_labels   : Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]] ):
		"""
		Calculate weights for different methods (proposed, TAO, and SHENG).

		Parameters
		----------
		workers_labels : pandas.DataFrame
			Matrix of labels assigned by workers to items.
		preds : pandas.Series or numpy.ndarray
			Predictions (estimated true labels).
		uncertainties : pandas.Series or numpy.ndarray
			Uncertainty values associated with predictions.
		noisy_true_labels : pandas.Series or numpy.ndarray
			Ground truth labels (possibly with noise).
		n_workers : int
			Number of workers who provided labels.

		Returns
		-------
		params.WeightType
			Named tuple containing weights for three methods:
			- PROPOSED: Weights calculated using the proposed technique based on predictions and uncertainties
			- TAO: Weights calculated based on Tao's method using actual labels
			- SHENG: Equal weights (1/n_workers) for all worker-item pairs
		"""

		# Measuring weights for the proposed technique
		weights_proposed = CalculateWeights.measuring_proposed_weights( config=config, preds=preds, uncertainties=uncertainties)

		# Benchmark accuracy measurement
		weights_Tao = CalculateWeights.measuring_Tao_weights_based_on_actual_labels( workers_labels=classifiers_preds, noisy_true_labels=workers_labels, n_workers=n_workers )

		weights_Sheng = pd.DataFrame(1 / n_workers, index=weights_Tao.index, columns=weights_Tao.columns)

		return params.WeightType(PROPOSED=weights_proposed, TAO=weights_Tao, SHENG=weights_Sheng)


	@staticmethod
	def measuring_proposed_weights(config: Settings, preds: pd.DataFrame, uncertainties: pd.DataFrame) -> pd.DataFrame:

		# TODO This is the part where I should measure the prob_mv_binary for different # of workers instead of all of them
		prob_mv_binary = preds.mean(axis=1) > 0.5

		T1    = ReliabilityCalculator.calculate_consistency( uncertainty=uncertainties, config=config )
		T2    = T1.copy()

		proposed_techniques = [l.value for l in params.ProposedTechniqueNames]
		w_hat = pd.DataFrame(index=proposed_techniques, columns=T1.columns, dtype=float)

		for worker in preds.columns:

			T2.loc[preds[worker].values != prob_mv_binary.values, worker] = 0

			w_hat[worker] = pd.DataFrame.from_dict({proposed_techniques[0]: T1[worker].mean(axis=0),
													proposed_techniques[1]: T2[worker].mean(axis=0)}, orient='index')


		# measuring average weight over all workers. used to normalize the weights.
		w_hat_mean_over_workers = w_hat.T.groupby(level=[1,2]).sum().T


		weights = pd.DataFrame().reindex_like(w_hat)
		for worker in preds.columns:
			weights[worker] = w_hat[worker].divide(w_hat_mean_over_workers)


		# This will return a series
		weights = weights.unstack().unstack(level='worker')
		return weights


class ReliabilityCalculator:
	"""
	Class for calculating various uncertainty metrics.

	This class provides methods to calculate different uncertainty metrics for a given DataFrame.
	The metrics include standard deviation, entropy, coefficient of variation, prediction interval,
	and confidence interval.

	Parameters
	----------
	config : Settings
		Configuration object containing uncertainty settings.
	n_workers : int
		Number of workers.
	seed : int
		Random seed for reproducibility.
	data : Dict[DataMode, pd.DataFrame]
		Dictionary containing training and test data.
	feature_columns : list[str]
		List of feature columns to be used for training classifiers.
	"""
	def __init__(self, config: 'Settings', n_workers: int, seed: int, data: Dict[DataMode, pd.DataFrame], feature_columns: list[str]):
		self.config = config
		self.n_workers = n_workers
		self.seed = seed
		self.data = data
		self.feature_columns = feature_columns
		# Initialize empty data structures with type hints
		self.workers_reliabilities: WorkerReliabilitiesSeriesType = pd.Series()
		self.workers_preds     : Dict[WorkerID, Dict[DataMode, pd.DataFrame]]                                                        = {}
		self.uncertainties     : Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]]                              = {}
		self.consistencies     : Dict[WorkerID, Dict[DataMode, Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]]] = {}
		self.classifiers_preds : Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]]                                               = {}
		self.workers_labels    : Dict[DataMode, WorkerLabelsDFType]                                                                  = {}

	@staticmethod
	def calculate_uncertainty(df: pd.DataFrame, config: 'Settings') -> Dict[UncertaintyTechniques, pd.Series]:
		"""
		Calculate uncertainty metrics for the given DataFrame based on configuration settings.

		This function computes various uncertainty measures across rows of data, where each row
		represents a sample and each column represents a different simulation's prediction.

		Parameters
		----------
		df : pd.DataFrame
			Input DataFrame containing integer values, where each row is a sample for one worker and
			each column represents a different simulation's prediction.
		config : Settings
			Configuration object containing settings for uncertainty calculation,
			particularly the list of uncertainty techniques to apply.

			Returns
			-------
		Dict[UncertaintyTechniques, pd.Series]
			A dictionary mapping uncertainty techniques to Series of uncertainty values.
			Available uncertainty techniques include:
				- STD: Standard deviation across annotations
				- ENTROPY: Normalized Shannon entropy of the annotations
				- CV: Coefficient of variation (std/mean), normalized with tanh to [0,1]
				- PI: Prediction interval (difference between 75th and 25th percentiles)
				- CI: 95% confidence interval width using t-distribution

		Raises
			------
		ValueError
			If the input DataFrame has fewer than two columns.

		Notes
		-----
			The function handles edge cases with zero values by adding a small epsilon to denominators
			and log arguments to avoid division by zero or undefined logarithms.
		"""
		# Initialize result dictionary
		df_uncertainties: Dict[UncertaintyTechniques, pd.Series] = {}

		# Validate input
		if len(df.columns) <= 1:
			raise ValueError("Input DataFrame must have at least two columns.")
			return df_uncertainties

		# Small constant to avoid division by zero
		epsilon = np.finfo(float).eps

		# Convert to binary values (0 or 1)
		df_binary = (df > 0.5).astype(int)

		# Calculate requested uncertainty metrics
		for tech in config.technique.uncertainty_techniques:
			if tech is params.UncertaintyTechniques.STD:
				# Standard deviation across annotations
				df_uncertainties[tech] = df_binary.std(axis=1)

			elif tech is params.UncertaintyTechniques.ENTROPY:
				# Normalize each row to sum to 1
				df_normalized = df_binary.div(df_binary.sum(axis=1) + epsilon, axis=0)
				# Calculate entropy with logarithm base e
				entropy = -(df_normalized * np.log(df_normalized + epsilon)).sum(axis=1)
				# Normalize entropy to [0, 1] by dividing by log(n)
				df_uncertainties[tech] = entropy / np.log(df_binary.shape[1])

			elif tech is params.UncertaintyTechniques.CV:
				# Coefficient of variation: std/mean, normalized with tanh to bound [0, 1]
				cv = df_binary.std(axis=1) / (df_binary.mean(axis=1) + epsilon)
				df_uncertainties[tech] = np.tanh(cv)

			elif tech is params.UncertaintyTechniques.PI:
				# Prediction interval: difference between 75th and 25th percentiles
				q75 = np.percentile(df_binary, 75, axis=1)
				q25 = np.percentile(df_binary, 25, axis=1)
				df_uncertainties[tech] = q75 - q25

			elif tech is params.UncertaintyTechniques.CI:
				# Calculate 95% confidence interval width using t-distribution
				stds = df_binary.std(axis=1)
				n = df_binary.shape[1]
				t_critical = scipy.stats.t.ppf(0.975, n-1)  # 95% CI uses 0.975 (two-tailed)
				# Calculate margin of error
				margin = t_critical * stds / np.sqrt(n)
				# Calculate CI width (upper - lower)
				ci_width = 2 * margin
				# Handle cases with zero standard deviation
				df_uncertainties[tech] = ci_width.fillna(0)

		return df_uncertainties


	@classmethod
	def generate(cls, config: 'Settings', n_workers: int, data: Dict[DataMode, pd.DataFrame],
			  feature_columns: list[str], seed: int) -> tuple[
				WorkerReliabilitiesSeriesType,
				Dict[DataMode, WorkerLabelsDFType],
				Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]],
				Dict[WorkerID, Dict[DataMode, Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]]],
				Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]]
			]:
		"""
		Generate worker reliabilities, labels, uncertainties, consistencies, and classifier predictions.

		This class method creates an instance of ReliabilityCalculator and uses it to generate
		all the necessary data for worker reliability analysis.

		Parameters
		----------
		config : Settings
			Configuration object containing settings for the reliability calculation.
		n_workers : int
			Number of workers to generate.
		data : Dict[DataMode, pd.DataFrame]
			Dictionary containing training and test data.
		feature_columns : list[str]
			List of feature columns to be used for training classifiers.
		seed : int
			Random seed for reproducibility.

		Returns
		-------
		tuple
			A tuple containing:
			- workers_reliabilities: Series of worker reliability scores
			- workers_labels: Dictionary of worker labels for each data mode
			- uncertainties: Dictionary of uncertainty metrics for each worker
			- consistencies: Dictionary of consistency metrics for each worker
			- classifiers_preds: Dictionary of classifier predictions for each worker
		"""
		# Create an instance of the class
		RC = cls(config=config, n_workers=n_workers, seed=seed,
									data=data, feature_columns=feature_columns)

		# Generate worker reliabilities and labels
		workers_reliabilities, workers_labels = WorkersGeneration.generate(
			config=config, data=data, n_workers=n_workers)

		# Calculating classifiers predictions for all workers
		classifiers_preds = RC.calculate_classifiers_preds_for_all_workers( data=data, config=config, feature_columns=feature_columns, workers_labels=workers_labels, seed=seed)

		# Calculating uncertainties for all workers
		uncertainties = RC.calculate_uncertainties_for_all_workers(classifiers_preds_all_workers=classifiers_preds, config=config)

		# Calculating consistency
		consistencies = RC.calculate_consistencies_for_all_workers(uncertainties=uncertainties, config=config)

		return workers_reliabilities, workers_labels, uncertainties, consistencies, classifiers_preds


	@staticmethod
	def calculate_classifiers_preds_for_all_workers(data: Dict[DataMode, pd.DataFrame],
												config: 'Settings',
												feature_columns: list[str],
												workers_labels: Dict[DataMode, WorkerLabelsDFType],
												seed: int) -> Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]]:
		"""
		Predict labels for both training and test data using trained classifiers.

		This function trains a classifier for each worker based on the simulation method
		specified in the configuration and uses it to predict labels for both training
		and test datasets.

		Parameters
			----------
			data : Dict[DataMode, pd.DataFrame]
				The dataset containing training and test data.
		config : Settings
				The configuration object containing simulation settings.
			feature_columns : list[str]
				List of feature columns to be used for training classifiers.
			workers_labels : Dict[DataMode, WorkerLabelsDFType]
				Dictionary containing ground truth labels for each worker.
			seed : int
				Random seed for classifier training.

		Returns
		-------
				Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]]
			Dictionary containing predicted labels for both training and test data,
			organized by worker ID and data mode.
		"""
		def calculate_classifiers_preds_for_one_worker(worker_name: str) -> Dict[DataMode, ClassifierPredsDFType]:
			"""
			Calculate classifier predictions for a single worker.

			Parameters
			----------
			worker_name : str
				The name/ID of the worker.

			Returns
			-------
			Dict[DataMode, ClassifierPredsDFType]
				Dictionary containing predictions for train and test data.
			"""
			# Initialize output dataframes with the same index as the input data
			output = {mode: pd.DataFrame(index=data[mode].index) for mode in get_args(DataMode)}

			# Run simulations
			for sim_num in range(config.simulation.num_simulations):

				# training a random forest on the aformentioned labels
				classifier = ClassifierTraining(config=config).get_classifier(simulation_number=sim_num, random_state=seed)

				classifier.fit( X=data['train'][feature_columns], y=workers_labels['train'][workers_name] )

				# This assumes the outputs of the classifier are pandas Series
				output['train'][ f'simulation_{sim_num}' ] = classifier.predict(data['train'][feature_columns])
				output['test'][  f'simulation_{sim_num}' ] = classifier.predict(data['test'][feature_columns])

			return output

		# Getting the list of worker identifiers (excluding 'truth')
		workers_names = set(workers_labels['train'].columns).difference({'truth'})

		# Calculate predictions for each worker
		return {worker: calculate_classifiers_preds_for_one_worker(worker_name=worker)
				for worker in worker_names}


	@staticmethod
	def calculate_consistencies_for_all_workers(uncertainties: Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]],
												config: 'Settings') -> Dict[WorkerID, Dict[DataMode, Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]]]:
		"""
		Calculate consistency metrics for all workers based on uncertainty values.

		This function transforms uncertainty metrics into consistency metrics using
		various techniques specified in the configuration.

		Parameters
		----------
		uncertainties : Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]]
			Dictionary containing uncertainty metrics for each worker, data mode, and uncertainty technique.
		config : Settings
			Configuration object containing settings for consistency calculation.

		Returns
		-------
		Dict[WorkerID, Dict[DataMode, Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]]]
			Dictionary containing consistency metrics for each worker, data mode, consistency technique,
			and uncertainty technique.
		"""
		def calculate_consistency_metrics(uncertainty: Dict[UncertaintyTechniques, pd.Series]) -> Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]:
			"""
			Calculate consistency metrics for a single set of uncertainty values.

			Parameters
			----------
			uncertainty : Dict[UncertaintyTechniques, pd.Series]
				Dictionary mapping uncertainty techniques to their corresponding values.

			Returns
			-------
			Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]
				Dictionary of consistency metrics organized by consistency technique and uncertainty technique.
			"""
			# Initialize nested dictionary structure
			consistency: Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]] = {ct: {} for ct in config.technique.consistency_techniques}

			for ut in uncertainty.keys():
				for ct in config.technique.consistency_techniques:

					if ct is params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY:
						consistency[ct][ut] = 1 - uncertainty[ut]

					elif ct is params.ConsistencyTechniques.ONE_DIVIDED_BY_UNCERTAINTY:
						consistency[ct][ut] = 1 / (uncertainty[ut] + np.finfo(float).eps)

			return consistency

		consistensies: Dict[WorkerID, Dict[DataMode, Dict[ConsistencyTechniques, Dict[UncertaintyTechniques, pd.Series]]]] = {}

		for wsi in uncertainties.keys():
			consistensies[wsi] = { mode: calculate(uncertainty=uncertainties[wsi][mode], config=config)
									for mode in get_args(DataMode)}

		return consistensies


	@staticmethod
	def calculate_uncertainties_for_all_workers(classifiers_preds_all_workers: Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]],
												config: 'Settings') -> Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]]:
		"""
		Calculate uncertainties for all workers based on classifier predictions across simulations.

		This function computes uncertainty metrics for each worker by processing their predictions
		from multiple simulations, for both training and testing datasets.

		Parameters
		----------
		classifiers_preds_all_workers : Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]]
			A nested dictionary structure containing classifier predictions for all simulations.
			The hierarchy is: {worker_id -> {data_mode -> DataFrame}},
			where data_mode is either 'train' or 'test'.

		config : Settings
			Configuration object containing settings for uncertainty calculation.

		Returns
		-------
		Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]]
			A dictionary mapping worker IDs to their uncertainty metrics for both training and testing data.
			Structure: {worker_id -> {data_mode -> {uncertainty_technique -> pd.Series}}}
		"""

		uncertainties: Dict[WorkerID, Dict[DataMode, Dict[UncertaintyTechniques, pd.Series]]] = {}

		# Loop over each worker
		for worker_id in classifiers_preds_all_workers:

			# uncertainties for each worker over all simulations
			uncertainties[worker_id] = {
			mode: ReliabilityCalculator.calculate_uncertainty(df=classifiers_preds_all_workers[worker_id][mode], config=config)
						for mode in get_args(DataMode)}

		return uncertainties


class CalculateMetrics:

	def __init__(self, config: 'Settings'):
		self.config = config


	@staticmethod
	def adding_accuracy_for_each_worker(n_workers: int, preds: Dict[WorkerID, Dict[DataMode, ClassifierPredsDFType]], workers_labels: Dict[DataMode, WorkerLabelsDFType], workers_assigned_reliabilities: WorkerReliabilitiesSeriesType) -> pd.DataFrame:
		"""
		Calculate and add accuracy measures for each worker in the crowd-sourcing task.

		This function computes two types of accuracy metrics for each worker:
		1. 'accuracy-test-classifier': The accuracy of the classifier's predictions against true labels
			in the test dataset for simulation_0
		2. 'accuracy-test': The accuracy of each worker's noisy labels against true labels in the
			test dataset

		The function updates the workers_strength DataFrame with these new accuracy metrics.

		Note:
			- This is a non-local function that modifies the outer scope variables: workers_strength, preds, and truth
		"""
		workers_accruacies: pd.DataFrame = workers_assigned_reliabilities.to_frame(name='reliability')
		workers_accruacies['accuracy-test-classifier'] = 0.0
		workers_accruacies['accuracy-test'] = 0.0

		for wsi in preds.keys():

			# accuracy of classifier in simulation_0
			workers_accruacies.loc[wsi, 'accuracy-test-classifier'] = ( preds[wsi]['test']['simulation_0'] == workers_labels['test'].truth).mean()

			# accuracy of noisy true labels for each worker
			workers_accruacies.loc[wsi, 'accuracy-test'] 		   = ( workers_labels['test'][wsi] == workers_labels['test'].truth).mean()

		return workers_accruacies


	@staticmethod
	def get_accuracy(aggregated_labels: pd.DataFrame, n_workers: int, delta_benchmark: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
		"""
		Calculates the accuracy of various crowdsourcing aggregation methods by comparing
		their predictions against ground truth.

		Parameters
		----------
		aggregated_labels : pd.DataFrame
			DataFrame containing the aggregated labels from different aggregation methods.
			Each column represents a different method, and each row represents an item.

		n_workers : int
			The number of workers used in the crowdsourcing task.

		delta_benchmark : pd.DataFrame
			Binary predicted labels from workers. Used to calculate majority voting accuracy.
			Rows are items and columns are workers.

		truth : pd.Series or array-like
			Ground truth labels for each item.

		Returns
		-------
		pd.DataFrame
			DataFrame with accuracy scores for each aggregation method.
			The index is the number of workers and columns are the different methods.

		Notes
		-----
		The function evaluates accuracy for three groups of methods:
		- params.ProposedTechniqueNames: Custom proposed techniques
		- params.MainBenchmarks: Main benchmark methods
		- params.OtherBenchmarkNames: Additional benchmark methods
		- MV_Classifier: Majority voting accuracy

		Accuracy is calculated by thresholding aggregated labels at 0.5 and
		comparing with ground truth.
		"""

		accuracy = pd.DataFrame(index=[n_workers])
		for methods in [params.ProposedTechniqueNames, params.MainBenchmarks, params.OtherBenchmarkNames]:
			for m in methods:
				accuracy[m] = ((aggregated_labels[m] >= 0.5) == truth).mean(axis=0)

		accuracy['MV_Classifier'] = ( (delta_benchmark.mean(axis=1) >= 0.5) == truth).mean(axis=0)

		return accuracy


	@staticmethod
	def get_AUC_ACC_F1(aggregated_labels: pd.Series, truth: pd.Series) -> pd.Series:

		metrics = pd.Series(index=params.EvaluationMetricNames.values())

		non_null = ~truth.isnull()
		truth_notnull = truth[non_null].to_numpy()

		if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):

			yhat = (aggregated_labels > 0.5).astype(int)[non_null]
			metrics[params.EvaluationMetricNames.AUC.value] = sk_metrics.roc_auc_score( truth_notnull, yhat)
			metrics[params.EvaluationMetricNames.ACC.value] = sk_metrics.accuracy_score(truth_notnull, yhat)
			metrics[params.EvaluationMetricNames.F1.value ] = sk_metrics.f1_score( 	    truth_notnull, yhat)

		return metrics


class WorkersGeneration:
	"""
	A class for generating simulated worker labels and reliabilities for crowdsourcing tasks.

	This class handles the creation of synthetic worker annotations by:
	1. Assigning random reliability scores to workers
	2. Generating noisy label sets based on those reliabilities
	3. Managing the worker label generation process

	Attributes:
		config (Settings)                         : Configuration object containing simulation parameters
		n_workers (int)                           : Number of workers to simulate
		workers_assigned_reliabilities (pd.Series): Worker reliability scores indexed by worker IDs
		workers_labels (Dict[Literal['train', 'test'], pd.DataFrame])  : Worker labels for train and test sets

	Methods:
		assign_reliabilities_to_each_worker() : Assigns random reliability scores to workers
		create_worker_label_set(data)        : Creates noisy label sets based on worker reliabilities
		generate(config, data, n_workers)    : Class method to generate worker labels and reliabilities
	"""

	def __init__(self, config: 'Settings', n_workers: int) -> None:
		self.config = config
		self.n_workers = n_workers
		self.workers_assigned_reliabilities: WorkerReliabilitiesSeriesType | None = None
		self.workers_labels: Dict[DataMode, WorkerLabelsDFType] | None = None

	def assign_reliabilities_to_each_worker(self) -> WorkerReliabilitiesSeriesType:
		"""
		Assigns random reliability scores to each worker in the crowdsourcing task.

		This method generates random reliability scores for each worker by sampling from a uniform
		distribution between the low_dis and high_dis values specified in the configuration.
		These scores represent each worker's reliability or accuracy in providing labels.

		Returns
		-------
		WorkerReliabilities
			Returns the reliability scores for each worker.

		Notes
		-----
		The reliability scores are stored in self.workers_assigned_reliabilities as a pandas Series
		indexed by worker IDs ('worker_0', 'worker_1', etc.).

		The reliability range is defined by:
		- config.simulation.low_dis: Lower bound of worker reliability
		- config.simulation.high_dis: Upper bound of worker reliability
		"""

		# Generating a random number between low_dis and high_dis for each worker
		workers_reliabilities_array = np.random.uniform(low=self.config.simulation.low_dis, high=self.config.simulation.high_dis, size=self.n_workers)

		# Assigning reliabilities to each worker
		self.workers_assigned_reliabilities = pd.Series(workers_reliabilities_array, index=[f'worker_{j}' for j in range(self.n_workers)])

		return self.workers_assigned_reliabilities

	def create_worker_label_set(self, data: Dict[DataMode, pd.DataFrame]) -> Dict[DataMode, WorkerLabelsDFType]:
		"""
		Creates simulated worker label sets based on ground truth and worker reliability.

		This method generates synthetic worker annotations by introducing controlled noise
		to the ground truth labels based on each worker's assigned reliability. The process
		simulates how human annotators with varying skill levels would label the same data.

		Parameters
		----------
		data : Dict[DataMode, pd.DataFrame]
			Dictionary containing training and testing DataFrames with ground truth labels
			in a column named 'true'. Expected keys are 'train' and 'test'.

		Returns
		-------
		Dict[DataMode, WorkerLabelsDFType]
			Returns the worker labels for each data mode.

		The method:
		1. Takes each sample's ground truth and creates noisy versions based on worker reliability
		2. Stores the generated worker labels in self.workers_labels dictionary
		3. Lower worker reliability results in more label errors

		The resulting self.workers_labels is a dictionary with keys 'train' and 'test',
		where each value is a DataFrame containing the ground truth and all workers' labels.

		Notes
		-----
		The output structure represents worker labels in the following format:
		{
			'train' | 'test': pd.DataFrame(
				columns = ['truth', worker_name1, worker_name2, ...],
				index   = data.index,
				values  = worker_labels
			)
		}
		"""

		def getting_noisy_manual_labels_for_each_worker(truth_array: np.ndarray, worker_reliability: float) -> np.ndarray:
			"""
			Simulates noisy manual labeling process by introducing randomized errors to true labels.

			This function takes a ground truth array and simulates the behavior of an annotator with
			a specified strength (accuracy) level. The annotator will correctly label samples with a
			probability equal to their reliability (w_strength), and incorrectly label them otherwise.

			Parameters
			----------
			truth_array : np.ndarray
				Array of ground truth values (typically binary or probabilities that can be thresholded).
				Shape is (num_samples,).
			worker_reliability : float
				Worker/annotator reliability as a probability between 0 and 1.
				Higher values indicate more reliable workers who make fewer mistakes.

			Returns
			-------
			np.ndarray
				Binary array of the same shape as truth_array containing the worker's noisy labels.
				1s represent positive class assignments, 0s represent negative class assignments.

			Notes
			-----
			The function works by:
			1. Determining which samples will be incorrectly labeled based on worker reliability
			2. Converting truth_array to binary labels (thresholding at 0.5)
			3. Flipping the binary labels for samples identified as incorrectly labeled
			"""

			# number of samples and workers/workers
			num_samples = truth_array.shape[0]

			# finding a random number for each instance
			true_label_assignment_prob = np.random.random(num_samples)

			# samples that will have an inaccurate true label
			false_samples = true_label_assignment_prob < 1 - worker_reliability

			# measuring the new labels for each worker/worker
			worker_truth = truth_array > 0.5
			worker_truth[false_samples] = ~ worker_truth[false_samples]

			return worker_truth

		def do_generate(mdata: pd.DataFrame) -> pd.DataFrame:
			"""
			Generate workers' labels based on ground truth and worker reliability.

			This function creates a DataFrame with the ground truth and noisy labels for each worker,
			based on the worker's reliability.

			Parameters
			----------
			mdata : pd.DataFrame
				Input DataFrame containing ground truth labels in the 'true' column.

			Returns
			-------
			pd.DataFrame
				DataFrame with columns for the 'truth' and each worker's labels.
				The index is preserved from the input DataFrame.
			"""
			if self.workers_assigned_reliabilities is None:
				raise ValueError("Worker reliabilities must be assigned before creating label sets")

			workers_labels = pd.DataFrame({'truth': mdata.true}, index=mdata.index)

			for wsi, worker_reliability in self.workers_assigned_reliabilities.items():
				workers_labels[wsi] = getting_noisy_manual_labels_for_each_worker(truth_array=mdata.true.to_numpy(), worker_reliability=worker_reliability)

			return workers_labels

		self.workers_labels: Dict[DataMode, WorkerLabelsDFType] = {mode: do_generate(mdata=data[mode]) for mode in get_args(DataMode)}

		return self.workers_labels

	@classmethod
	def generate(cls, config: 'Settings', data: Dict[DataMode, pd.DataFrame], n_workers: int) -> Tuple[WorkerReliabilitiesSeriesType, Dict[DataMode, WorkerLabelsDFType]]:
		"""
		Generate simulated worker labels and reliabilities for crowdsourcing tasks.

		This class method creates a set of synthetic worker annotations by:
		1. Assigning random reliability scores to each worker
		2. Creating noisy label sets based on those reliabilities

		Parameters
		----------
		config : Settings
			Configuration object containing simulation parameters
		data : Dict[DataMode, pd.DataFrame]
			Dictionary containing training and test data with ground truth labels
		n_workers : int
			Number of workers to simulate

		Returns
		-------
		Tuple[WorkerReliabilitiesSeriesType, Dict[DataMode, WorkerLabelsDFType]]
			- WorkerReliabilities: Worker reliability scores indexed by worker IDs
			- WorkerLabels: Worker labels for train and test sets
				Format: {'train'|'test': DataFrame(columns=['truth', 'worker_0', 'worker_1', ...])}

		Notes
		-----
		Worker reliabilities are randomly sampled from the range specified in config.
		Labels are generated by introducing controlled noise based on worker reliability.
		"""
		WG = cls(config=config, n_workers=n_workers)
		WG.assign_reliabilities_to_each_worker()
		WG.create_worker_label_set(data=data)

		if WG.workers_assigned_reliabilities is None or WG.workers_labels is None:
			raise RuntimeError("Failed to generate worker reliabilities or labels")

		return WG.workers_assigned_reliabilities, WG.workers_labels
