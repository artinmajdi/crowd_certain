import functools
import multiprocessing
import pickle
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from crowdkit import aggregation as crowdkit_aggregation
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.special import bdtrc
from sklearn import ensemble as sk_ensemble, metrics as sk_metrics

from crowd_certain.utilities import load_data
from crowd_certain.utilities.params import ConfidenceScoreNames, ConsistencyTechniques, DatasetNames, \
	EvaluationMetricNames, MainBenchmarks, \
	OtherBenchmarkNames, ProposedTechniqueNames, SimulationMethods, StrategyNames, UncertaintyTechniques
from crowd_certain.utilities.settings import OutputModes, Settings


class LoadSaveFile:
	"""
	A utility class for loading and saving files with different formats.

	This class provides a simple interface to load and save files with extensions
	such as .pkl, .csv, and .xlsx, handling the appropriate serialization format
	automatically based on the file extension.

	Parameters
	----------
	path : pathlib.Path or str
		The file path where the file should be loaded from or saved to.

	Methods
	-------
	load(index_col=None, header=None)
		Load a file from the specified path based on its extension.

	dump(file, index=False)
		Save a file to the specified path based on its extension.
	"""
	def __init__(self, path):
		self.path = path

	def load(self, index_col=None, header=None):

		if self.path.exists():

			if self.path.suffix == '.pkl':
				with open(self.path, 'rb') as f:
					return pickle.load(f)

			elif self.path.suffix == '.csv':
				return pd.read_csv(self.path)

			elif self.path.suffix == '.xlsx':
				return pd.read_excel(self.path, index_col=index_col, header=header)

		return None

	def dump(self, file, index=False):

		self.path.parent.mkdir(parents=True, exist_ok=True)

		if self.path.suffix == '.pkl':
			with open(self.path, 'wb') as f:
				pickle.dump(file, f)

		elif self.path.suffix == '.csv':
			file.to_csv(self.path, index=index)

		elif self.path.suffix == '.xlsx':
			file.to_excel(self.path, index=index)


# @functools.cache
class BenchmarkTechniques:
	"""
	A utility class to apply and evaluate various crowdsourcing benchmark techniques from the crowdkit library.

	This class provides functionality to aggregate crowd-sourced labels using different benchmark algorithms,
	format data for compatibility with the crowdkit library, and evaluate the performance of these techniques.
	It supports parallel processing for improved performance when applying multiple techniques.

	Key Features:
	- Support for multiple benchmark techniques (MajorityVote, MACE, MMSR, Wawa, ZeroBasedSkill, GLAD, DawidSkene, KOS)
	- Exception handling to gracefully manage technique failures
	- Parallel processing option for improved performance
	- Data preprocessing to convert between different formats

	Methods:
		__init__                              : Initialize with crowd labels and ground truth
		get_techniques                        : Apply a specific benchmark technique to the test data
		wrapper                               : Wrapper function to apply a benchmark technique and return its name with results
		objective_function                    : Create a partial function for applying benchmark techniques to specific test data
		apply                                 : Apply all benchmark techniques to the provided labeled data
		calculate                             : Calculate aggregated labels using all benchmark techniques
		reshape_dataframe_into_this_sdk_format: Preprocess data to match the crowdkit SDK format

	Usage:
		# Create instance with crowd labels and ground truth
		benchmarks = BenchmarkTechniques(crowd_labels=crowd_data, ground_truth=ground_truth)

		# Calculate results from all benchmark techniques
		results = benchmarks.calculate(use_parallelization_benchmarks=True)

		# Or use the class method directly
		results = BenchmarkTechniques.apply(true_labels, use_parallelization_benchmarks=True)

	A utility class to apply and evaluate various crowdsourcing benchmark techniques.

	This class handles the application of different benchmark techniques for aggregating
	crowd-sourced labels, using implementations from the crowdkit library. It manages
	data formatting and processing for these techniques.

	Attributes:
		ground_truth (dict): Dictionary containing ground truth labels for train/test data
		crowd_labels (dict): Dictionary containing crowd-sourced labels for train/test data
		crowd_labels_original (dict): Backup of the original crowd labels
	"""

	def __init__(self, crowd_labels, ground_truth): # type: (Dict, Dict) -> None
		"""
		Initialize the BenchmarkTechniques class with crowd labels and ground truth.

		Parameters:
			crowd_labels (dict): Dictionary with 'train' and 'test' keys, each containing
								a DataFrame with crowd-sourced labels.
			ground_truth (dict): Dictionary with 'train' and 'test' keys, each containing
								a DataFrame with ground truth labels.
		"""
		self.ground_truth          = ground_truth
		self.crowd_labels          = crowd_labels
		self.crowd_labels_original = crowd_labels.copy()

		for mode in ['train', 'test']:
			self.crowd_labels[mode] = self.reshape_dataframe_into_this_sdk_format(self.crowd_labels[mode])


	@classmethod
	def get_techniques(cls, benchmark_name: OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray):
		"""
		Apply a specific benchmark technique to the test data.

		This method applies the specified benchmark technique to the test data and returns
		the aggregated predictions. Each technique is wrapped in an exception handler to
		return zeros in case of failure.

		Parameters:
			benchmark_name (OtherBenchmarkNames): The benchmark technique to apply
			test (pd.DataFrame): The test data in the crowdkit format
			test_unique (np.ndarray): Unique task IDs in the test data

		Returns:
			np.ndarray: Aggregated predictions from the applied benchmark technique
		"""
		def exception_handler(func):
			"""
			Decorator to handle exceptions in benchmark techniques.

			If the wrapped function raises an exception, returns zeros instead.

			Parameters:
				func: The function to wrap with exception handling

			Returns:
				function: Wrapped function that returns zeros on exception
			"""
			def inner_function(*args, **kwargs):
				try:
					return func(*args, **kwargs)
				except Exception:
					return np.zeros(test_unique.shape)

			return inner_function

		@exception_handler
		def KOS():
			r"""Karger-Oh-Shah aggregation model.

				Iterative algorithm that calculates the log-likelihood of the task being positive while modeling
				the reliabilities of the workers.

				Let $A_{ij}$ be a matrix of answers of worker $j$ on task $i$.
				$A_{ij} = 0$ if worker $j$ didn't answer the task $i$, otherwise $|A_{ij}| = 1$.
				The algorithm operates on real-valued task messages $x_{i \rightarrow j}$ and
				worker messages $y_{j \rightarrow i}$. A task message $x_{i \rightarrow j}$ represents
				the log-likelihood of task $i$ being a positive task, and a worker message $y_{j \rightarrow i}$ represents
				how reliable worker $j$ is.

				On iteration $k$ the values are updated as follows:
				$$
				x_{i \rightarrow j}^{(k)} = \sum_{j^{'} \in \partial i \backslash j} A_{ij^{'}} y_{j^{'} \rightarrow i}^{(k-1)} \\
				y_{j \rightarrow i}^{(k)} = \sum_{i^{'} \in \partial j \backslash i} A_{i^{'}j} x_{i^{'} \rightarrow j}^{(k-1)}
				$$

				Karger, David R., Sewoong Oh, and Devavrat Shah. Budget-optimal task allocation for reliable crowdsourcing systems.
				Operations Research 62.1 (2014): 1-24.

				<https://arxiv.org/abs/1110.3564>

			"""
			return crowdkit_aggregation.KOS().fit_predict(test)

		@exception_handler
		def MACE():
			"""
			MACE (Multi-Annotator Competence Estimation) aggregation model.

			A probabilistic model that estimates worker competence and assigns weights
			accordingly. Returns probability of the positive class.

			Returns:
				np.ndarray: Probability of the positive class for each task
			"""
			return crowdkit_aggregation.MACE(n_iter=10).fit_predict_proba(test)[1]

		@exception_handler
		def MajorityVote():
			"""
			Simple majority voting aggregation.

			Aggregates labels by taking the majority vote from all workers.

			Returns:
				np.ndarray: Majority vote result for each task
			"""
			return crowdkit_aggregation.MajorityVote().fit_predict(test)

		@exception_handler
		def MMSR():
			"""
			MMSR (Multi-class Minimax Soft Reliability) aggregation model.

			A generalization of Dawid-Skene that uses a minimax entropy approach.

			Returns:
				np.ndarray: Predicted labels for each task
			"""
			return crowdkit_aggregation.MMSR().fit_predict(test)

		@exception_handler
		def Wawa():
			"""
			Wawa aggregation model.

			A worker-weighted aggregation model where weights are determined by
			worker performance.

			Returns:
				np.ndarray: Probability of the positive class for each task
			"""
			return crowdkit_aggregation.Wawa().fit_predict_proba(test)[1]

		@exception_handler
		def ZeroBasedSkill():
			"""
			Zero-Based Skill aggregation model.

			Estimates worker skill from a baseline of zero and weighs votes accordingly.

			Returns:
				np.ndarray: Probability of the positive class for each task
			"""
			return crowdkit_aggregation.ZeroBasedSkill().fit_predict_proba(test)[1]

		@exception_handler
		def GLAD():
			"""
			GLAD (Generative model of Labels, Abilities, and Difficulties) aggregation.

			Models both worker ability and task difficulty to weigh responses.

			Returns:
				np.ndarray: Probability of the positive class for each task
			"""
			return crowdkit_aggregation.GLAD().fit_predict_proba(test)[1]

		@exception_handler
		def DawidSkene():
			"""
			Dawid-Skene aggregation model.

			A probabilistic model that estimates worker confusion matrices
			to determine worker reliability.

			Returns:
				np.ndarray: Predicted labels for each task
			"""
			return crowdkit_aggregation.DawidSkene().fit_predict(test)


		return eval(benchmark_name.value)()


	@staticmethod
	def wrapper(benchmark_name: OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray) -> Tuple[OtherBenchmarkNames, np.ndarray]:
		"""
		Wrapper function to apply a benchmark technique and return its name with results.

		This static method applies the specified benchmark technique and pairs its name
		with the resulting predictions.

		Parameters:
			benchmark_name (OtherBenchmarkNames): The benchmark technique to apply
			test (pd.DataFrame): The test data in the crowdkit format
			test_unique (np.ndarray): Unique task IDs in the test data

		Returns:
			tuple: A tuple containing (benchmark_name, aggregated_labels)
		"""
		return benchmark_name, BenchmarkTechniques.get_techniques( benchmark_name=benchmark_name, test=test, test_unique=test_unique )


	@staticmethod
	def objective_function(test, test_unique):
		"""
		Create a partial function for applying benchmark techniques to specific test data.

		This method returns a partial function that can be used with map() for parallel
		processing of multiple benchmark techniques on the same test data.

		Parameters:
			test (pd.DataFrame): The test data in the crowdkit format
			test_unique (np.ndarray): Unique task IDs in the test data

		Returns:
			functools.partial: A partial function with fixed test data parameters
		"""
		return functools.partial(BenchmarkTechniques.wrapper, test=test, test_unique=test_unique)


	@classmethod
	def apply(cls, true_labels, use_parallelization_benchmarks) -> pd.DataFrame:
		"""
		Apply all benchmark techniques to the provided labeled data.

		This class method creates a BenchmarkTechniques instance with the provided data
		and applies all benchmark techniques, optionally using parallel processing.

		Parameters:
			true_labels (dict): Dictionary with 'train' and 'test' keys, each containing a DataFrame with ground truth and crowd labels
			use_parallelization_benchmarks (bool): Whether to use parallel processing

		Returns:
			pd.DataFrame: DataFrame with all benchmark technique results as columns
		"""
		ground_truth = {n: true_labels[n].truth.copy() 				     for n in ['train', 'test']}
		crowd_labels = {n: true_labels[n].drop(columns=['truth']).copy() for n in ['train', 'test']}
		return cls(crowd_labels=crowd_labels, ground_truth=ground_truth).calculate(use_parallelization_benchmarks)


	def calculate(self, use_parallelization_benchmarks) ->  pd.DataFrame:
		"""
		Calculate aggregated labels using all benchmark techniques.

		This method applies all defined benchmark techniques to the test data,
		either in parallel or sequentially based on the parameter.

		Parameters:
			use_parallelization_benchmarks (bool): Whether to use parallel processing

		Returns:
			pd.DataFrame: DataFrame with benchmark technique names as columns and
						aggregated labels as values
		"""
		# train    = self.crowd_labels['train']
		# train_gt = self.ground_truth['train']
		test: pd.DataFrame = self.crowd_labels['test']
		test_unique: np.ndarray = test.task.unique()

		function = self.objective_function(test=test, test_unique=test_unique)

		if use_parallelization_benchmarks:
			with multiprocessing.Pool(processes=len(OtherBenchmarkNames)) as pool:
				output = pool.map(function, list(OtherBenchmarkNames))

		else:
			output = [function(benchmark_name=m) for m in OtherBenchmarkNames]

		return pd.DataFrame({benchmark_name.value: aggregated_labels for benchmark_name, aggregated_labels in output})


	@staticmethod
	def reshape_dataframe_into_this_sdk_format(df_predicted_labels):
		"""
		Preprocess data to match the crowdkit SDK format.

		This method transforms a DataFrame of predicted labels from a wide format (with workers
		as columns) to a long format suitable for the crowdkit library, with worker, task, and
		label as separate columns.

		Parameters:
			df_predicted_labels (pd.DataFrame): DataFrame with worker labels as columns

		Returns:
			pd.DataFrame: Reshaped DataFrame in crowdkit format with columns [worker, task, label]
		"""
		# Converting labels from binary to integer
		df_crowd_labels: pd.DataFrame = df_predicted_labels.astype(int).copy()

		# Separating the ground truth labels from the crowd labels
		# ground_truth = df_crowd_labels.pop('truth')

		# Stacking all the workers labels into one column
		df_crowd_labels = df_crowd_labels.stack().reset_index().rename( columns={'level_0': 'task', 'level_1': 'worker', 0: 'label'})

		# Reordering the columns to make it similar to crowd-kit examples
		df_crowd_labels = df_crowd_labels[['worker', 'task', 'label']]

		return df_crowd_labels  # , ground_truth


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
	workers_strength: pd.DataFrame
	n_workers       : int
	true_label      : dict[str, pd.DataFrame]


@dataclass
class ResultComparisonsType:
	outputs                   : dict
	config                    : 'Settings'
	weight_strength_relation  : pd.DataFrame


@dataclass
class AIM1_3:
	"""
	Main class for AIM1_3 implementation that handles the calculation of worker weights, uncertainties, and confidence scores
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
		aim1_3_meauring_probs_uncertainties(): Simulates worker responses and calculates uncertainty metrics
		calculate_uncertainties(): Computes various uncertainty metrics for predictions
		calculate_consistency(): Converts uncertainties to consistency scores
		aim1_3_measuring_proposed_weights(): Calculates weights for the proposed techniques
		get_weights(): Returns weights for proposed and benchmark techniques
		measuring_nu_and_confidence_score(): Calculates confidence scores and aggregated labels
		core_measurements(): Main method that orchestrates the calculation pipeline
		calculate_one_dataset(): Entry point for running simulations on a single dataset
		calculate_all_datasets(): Entry point for running simulations on multiple datasets
	"""
	data            : dict
	feature_columns : list
	config           : 'Settings'
	n_workers       : int = 3
	seed            : int = 0

	def __post_init__(self):
		# Setting the random seed
		np.random.seed(self.seed + 1)


	@staticmethod
	def get_accuracy(aggregated_labels, n_workers, delta_benchmark, truth):
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
		- ProposedTechniqueNames: Custom proposed techniques
		- MainBenchmarks: Main benchmark methods
		- OtherBenchmarkNames: Additional benchmark methods
		- MV_Classifier: Majority voting accuracy

		Accuracy is calculated by thresholding aggregated labels at 0.5 and
		comparing with ground truth.
		"""

		accuracy = pd.DataFrame(index=[n_workers])
		for methods in [ProposedTechniqueNames, MainBenchmarks, OtherBenchmarkNames]:
			for m in methods:
				accuracy[m] = ((aggregated_labels[m] >= 0.5) == truth).mean(axis=0)

		accuracy['MV_Classifier'] = ( (delta_benchmark.mean(axis=1) >= 0.5) == truth).mean(axis=0)

		return accuracy


	def calculate_uncertainties(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Calculate various uncertainty metrics for a DataFrame.

		This method calculates different uncertainty metrics for each row in the provided DataFrame.
		The metrics are specified in the configuration and can include standard deviation,
		entropy, coefficient of variation, prediction interval, and confidence interval.

		Parameters
		----------
		df : pd.DataFrame
			DataFrame containing values for which uncertainty needs to be calculated.
			Each row represents one item, and columns represent different predictions or values.
			Values should be convertible to integers.

		Returns
		-------
		pd.DataFrame
			DataFrame containing calculated uncertainty values for each row in the input DataFrame.
			Each column corresponds to one uncertainty technique specified in the configuration.
			The columns are named according to the value attribute of the uncertainty technique enum.

		Notes
		-----
		- The entropy measure is normalized to fall between 0 and 1.
		- The coefficient of variation (CV) is transformed using hyperbolic tangent to bound it between 0 and 1.
		- For confidence interval calculation, rows with zero standard deviation will have NaN values, which are then filled with zeros.
		"""

		epsilon = np.finfo(float).eps
		df = df.astype(int)

		df_uncertainties = pd.DataFrame(columns=[l.value for l in self.config.technique.uncertainty_techniques], index=df.index)

		for tech in self.config.technique.uncertainty_techniques:

			if tech is UncertaintyTechniques.STD:
				df_uncertainties[tech.value] = df.std( axis=1 )

			elif tech is UncertaintyTechniques.ENTROPY:
				# Normalize each row to sum to 1
				df_normalized = df.div(df.sum(axis=1) + epsilon, axis=0)
				# Calculate entropy
				entropy = - (df_normalized * np.log(df_normalized + epsilon)).sum(axis=1)
				# entropy = df.apply(lambda x: scipy.stats.entropy(x), axis=1).fillna(0)

				# normalizing entropy values to be between 0 and 1
				df_uncertainties[tech.value] = entropy / np.log(df.shape[1])

			elif tech is UncertaintyTechniques.CV:
				# The coefficient of variation (CoV) is a measure of relative variability. It is defined as the ratio of the standard deviation. CoV doesn't have an upper bound, but it's always non-negative. Normalizing CoV to a range of [0, 1] isn't straightforward because it can theoretically take any non-negative value. A common approach is to use a transformation that asymptotically approaches 1 as CoV increases. However, the choice of transformation can be somewhat arbitrary and may depend on the context of your data. One simple approach is to use a bounded function like the hyperbolic tangent:

				coefficient_of_variation = df.std(axis=1) / (df.mean(axis=1) + epsilon)
				df_uncertainties[tech.value] = np.tanh(coefficient_of_variation)

			elif tech is UncertaintyTechniques.PI:
				df_uncertainties[tech.value] = df.apply(lambda row: np.percentile(row.astype(int), 75) - np.percentile(row.astype(int), 25), axis=1)

			elif tech is UncertaintyTechniques.CI:
				# Lower Bound: This is the first value in the tuple. It indicates the lower end of the range. If you have a 95% confidence interval, this means that you can be 95% confident that the true population mean is greater than or equal to this value.

				# Upper Bound: This is the second value in the tuple. It represents the upper end of the range. Continuing with the 95% confidence interval example, you can be 95% confident that the true population mean is less than or equal to this value.

				confidene_interval = df.apply(lambda row: scipy.stats.norm.interval(0.95, loc=np.mean(row), scale=scipy.stats.sem(row)) if np.std(row) > 0 else (np.nan, np.nan),axis=1).apply(pd.Series)

				# Calculate the width of the confidence interval (uncertainty score)
				weight = confidene_interval[1] - confidene_interval[0]

				# Optional: Normalize by the mean or another relevant value,
				# For example, normalize by the midpoint of the interval
				# midpoint = (confidene_interval[1] + confidene_interval[0]) / 2

				df_uncertainties[tech.value] = weight.fillna(0) # / midpoint

		return df_uncertainties


	def calculate_consistency(self, uncertainty: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
		"""
		Calculates consistency metrics from uncertainty values.

		This method computes consistency scores using different techniques based on the provided uncertainty data.
		Consistency is the inverse of uncertainty - higher consistency implies lower uncertainty.

		Parameters
		----------
		uncertainty : Union[pd.DataFrame, pd.Series, np.ndarray]
			The uncertainty values to convert to consistency.
			- If DataFrame: Assumes multi-level columns with uncertainty values
			- If Series: A simple series of uncertainty values
			- If ndarray: A numpy array containing uncertainty values

		Returns
		-------
		pd.DataFrame
			A DataFrame containing the calculated consistency values.
			- If input was DataFrame: Returns multi-level columned DataFrame with consistency techniques as the second level (worker is the highest level)
			- If input was Series/ndarray: Returns DataFrame with consistency techniques as columns

		Notes
		-----
		Supported consistency calculation techniques:
			- ONE_MINUS_UNCERTAINTY: 1 - uncertainty
			- ONE_DIVIDED_BY_UNCERTAINTY: 1 / uncertainty (with epsilon to prevent division by zero)
		"""

		def initialize_consistency():
			nonlocal consistency
			upper_level = [l.value for l in self.config.technique.consistency_techniques]

			if isinstance(uncertainty, pd.DataFrame):
				# The use of OrderedDict helps preserving the order of columns.
				levels = [list(OrderedDict.fromkeys(uncertainty.columns.get_level_values(i))) for i in range(uncertainty.columns.nlevels)]

				new_columns 	 = [upper_level] + levels
				new_columnsnames = ['consistency_technique'] + uncertainty.columns.names

				columns = pd.MultiIndex.from_product(new_columns, names=new_columnsnames)
				consistency = pd.DataFrame(columns=columns, index=uncertainty.index)

			elif isinstance(uncertainty, pd.Series):
				consistency = pd.DataFrame(columns=upper_level, index=uncertainty.index)

			elif isinstance(uncertainty, np.ndarray):
				consistency = pd.DataFrame(columns=upper_level, index=range(uncertainty.shape[0]))

			return consistency

		consistency = initialize_consistency()

		for tech in self.config.technique.consistency_techniques:
			if tech is ConsistencyTechniques.ONE_MINUS_UNCERTAINTY:
				consistency[tech.value] = 1 - uncertainty

			elif tech is ConsistencyTechniques.ONE_DIVIDED_BY_UNCERTAINTY:
				consistency[tech.value] = 1 / (uncertainty + np.finfo(float).eps)

		# Making the worker the highest level in the columns
		if isinstance(uncertainty, pd.DataFrame):
			return consistency.swaplevel(0, 1, axis=1)

		return consistency

	@staticmethod
	def get_AUC_ACC_F1(aggregated_labels: pd.Series, truth: pd.Series) -> pd.Series:
		"""
		Calculate AUC, accuracy, and F1 score metrics between aggregated labels and ground truth.

		This function computes evaluation metrics for binary classification problems by comparing
		aggregated labels (predictions) with the ground truth labels. It handles missing values
		in the truth series by filtering them out before calculation.

		Parameters:
		-----------
		aggregated_labels : pd.Series
			The aggregated (predicted) probability labels, typically between 0 and 1.

		truth : pd.Series
			The ground truth labels with the same index as aggregated_labels.
			Can contain null values which will be filtered out.

		Returns:
		--------
		pd.Series
			A pandas Series containing the following metrics:
			- AUC (Area Under the ROC Curve)
			- Accuracy
			- F1 score

			The Series is indexed by values from the EvaluationMetricNames enum.
			If truth has no valid values or is not binary, the metrics will be NaN.

		Notes:
		------
		- The aggregated labels are thresholded at 0.5 to convert to binary predictions
		- Metrics are only calculated if there are non-null values in truth and truth contains exactly 2 unique values
		"""

		metrics = pd.Series(index=EvaluationMetricNames.values())

		non_null = ~truth.isnull()
		truth_notnull = truth[non_null].to_numpy()

		if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):

			yhat = (aggregated_labels > 0.5).astype(int)[non_null]
			metrics[EvaluationMetricNames.AUC.value] = sk_metrics.roc_auc_score( truth_notnull, yhat)
			metrics[EvaluationMetricNames.ACC.value] = sk_metrics.accuracy_score(truth_notnull, yhat)
			metrics[EvaluationMetricNames.F1.value ] = sk_metrics.f1_score( 	 truth_notnull, yhat)

		return metrics


	def aim1_3_meauring_probs_uncertainties(self):
		"""
		Simulate workers with varying skill levels and measure their predictions and uncertainties.

		This method performs a simulation where:
		1. Workers with different strengths (skills) are generated randomly
		2. These workers provide noisy labels for the data based on their strength
		3. Classifiers are trained on these noisy labels to make predictions
		4. Uncertainties are calculated for each worker's predictions
		5. Different simulation methods can be used (random states or multiple classifiers)

		Returns:
			tuple: A tuple containing:
				- preds (dict): Predictions organized by mode ('train'/'test'), simulation, and worker
				- uncertainties (dict): Uncertainty measures for each worker and uncertainty technique
				- truth (dict): The true labels and worker-annotated labels
				- workers_strength (pd.DataFrame): Worker strength information including:
					* workers_strength: The randomly assigned strength value
					* accuracy-test-classifier: Accuracy of each worker's classifier on test data
					* accuracy-test: Accuracy of each worker's noisy labels compared to ground truth
		"""

		def assigning_strengths_randomly_to_each_worker():
			"""
			Generates a dataframe of workers with randomly assigned strength values.

			The function creates a list of worker names and assigns a random strength value
			to each worker from a uniform distribution defined by the configuration parameters.

			Returns:
				pandas.DataFrame: A dataframe with worker names as the index and their assigned
									strength values in a column named 'workers_strength'.

			Note:
				The strength values are sampled from a uniform distribution with bounds
				specified by self.config.simulation.low_dis and self.config.simulation.high_dis.
			"""

			workers_names = [f'worker_{j}' for j in range(self.n_workers)]

			workers_strength_array = np.random.uniform(low=self.config.simulation.low_dis, high=self.config.simulation.high_dis, size=self.n_workers)

			return pd.DataFrame({'workers_strength': workers_strength_array}, index=workers_names)

		def looping_over_all_workers():
			"""
			Performs simulations to evaluate worker performance in crowdsourced labeling tasks.

			This function executes a series of simulations that:
			1. Initializes dataframes for predicted labels, ground truth, and uncertainties
			2. Generates simulated noisy labels for each worker based on their strength
			3. Trains classifiers on these noisy labels and makes predictions
			4. Calculates uncertainty metrics across multiple simulations
			5. Aggregates results across workers and simulations

			The function uses nested helper functions to modularize the workflow:
			- initialize_variables: Sets up data structures for storing results
			- update_noisy_labels: Creates synthetic noisy labels based on worker quality
			- update_predicted_labels_all_sims: Trains classifiers and makes predictions for each simulation
			- update_uncertainties: Calculates uncertainty metrics for predictions
			- get_n_simulations: Determines number of simulations based on config
			- swap_axes: Reorganizes the prediction results to be grouped by simulation instead of by worker

			Returns:
				tuple: Contains (truth, uncertainties, preds)
					- truth: Dict with train/test DataFrames containing ground truth and noisy labels
					- uncertainties: Dict with train/test DataFrames containing uncertainty metrics per worker
					- preds: Dict with predictions reorganized by simulation rather than by worker
			"""
			nonlocal workers_strength, preds, truth, uncertainties

			def initialize_variables():
				"""
				Initialize data structures for storing prediction results, ground truth, and uncertainties.

				Returns:
					tuple: A tuple containing:
						- predicted_labels_all_sims (dict): Dictionary with 'train' and 'test' keys, where each contains simulation results
						- truth (dict): Dictionary with 'train' and 'test' keys, each containing a pandas DataFrame for ground truth values
						- uncertainties (dict): Dictionary with 'train' and 'test' keys, each containing a pandas DataFrame with a MultiIndex
							of worker IDs and uncertainty techniques

				Note:
					This function uses nonlocal variables that must be defined in the outer scope.
					The uncertainties DataFrame will have columns based on the worker IDs and configured uncertainty techniques.
				"""
				nonlocal predicted_labels_all_sims, truth, uncertainties

				predicted_labels_all_sims = {'train': {}, 'test': {}}

				truth = { 'train': pd.DataFrame(), 'test': pd.DataFrame() }

				columns = pd.MultiIndex.from_product([workers_strength.index, [l.value for l in self.config.technique.uncertainty_techniques]], names=['worker', 'uncertainty_technique'])

				uncertainties = { 'train': pd.DataFrame(columns=columns), 'test': pd.DataFrame(columns=columns) }

				return predicted_labels_all_sims, truth, uncertainties

			def update_noisy_labels():
				"""
				Updates noisy labels for simulation by introducing controlled inaccuracies in true labels based
				on worker strength.

				The function manipulates the global variables predicted_labels_all_sims and truth, creating
				noisy versions of the ground truth labels for both training and test datasets. The noise level
				is determined by the worker's strength parameter, where lower strength values result in more
				label errors.

				Note:
					This function uses nonlocal variables and modifies them directly.

				Internal Functions:
					getting_noisy_manual_labels_for_each_worker: Creates noisy versions of true labels
					by flipping some labels based on the labeler's strength parameter.

				Side Effects:
					- Updates predicted_labels_all_sims with empty dictionaries for the current labeler
					- Updates truth dictionary with both original and noisy versions of labels for train and test sets
				"""
				nonlocal predicted_labels_all_sims, truth

				def getting_noisy_manual_labels_for_each_worker(truth_array: np.ndarray, l_strength: float):
					"""
					Generate noisy labels by flipping true labels with a probability determined by labeling strength.

					This function simulates worker annotations by introducing noise to ground truth labels.
					Each sample has a probability of (1 - l_strength) to have its true label flipped.

					Parameters
					----------
					truth_array : np.ndarray
						Array of ground truth probabilities/scores for each sample.
						Shape: (num_samples,)

					l_strength : float
						Labeling strength parameter in range [0,1].
						Higher values mean more accurate labels (less noise).
						At l_strength=1, no labels are flipped.
						At l_strength=0, all labels are flipped.

					Returns
					-------
					np.ndarray
						Binary array of noisy labels where some true labels are flipped based on l_strength.
						Shape: (num_samples,)

					Notes
					-----
					The function first converts truth_array to binary labels using a 0.5 threshold,
					then randomly flips some labels based on the labeling strength parameter.
					"""

					# number of samples and workers/workers
					num_samples = truth_array.shape[0]

					# finding a random number for each instance
					true_label_assignment_prob = np.random.random(num_samples)

					# samples that will have an inaccurate true label
					false_samples = true_label_assignment_prob < 1 - l_strength

					# measuring the new labels for each worker/worker
					worker_truth = truth_array > 0.5
					worker_truth[false_samples] = ~ worker_truth[false_samples]

					return worker_truth


				# Initializationn
				for mode in ['train', 'test']:
					predicted_labels_all_sims[mode][LB] = {}
					truth[mode]['truth'] = self.data[mode].true.copy()

				# Extracting the simulated noisy manual labels based on the worker's strength
				# TODO: check if the seed_num should be LB_index instead of 0 an 1
				truth['train'][LB] = getting_noisy_manual_labels_for_each_worker( truth_array=self.data['train'].true.values, l_strength=workers_strength.T[LB].values )
				truth['test'][LB]  = getting_noisy_manual_labels_for_each_worker( truth_array=self.data['test'].true.values, l_strength=workers_strength.T[LB].values )

			def update_predicted_labels_all_sims(LB, sim_num):
				"""
				Predicts labels for both training and test data using trained classifiers.

				This function trains a classifier based on the simulation method specified in the configuration
				and uses it to predict labels for both training and test datasets. The predictions are stored
				in the `predicted_labels_all_sims` dictionary.

				Parameters:
				----------
				LB : str
					The label column name to be predicted.
				sim_num : int
					The simulation number/index used to determine the classifier or its random state.

				Notes:
				-----
				The function uses the nonlocal variable `predicted_labels_all_sims` to store prediction results.
				Predictions are organized by mode ('train' or 'test'), label column (LB), and simulation number.

				For SimulationMethods.RANDOM_STATES, it uses a RandomForestClassifier with varying random states.
				For SimulationMethods.MULTIPLE_CLASSIFIERS, it selects different classifier types from a predefined list.
				"""
				nonlocal predicted_labels_all_sims

				def get_classifier():
					"""
					Returns a classifier based on the simulation method configuration.

					For SimulationMethods.RANDOM_STATES:
						Returns a RandomForestClassifier with fixed parameters but varying random_state
						for different simulation numbers.

					For SimulationMethods.MULTIPLE_CLASSIFIERS:
						Returns a classifier from the pre-configured list of classifiers based on
						the simulation number.

					Returns:
						sklearn estimator: A classifier instance configured according to the simulation method.
					"""
					if self.config.simulation.simulation_methods is SimulationMethods.RANDOM_STATES:
						return sk_ensemble.RandomForestClassifier(n_estimators=4, max_depth=4, random_state=self.seed * sim_num) # n_estimators=4, max_depth=4

					elif self.config.simulation.simulation_methods is SimulationMethods.MULTIPLE_CLASSIFIERS:
						return self.config.simulation.classifiers_list[sim_num]

				# training a random forest on the aformentioned labels
				classifier = get_classifier()

				classifier.fit( X=self.data['train'][self.feature_columns], y=truth['train'][LB] )

				for mode in ['train', 'test']:
					predicted_labels_all_sims[mode][LB] [ f'simulation_{sim_num}' ] = classifier.predict(self.data[mode][self.feature_columns])

			def update_uncertainties(LB):
				"""
				Updates the uncertainties and predicted labels for a specific label for both training and test data.

				This function processes the prediction results across all simulations for a given label (LB).
				It converts the predictions to pandas DataFrame format, calculates uncertainties for each worker
				across all simulations, and computes the majority vote prediction for each item.

				Parameters
				----------
				LB : str
					The label identifier for which uncertainties need to be updated.

				Notes
				-----
				This function uses nonlocal variables:
				- predicted_labels_all_sims: Dictionary containing prediction results for all simulations
				- uncertainties: Dictionary to store calculated uncertainty values

				The function processes both 'train' and 'test' modes separately.
				"""
				nonlocal predicted_labels_all_sims, uncertainties

				# Measuring the prediction and uncertainties values after MV over all simulations
				for mode in ['train', 'test']:
					# converting to dataframe
					predicted_labels_all_sims[mode][LB] = pd.DataFrame(predicted_labels_all_sims[mode][LB], index=self.data[mode].index)

					# uncertainties for each worker over all simulations
					uncertainties[mode][LB] = self.calculate_uncertainties(df=predicted_labels_all_sims[mode][LB])

					# predicted probability of each class after MV over all simulations
					predicted_labels_all_sims[mode][LB]['mv'] = ( predicted_labels_all_sims[mode][LB].mean(axis=1) > 0.5)

			def get_n_simulations():
				"""
				Determines and returns the number of simulations to be run based on the simulation method.

				Returns:
					int: The number of simulations.
						- If using RANDOM_STATES method, returns the configured number of simulations.
						- If using MULTIPLE_CLASSIFIERS method, returns the number of classifiers in the list.
				"""
				if self.config.simulation.simulation_methods is SimulationMethods.RANDOM_STATES:
					return self.config.simulation.num_simulations

				elif self.config.simulation.simulation_methods is SimulationMethods.MULTIPLE_CLASSIFIERS:
					return len(self.config.simulation.classifiers_list)

			def swap_axes(predicted_labels_all_sims) -> dict[str, dict[str, pd.DataFrame]]:
				"""
				Swaps the axes of the predicted labels dataframe organization.

				Transforms the structure from {mode: {worker: {simulation: predictions}}} to
				{mode: {simulation: dataframe}}, where each dataframe has workers as columns.

				Parameters
				----------
				predicted_labels_all_sims : dict
					A nested dictionary structure containing predicted labels organized by
					training/test mode, workers, and simulations.

				Returns
				-------
				dict[str, dict[str, pd.DataFrame]]
					Reorganized predictions where:
					- First level keys are 'train' and 'test'
					- Second level keys are simulation identifiers ('simulation_0', ..., 'mv')
					- Values are pandas DataFrames with workers as columns and samples as rows
				"""

				# reshaping the dataframes
				preds_swaped: dict[str, dict[str, pd.DataFrame]] = { 'train': defaultdict(pd.DataFrame), 'test': defaultdict(pd.DataFrame) }

				for mode in ['train', 'test']:

					# reversing the order of simulations and workers. NOTE: for the final experiment I should use simulation_0. if I use the mv, then because the augmented truths keeps changing in each simulation, then with enough simulations, I'll end up witht perfect workers.
					for i in range(self.config.simulation.num_simulations + 1):

						SM = f'simulation_{i}' if i < self.config.simulation.num_simulations else 'mv'

						preds_swaped[mode][SM] = pd.DataFrame()
						for LBx in [f'worker_{j}' for j in range( self.n_workers )]:
							preds_swaped[mode][SM][LBx] = predicted_labels_all_sims[mode][LBx][SM]

				return preds_swaped


			predicted_labels_all_sims, truth, uncertainties = initialize_variables()

			for LB_index, LB in enumerate(workers_strength.index):

				update_noisy_labels()

				for sim_num in range(get_n_simulations()):
					update_predicted_labels_all_sims(LB=LB, sim_num=sim_num)

				update_uncertainties(LB)

			preds = swap_axes(predicted_labels_all_sims)

			return truth, uncertainties, preds

		def adding_accuracy_for_each_worker():
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
			nonlocal workers_strength, preds, truth

			workers_strength['accuracy-test-classifier'] = 0.0
			workers_strength['accuracy-test'] = 0.0

			for i in range(self.n_workers):
				LB = f'worker_{i}'

				# accuracy of classifier in simulation_0
				workers_strength.loc[LB, 'accuracy-test-classifier'] = ( preds['test']['simulation_0'][LB] == truth['test'].truth).mean()

				# accuracy of noisy true labels for each worker
				workers_strength.loc[LB, 'accuracy-test'] 		     = ( truth['test'][LB] == truth['test'].truth).mean()


		workers_strength = assigning_strengths_randomly_to_each_worker()

		truth, uncertainties, preds = looping_over_all_workers()

		adding_accuracy_for_each_worker()

		return preds, uncertainties, truth, workers_strength


	def aim1_3_measuring_proposed_weights(self, preds, uncertainties) -> pd.DataFrame:
		"""Calculates proposed weights for worker predictions based on consistency and uncertainty metrics.

		This method computes two sets of weights based on different techniques:
		1. The first technique uses raw consistency scores (T1)
		2. The second technique zeros out consistency scores for incorrect predictions (T2)

		The weights are then normalized by the mean weight across all workers.

			preds (pd.DataFrame): 	Worker predictions with data indices as rows and workers as columns
									Shape: (n_samples, n_workers)
			uncertainties (pd.DataFrame): 	Uncertainty metrics for each worker and technique
										 	Shape: (n_samples, n_workers * n_uncertainty_techniques) with MultiIndex columns [worker, uncertainty_technique]

			pd.DataFrame: Calculated weights for each worker across different consistency techniques,
							uncertainty techniques, and proposed weighting methods.
							The DataFrame has a MultiIndex with levels:
							[ConsistencyTechnique, UncertaintyTechnique, ProposedTechniqueName]
							and columns representing workers.


		Args:
			preds 		  (pd.DataFrame): pd.DataFrame( index   = data_indices,
														columns = [worker_0, worker_1, ...])
			uncertainties (pd.DataFrame): pd.DataFrame( index   = data_indices,
														columns = pd.MultiIndex.from_product([worker_0, ...], UncertaintyTechniques.values()))

		Returns:
			weights = pd.DataFrame( index = pd.MultiIndex.from_product([ConsistencyTechniques.values(),
																		UncertaintyTechniques.values(),
																		ProposedTechniqueNames.values()]),
									columns = ['worker_0', .... ])
		"""

		# TODO This is the part where I should measure the prob_mv_binary for different # of workers instead of all of them
		prob_mv_binary = preds.mean(axis=1) > 0.5

		T1    = self.calculate_consistency( uncertainties )
		T2    = T1.copy()

		proposed_techniques = [l.value for l in ProposedTechniqueNames]
		w_hat = pd.DataFrame(index=proposed_techniques, columns=T1.columns, dtype=float)
		# w_hat2 = pd.Series(index=T1.columns)

		for worker in preds.columns:

			T2.loc[preds[worker].values != prob_mv_binary.values, worker] = 0

			w_hat[worker] = pd.DataFrame.from_dict({proposed_techniques[0]: T1[worker].mean(axis=0),
													proposed_techniques[1]: T2[worker].mean(axis=0)}, orient='index')

			# w_hat.loc[proposed_techniques[0],worker] = T1[worker].mean(axis=0)
			# w_hat[worker].iloc[proposed_techniques[1]] = T2[worker].mean(axis=0)

		# w_hat = pd.DataFrame([w_hat1, w_hat2], index=list(ProposedTechniqueNames)).T

		# measuring average weight over all workers. used to normalize the weights.
		w_hat_mean_over_workers = w_hat.T.groupby(level=[1,2]).sum().T

		# The below for loop can also be written as the following. this would be very slightly faster but much more complicated logit
		# weights = w_hat.T.groupby(level=0, group_keys=False).apply(lambda row: row.divide(w_hat_mean_over_workers.T)).swaplevel(1,2).swaplevel(0,1).sort_index().T
		weights = pd.DataFrame().reindex_like(w_hat)
		for worker in preds.columns:
			weights[worker] = w_hat[worker].divide(w_hat_mean_over_workers)


		# probs_weighted = pd.DataFrame(columns=proposed_techniques, index=preds.index)
		# for method in proposed_techniques:
		# 	# probs_weighted[method] =( predicted_uncertainty * weights[method] ).sum(axis=1)
		# 	probs_weighted[method] = (preds * weights[method]).sum( axis=1 )

		# This will return a series
		weights = weights.unstack().unstack(level='worker')
		return weights


	@staticmethod
	def measuring_Tao_weights_based_on_classifier_labels(delta, noisy_true_labels):
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
	def measuring_Tao_weights_based_on_actual_labels(workers_labels, noisy_true_labels, n_workers):
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
	def calculate_confidence_scores(delta, w: Union[pd.DataFrame, pd.Series], n_workers) -> pd.DataFrame:
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

		P_pos = ( delta * w).sum( axis=1 )
		P_neg = (~delta * w).sum( axis=1 )

		def get_freq():
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
			# F[out['P_pos'] < out['P_neg']] = out['P_neg'][out['P_pos'] < out['P_neg']]
			return pd.DataFrame({'F': out.max(axis=1), 'F_pos': P_pos})

		def get_beta():
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
			# k = l_alpha.floordiv(1)
			# n = (l_alpha+u_beta).floordiv(1) - 1

			get_I = lambda row: bdtrc(row['k'], row['n'], 0.5)
			out['I'] = out.apply(get_I, axis=1)
			# out['I'] = np.nan
			# for index in out['n'].index:
			# 	out['I'][index] = bdtrc(out['k'][index], out['n'][index], 0.5)

			get_F_lambda = lambda row: max(row['I'], 1-row['I'])
			# F = I.copy()
			# F[I < 0.5] = (1 - F)[I < 0.5]
			return pd.DataFrame({'F': out.apply(get_F_lambda, axis=1), 'F_pos': out['I']})

		columns = pd.MultiIndex.from_product([StrategyNames.values(), ['F', 'F_pos']], names=['strategies', 'F_F_pos'])
		confidence_scores = pd.DataFrame(columns=columns, index=delta.index)
		confidence_scores[StrategyNames.FREQ.value] = get_freq()
		confidence_scores[StrategyNames.BETA.value] = get_beta()

		return confidence_scores


	def get_weights(self, workers_labels, preds, uncertainties, noisy_true_labels, n_workers) -> WeightType:
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
		WeightType
			Named tuple containing weights for three methods:
			- PROPOSED: Weights calculated using the proposed technique based on predictions and uncertainties
			- TAO: Weights calculated based on Tao's method using actual labels
			- SHENG: Equal weights (1/n_workers) for all worker-item pairs
		"""

		# Measuring weights for the proposed technique
		weights_proposed = self.aim1_3_measuring_proposed_weights( preds=preds, uncertainties=uncertainties)

		# Benchmark accuracy measurement
		weights_Tao = self.measuring_Tao_weights_based_on_actual_labels( workers_labels=workers_labels, noisy_true_labels=noisy_true_labels, n_workers=n_workers)

		weights_Sheng = pd.DataFrame(1 / n_workers, index=weights_Tao.index, columns=weights_Tao.columns)

		return WeightType(PROPOSED=weights_proposed, TAO=weights_Tao, SHENG=weights_Sheng)


	def measuring_nu_and_confidence_score(self, weights: WeightType, preds_all, true_labels, use_parallelization_benchmarks: bool=False) -> Tuple[ResultType, ResultType]:
		"""Calculates the confidence scores (nu) and evaluation metrics for proposed methods and benchmarks.

		This function computes the confidence scores and aggregated labels using the provided weights
		for both the proposed techniques and benchmark methods (Tao and Sheng). It also calculates
		evaluation metrics (AUC, Accuracy, and F1 score) for all methods.

			weights (WeightType): Object containing weights for proposed methods and benchmarks.
			preds_all: Dictionary containing workers' predictions for different datasets and methods.
			true_labels: Ground truth labels for evaluation.
			use_parallelization_benchmarks (bool, optional): Whether to use parallelization for benchmark calculation.
				Defaults to False.

			Tuple[ResultType, ResultType]: A tuple containing:
				- results_proposed: Results for proposed methods (aggregated labels, confidence scores, metrics)
				- results_benchmarks: Results for benchmark methods (aggregated labels, confidence scores, metrics)
		"""
		def get_results_proposed(preds: pd.DataFrame) -> ResultType:
			"""
			Returns:
				F: pd.DataFrame(columns = pd.MultiIndex.from_product([ConsistencyTechniques, UncertaintyTechniques,ProposedTechniqueNames]),
								index   = pd.MultiIndex.from_product([StrategyNames, data_indices])

				aggregated_labels: pd.DataFrame(columns = pd.MultiIndex.from_product([ConsistencyTechniques, UncertaintyTechniques,ProposedTechniqueNames])
												index   = data_indices)
			"""

			agg_labels = pd.DataFrame(columns=weights.PROPOSED.index, index=preds.index)

			index = pd.MultiIndex.from_product([ ['F', 'F_pos'], StrategyNames.values(), preds.index ], names=['F_F_pos', 'strategies', 'indices'])
			confidence_scores = {}

			metrics = pd.DataFrame(columns=weights.PROPOSED.index, index=EvaluationMetricNames.values())

			for cln in weights.PROPOSED.index:
				agg_labels[cln] = (preds * weights.PROPOSED.T[cln]).sum(axis=1)

				confidence_scores[cln] = AIM1_3.calculate_confidence_scores(delta=preds, w=weights.PROPOSED.T[cln], n_workers=self.n_workers )

				# Measuring the metrics
				metrics[cln] = AIM1_3.get_AUC_ACC_F1(aggregated_labels=agg_labels[cln], truth=true_labels['test'].truth)

			return ResultType(aggregated_labels=agg_labels, confidence_scores=confidence_scores, metrics=metrics)

		def get_results_tao_sheng(workers_labels) -> ResultType:
			"""Calculating the weights for Tao and Sheng methods

			Args:
				workers_labels (pd.DataFrame): pd.DataFrame(columns=[worker_0, ...], index=data_indices)

			Returns:
				Tuple[pd.DataFrame, pd.DataFrame]:
				F 							 = pd.DataFrame(columns = pd.MultiIndex.from_product([MainBenchmarks, StrategyNames] , index = workers_labels.index)
				aggregated_labels_benchmarks = pd.DataFrame(columns = MainBenchmarks 											 , index = workers_labels.index)
			"""

			def get_v_F_Tao_sheng() -> Tuple[pd.DataFrame, dict[str, pd.DataFrame]]:

				def initialize():
					nonlocal agg_labels, confidence_scores

					confidence_scores = {}

					agg_labels = pd.DataFrame(index=workers_labels.index, columns=MainBenchmarks.values())
					return agg_labels, confidence_scores

				agg_labels, confidence_scores = initialize()

				for m in MainBenchmarks:

					w = weights.TAO if m is MainBenchmarks.TAO else weights.SHENG

					agg_labels[m.value] = (workers_labels * w).sum(axis=1)

					confidence_scores[m.value] = AIM1_3.calculate_confidence_scores( delta=workers_labels, w=w, n_workers=self.n_workers )

				return agg_labels, confidence_scores

			v_benchmarks, F_benchmarks = get_v_F_Tao_sheng()

			v_other_benchmarks = BenchmarkTechniques.apply(true_labels=true_labels, use_parallelization_benchmarks=use_parallelization_benchmarks)

			v_benchmarks = pd.concat( [v_benchmarks, v_other_benchmarks], axis=1)

			# Measuring the metrics

			metrics_benchmarks = pd.DataFrame({cln: AIM1_3.get_AUC_ACC_F1(aggregated_labels=v_benchmarks[cln], truth=true_labels['test'].truth) for cln in v_benchmarks.columns})

			return ResultType(aggregated_labels=v_benchmarks, confidence_scores=F_benchmarks, metrics=metrics_benchmarks)

		# Getting confidence scores and aggregated labels for proposed techniques
		results_proposed   = get_results_proposed(  preds_all['test']['mv'] )
		results_benchmarks = get_results_tao_sheng( preds_all['test']['simulation_0'] )

		return results_proposed, results_benchmarks


	@classmethod
	def core_measurements(cls, data, n_workers, config, feature_columns, seed, use_parallelization_benchmarks=False) -> Result2Type:
		""" Final pred labels & uncertainties for proposed technique
				dataframe = preds[train, test] * [mv] <=> {rows: samples, columns: workers}
				dataframe = uncertainties[train, test]  {rows: samples, columns: workers}

			Final pred labels for proposed benchmarks
				dataframe = preds[train, test] * [simulation_0] <=> {rows: samples,  columns: workers}

			Note: A separate boolean flag is used for use_parallelization_benchmarks.
			This will ensure we only apply parallelization when called through worker_weight_strength_relation function """

		# def merge_worker_strengths_and_weights() -> pd.DataFrame:
		# 	nonlocal workers_strength

		# 	weights_Tao_mean  = weights.TAO.mean().to_frame(MainBenchmarks.TAO.value)

		# 	return pd.concat( [workers_strength, weights.PROPOSED * n_workers, weights_Tao_mean], axis=1)

		aim1_3 = cls(data=data, config=config, feature_columns=feature_columns, n_workers=n_workers, seed=seed)

		# calculating uncertainty and predicted probability values
		preds_all, uncertainties_all, true_labels, workers_strength = aim1_3.aim1_3_meauring_probs_uncertainties()

		# Calculating weights for proposed techniques and TAO & Sheng benchmarks
		weights = aim1_3.get_weights(   workers_labels    = preds_all['test']['simulation_0'],
										preds 			  = preds_all['test']['mv'],
										uncertainties 	  = uncertainties_all['test'],
										noisy_true_labels = true_labels['test'].drop(columns=['truth']),
										n_workers		  = n_workers)

		# Calculating results for proposed techniques and benchmarks
		results_proposed, results_benchmarks = aim1_3.measuring_nu_and_confidence_score(weights     = weights,
																						preds_all   = preds_all,
																						true_labels = true_labels,
																						use_parallelization_benchmarks = use_parallelization_benchmarks)

		# merge workers_strength and weights
		# merge_worker_strengths_and_weights()

		return Result2Type( proposed		 = results_proposed,
							benchmark        = results_benchmarks,
							weight           = weights,
							workers_strength = workers_strength,
							n_workers        = n_workers,
							true_label       = true_labels )


	@staticmethod
	def wrapper(args: dict[str, Any], config, data, feature_columns) -> Tuple[dict[str, Any], Result2Type]:
		return args, AIM1_3.core_measurements(data=data, n_workers=args['nl'], config=config, feature_columns=feature_columns, seed=args['seed'], use_parallelization_benchmarks=False)


	@staticmethod
	def objective_function(config, data, feature_columns):
		return functools.partial(AIM1_3.wrapper, config=config, data=data, feature_columns=feature_columns)


	def get_outputs(self) -> Dict[str, List[ResultType]]:
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
			- If output saving is enabled in the configuration, dumps the outputs to a file.
			- Returns the populated outputs dictionary.

		2. Non-CALCULATE mode:
			- Loads the outputs directly from the specified file path.

		Returns:
			Dict[str, List[ResultType]]:
				A dictionary mapping output keys to lists containing the results for each seed.
		"""



		path = self.config.output.path / 'outputs' / f'{self.config.dataset.dataset_name}.pkl'

		def update_outputs() -> Dict[str, List[ResultType]]:
			"""
			Update and return the simulation outputs based on the current core results.

			This function constructs an output dictionary where each key follows the format 'NL{nl}'
			for each worker as specified in the simulation configuration. For each worker, it initializes
			a list of length `num_seeds` filled with NaN values using numpy's np.full. It then updates
			the list by replacing the NaN at the index corresponding to the seed (as provided in the
			core_results tuple) with the computed result value.

			If the configuration flag to save outputs is enabled (self.config.output.save), the
			outputs dictionary is dumped to a file via LoadSaveFile.

			Returns:
				Dict[str, List[ResultType]]: A dictionary mapping worker identifiers to lists
				of results updated for each seed.
			"""
			nonlocal core_results

			# Initialize output structure with NaN values
			outputs = {f'NL{nl}': np.full(self.config.simulation.num_seeds, np.nan).tolist() for nl in self.config.simulation.workers_list}

			for args, values in core_results:
				outputs[f'NL{args["nl"]}'][args['seed']] = values

			if self.config.output.save:
				LoadSaveFile(path).dump(outputs)

			return outputs

		if self.config.output.mode is OutputModes.CALCULATE:

			input_list = [{'nl': nl, 'seed': seed} for nl in self.config.simulation.workers_list for seed in range(self.config.simulation.num_seeds)]

			function = AIM1_3.objective_function(config=self.config, data=self.data, feature_columns=self.feature_columns)

			if self.config.simulation.use_parallelization:
				with multiprocessing.Pool(processes=min(self.config.simulation.max_parallel_workers, len(input_list))) as pool:
					core_results = pool.map(function, input_list)

			else:
				core_results = [function(args) for args in input_list]

			return update_outputs()

		return LoadSaveFile(path).load()


	def worker_weight_strength_relation(self, seed=0, n_workers=10) -> pd.DataFrame:

		metric_name = 'weight_strength_relation'
		path_main = self.config.output.path / metric_name / self.config.dataset.dataset_name.value
		lsf = LoadSaveFile(path_main / f'{metric_name}.xlsx')

		if self.config.output.mode is OutputModes.CALCULATE:
			params = {	'seed'           : seed,
						'n_workers'      : n_workers,
						'config'         : self.config,
						'data'           : self.data,
						'feature_columns': self.feature_columns,
						'use_parallelization_benchmarks':self.config.simulation.use_parallelization}

			df = AIM1_3.core_measurements(**params).workers_strength.set_index('workers_strength').sort_index()

			if self.config.output.save:
				lsf.dump(df, index=True)
			return df

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
	def calculate_one_dataset(cls, config: 'Settings', dataset_name: DatasetNames=DatasetNames.IONOSPHERE) -> ResultComparisonsType:
		"""
		Calculates various output metrics and analyses for a given dataset using the provided configuration.

		This method updates the configuration to specify the desired dataset, loads the corresponding data,
		computes model outputs using the instance created with the input data and configuration, and then
		evaluates the worker weight strength relationship. The results, along with the outputs and configuration,
		are encapsulated in a ResultComparisonsType object that is returned to the caller.

		Parameters:
			config (Settings): Configuration settings that include parameters, dataset details, and other experiment configurations.
			dataset_name (DatasetNames, optional): The identifier for the dataset to be processed.
				Defaults to DatasetNames.IONOSPHERE.

		Returns:
			ResultComparisonsType: An object containing:
				- weight_strength_relation: The computed relationship between worker weights and strength.
				- outputs: The outputs produced by the model after running the dataset.
				- config: The configuration that was used for the computation.

		Note:
			The method internally loads the dataset using the UCI database reader,
			creates an instance for further analysis, and computes the required metrics.
		"""

		config.dataset.dataset_name = dataset_name

		# loading the dataset
		data, feature_columns = load_data.aim1_3_read_download_UCI_database(config=config)

		aim1_3 = cls(data=data, feature_columns=feature_columns, config=config)

		# getting the outputs
		outputs = aim1_3.get_outputs()

		# measuring the confidence scores
		# findings_confidence_score = aim1_3.get_confidence_scores(outputs)

		# measuring the worker strength weight relationship for proposed and Tao
		weight_strength_relation = aim1_3.worker_weight_strength_relation(seed=0, n_workers=10 )


		return ResultComparisonsType( weight_strength_relation=weight_strength_relation, outputs=outputs, config=config )


	@classmethod
	def calculate_all_datasets(cls, config: 'Settings') -> Dict[DatasetNames, ResultComparisonsType]:
		return {dt: cls.calculate_one_dataset(dataset_name=dt, config=config) for dt in config.dataset.datasetNames}


class Aim1_3_Data_Analysis_Results:
	"""
	A class for analyzing and visualizing results from the AIM1_3 experiment.

	This class provides methods to:
	1. Load and process experimental results across multiple datasets
	2. Extract specific metrics and results based on various parameters
	3. Generate visualizations of the results
	4. Save outputs to files

	Attributes:
		results_all_datasets: Dictionary containing results for all datasets
		outputs: Processed outputs from the experiments
		accuracy: Dictionary containing frequency and beta accuracy DataFrames
		config: Configuration parameters for the analysis

	Methods:
		update(): Updates the results by calculating all datasets
		get_result(): Retrieves specific results based on provided parameters
		get_evaluation_metrics_for_confidence_scores(): Calculates evaluation metrics for confidence scores
		save_outputs(): Saves outputs to files
		figure_weight_quality_relation(): Generates and saves figures showing the relationship between weights and quality
		figure_metrics_mean_over_seeds_per_dataset_per_worker(): Creates heatmap visualizations of metrics
		figure_metrics_all_datasets_workers(): Generates boxplots of metrics across datasets and workers
		figure_F_heatmap(): Creates heatmap visualizations of confidence score evaluations
	"""

	def __init__(self, config):
		"""Initialize CrowdUtilities with the given configuration.

		This function initializes the utility class to process and analyze crowdsourced data.

		Parameters
		----------
		config : dict or Configuration object
			Configuration object containing settings for the crowd analysis,
			such as paths, parameters, and options for data processing.

		Attributes
		----------
		results_all_datasets : None
			Will store results across all datasets once processed.
		outputs : None
			Will store the output data from crowd analysis.
		accuracy : dict
			Dictionary with pandas DataFrames to store accuracy measurements:
			- 'freq': DataFrame for frequency-based accuracy metrics
			- 'beta': DataFrame for beta-based accuracy metrics
		config : dict or Configuration object
			Stored configuration object passed during initialization.
		"""

		self.results_all_datasets = None
		self.outputs  = None
		self.accuracy = dict(freq=pd.DataFrame(), beta=pd.DataFrame())
		self.config   = config

	def update(self) -> 'Aim1_3_Data_Analysis_Results':
		"""
		Update the internal state with calculated results for all datasets.

		This method triggers the calculation of results across all datasets using the
		current configuration and updates the internal state with these results.

		Returns:
			Aim1_3_Data_Analysis_Results: Returns self to allow for method chaining.
		"""
		self.results_all_datasets  = AIM1_3.calculate_all_datasets(config=self.config)
		return self


	def get_result(self, metric_name='F_all', dataset_name: DatasetNames=DatasetNames.MUSHROOM, strategy=StrategyNames.FREQ , nl='NL3', seed_ix=0, method_name=ProposedTechniqueNames.PROPOSED, data_mode='test'):
		"""
		Retrieve result metrics and data based on specified parameters.

		This function serves as a centralized access point for various metrics and evaluation data
		from crowdsourcing simulations. It handles different types of metrics including F-scores,
		aggregated labels, true labels, and evaluation metrics across different datasets, worker counts,
		and simulation seeds.

		Parameters
		----------
		metric_name : str, default='F_all'
			The type of metric or data to retrieve. Options include:
			- 'F_all', 'F_pos_all', 'F_mean_over_seeds', 'F_pos_mean_over_seeds': Confidence scores
			- 'F_eval_one_dataset_all_workers', 'F_eval_one_worker_all_datasets': Evaluation metrics for confidence scores
			- 'weight_strength_relation': Relationship between worker strength and weights
			- 'F': F-score for a specific method
			- 'aggregated_labels': Labels aggregated from workers
			- 'true_labels': Ground truth labels
			- 'metrics_per_dataset_per_worker_per_seed': Raw metrics for specific configuration
			- 'metrics_mean_over_seeds_per_dataset_per_worker': Metrics averaged over seeds
			- 'metrics_all_datasets_workers': Comprehensive metrics across all datasets and worker counts

		dataset_name : DatasetNames, default=DatasetNames.MUSHROOM
			The dataset to retrieve results for.

		strategy : StrategyNames, default=StrategyNames.FREQ
			The strategy used for aggregation or confidence calculation.

		nl : str, default='NL3'
			Worker count identifier (e.g., 'NL3' for 3 workers).

		seed_ix : int, default=0
			The simulation seed index to use.

		method_name : ProposedTechniqueNames, default=ProposedTechniqueNames.PROPOSED
			The method to retrieve results for.

		data_mode : str, default='test'
			Whether to use 'train' or 'test' data.

		Returns
		-------
		pandas.DataFrame
			The requested metrics or data according to the specified parameters.
		"""

		metrics_list = [EvaluationMetricNames.AUC, EvaluationMetricNames.ACC, EvaluationMetricNames.F1]

		def drop_proposed_rename_crowd_certain(dataframe, orient='columns'):
			"""
			Removes the proposed technique entry from the dataframe.

			This function takes a dataframe and removes the entry corresponding to the proposed technique,
			either from columns or from index depending on the orientation parameter.

			Parameters
			----------
			dataframe : pandas.DataFrame
				The input dataframe containing technique data.
			orient : str, default='columns'
				Specifies the orientation from which to drop the proposed technique.
				If 'columns', drops from columns; otherwise, drops from index.

			Returns
			-------
			pandas.DataFrame
				A new dataframe with the proposed technique entry removed.
			"""

			if orient == 'columns':
				return dataframe.drop( columns=[ProposedTechniqueNames.PROPOSED] )

			else:
				return dataframe.drop( index=[ProposedTechniqueNames.PROPOSED] )

		def get_metrics_mean_over_seeds(dataset_name1: DatasetNames, n_workers) -> pd.DataFrame:
			"""
			Calculate the mean of metrics across different random seeds for a specific dataset and number of workers.

			This function retrieves metrics for each seed from the simulation results, organizes them in a DataFrame,
			and then computes the mean value for each metric across all seeds.

			Parameters:
			-----------
			dataset_name1 : DatasetNames
				The name of the dataset for which to calculate metrics
			n_workers : int
				The number of workers in the crowdsourcing simulation

			Returns:
			--------
			pd.DataFrame
				A DataFrame containing the mean values of each metric across all seeds,
				where rows correspond to data points and columns correspond to different metrics
			"""
			seed_list = list(range(self.config.simulation.num_seeds))
			df_all    = pd.DataFrame(columns=pd.MultiIndex.from_product([seed_list, metrics_list], names=['s', 'metric']))

			for s in range( self.config.simulation.num_seeds ):
				df_all[s] = self.results_all_datasets[dataset_name1].outputs[n_workers][s].metrics.T.astype( float )[metrics_list]

			df_all = df_all.swaplevel(axis=1)

			return pd.DataFrame({m: df_all[m].mean(axis=1) for m in metrics_list}) # df_all.T.groupby(level='metric').mean().T

		if metric_name in ['F_all', 'F_pos_all', 'F_mean_over_seeds', 'F_pos_mean_over_seeds']:

			df = self.results_all_datasets[dataset_name].findings_confidence_score[metric_name][strategy][nl]

			if strategy == StrategyNames.FREQ:
				df = 1 - df

			df['truth'] = self.results_all_datasets[dataset_name].outputs[nl][seed_ix].true_labels[data_mode].truth
			return df

		elif metric_name in ['F_eval_one_dataset_all_workers', 'F_eval_one_worker_all_datasets']:
			return self.get_evaluation_metrics_for_confidence_scores(metric_name=metric_name, dataset_name=dataset_name, nl=nl)

		elif metric_name == 'weight_strength_relation': # 'df_weight_stuff'
			techniques: list[Any] = list(ProposedTechniqueNames) + [MainBenchmarks.TAO]
			value_vars     = [n.value    for n in techniques]
			rename_columns = {n: n.value for n in techniques}

			wwr = pd.DataFrame()
			for dt in self.config.dataset.datasetNames:

				df = (self.results_all_datasets[dt].weight_strength_relation
						.rename(columns=rename_columns)
						.reset_index()
						.melt(id_vars=['workers_strength'], value_vars=value_vars, var_name='Method', value_name='Weight'))

				wwr = pd.concat([wwr, df.assign(dataset_name=dt.value)], axis=0)

			return wwr


		elif metric_name in ['F', 'aggregated_labels','true_labels']:

			if metric_name == 'F':
				assert method_name in list(ProposedTechniqueNames) + list(MainBenchmarks)
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].F[strategy][method_name]

			elif metric_name in ['aggregated_labels']:
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].aggregated_labels

			elif metric_name == 'true_labels':
				assert data_mode in ['train', 'test']
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].true_labels[data_mode]

		elif 'metric' in metric_name:

			if metric_name == 'metrics_per_dataset_per_worker_per_seed':
				return self.results_all_datasets[dataset_name].outputs[nl][seed_ix].metrics.T.astype(float)

			elif metric_name == 'metrics_mean_over_seeds_per_dataset_per_worker':
				df = get_metrics_mean_over_seeds(dataset_name, nl)
				return drop_proposed_rename_crowd_certain(df, orient='index')

			elif metric_name == 'metrics_all_datasets_workers':
				workers_list = [f'NL{i}' for i in self.config.simulation.workers_list]

				columns = pd.MultiIndex.from_product([metrics_list, self.config.dataset.datasetNames, workers_list], names=['metric', 'dataset', 'workers'])
				df = pd.DataFrame(columns=columns)

				for dt in self.config.dataset.datasetNames:
					for nl in workers_list:
						df_temp = get_metrics_mean_over_seeds(dt, nl)
						df_temp = drop_proposed_rename_crowd_certain(df_temp, orient='index')

						for metric in metrics_list:
							df[(metric, dt, nl)] = df_temp[metric].copy()

				return df


	def get_evaluation_metrics_for_confidence_scores(self, metric_name='F_eval_one_dataset_all_workers', dataset_name: DatasetNames=DatasetNames.IONOSPHERE, nl='NL3'):
		"""
		Calculates evaluation metrics for confidence scores across datasets or workers.

		This method computes confidence calibration metrics such as Expected Calibration Error (ECE)
		and Brier Score for different techniques and strategies.

		Parameters
		----------
		metric_name : str, default='F_eval_one_dataset_all_workers'
			Specifies which evaluation to perform:
			- 'F_eval_one_dataset_all_workers': Evaluate one dataset across all worker counts
			- 'F_eval_one_worker_all_datasets': Evaluate one worker count across all datasets

		dataset_name : DatasetNames, default=DatasetNames.IONOSPHERE
			The dataset to use when evaluating across workers.

		nl : str, default='NL3'
			The worker count identifier to use when evaluating across datasets.

		Returns
		-------
		pandas.DataFrame
			A multi-indexed DataFrame containing ECE and Brier scores.
			- Index: Two-level MultiIndex with (strategy, technique)
			- Columns: Two-level MultiIndex with (metric, target)
			where metric is either ECE or Brier score and target varies based on the metric_name
			(either worker counts or dataset names).

		Raises
		------
		ValueError
			If metric_name is not one of the recognized options.
		"""

		from sklearn.metrics import brier_score_loss

		def ece_score(y_true, conf_scores_per_method, n_bins=10):
			"""
			Calculate the Expected Calibration Error (ECE) score.

			ECE measures the difference between confidence predictions and actual accuracy.
			It divides predictions into bins, computes the weighted average of the absolute
			differences between accuracy and confidence within each bin.

			Parameters
			----------
			y_true : numpy.ndarray
				Binary ground truth labels (0 or 1).
			conf_scores_per_method : numpy.ndarray
				Confidence scores predicted by the model, should be between 0 and 1.
			n_bins : int, default=10
				Number of bins to use when discretizing the confidence scores.

			Returns
			-------
			float
				The Expected Calibration Error score. Lower values indicate better calibration.
				A perfectly calibrated model would have an ECE of 0.

			Notes
			-----
			The implementation involves:
			1. Dividing the confidence range [0,1] into n_bins equal bins
			2. For each bin, calculating the mean accuracy and mean confidence
			3. Computing the weighted absolute difference between accuracy and confidence
			4. Summing these differences to produce the final ECE score
			"""
			bin_boundaries = np.linspace(0, 1, n_bins + 1)
			bin_lowers = bin_boundaries[:-1]
			bin_uppers = bin_boundaries[1:]

			accuracies = []
			confidences = []
			for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
				# Filter out data points from the bin
				in_bin = np.logical_and(bin_lower <= conf_scores_per_method,
										conf_scores_per_method < bin_upper)
				prop_in_bin = in_bin.mean()

				if prop_in_bin > 0:
					accuracy_in_bin   = y_true[in_bin].mean()
					confidence_in_bin = conf_scores_per_method[in_bin].mean()

					accuracies.append(accuracy_in_bin)
					confidences.append(confidence_in_bin)

			accuracies = np.array(accuracies)
			confidences = np.array(confidences)

			ece = np.sum(np.abs(accuracies - confidences) * confidences)
			return ece

		if metric_name == 'F_eval_one_dataset_all_workers':
			target_list = [f'NL{x}' for x in self.config.simulation.workers_list]
			target_name = 'n_workers'
			get_params = lambda i: dict(nl=i, dataset_name=dataset_name)

		elif metric_name == 'F_eval_one_worker_all_datasets':
			target_list = self.config.dataset.datasetNames
			target_name = 'dataset_name'
			get_params = lambda i: dict(nl=nl, dataset_name=i)

		else:
			raise ValueError('metric_name should be either workers or datasets')


		index   = pd.MultiIndex.from_product([list(StrategyNames), [ProposedTechniqueNames.PROPOSED_PENALIZED] + list(MainBenchmarks)], names=['strategy', 'technique'])

		columns = pd.MultiIndex.from_product([list(ConfidenceScoreNames), target_list], names=['metric', target_name])

		df_cs   = pd.DataFrame(columns=columns, index=index)

		techniques = [ProposedTechniqueNames.PROPOSED_PENALIZED, MainBenchmarks.TAO, MainBenchmarks.SHENG]

		for st in StrategyNames:
			for ix in target_list:
				conf_scores = self.get_result(metric_name='F_pos_mean_over_seeds', strategy=st, **get_params(ix))
				conf_scores = conf_scores[techniques + ['truth']]

				for m in techniques:
					df_cs[(ConfidenceScoreNames.ECE  ,ix)][(st, m)] = ece_score(conf_scores.truth, conf_scores[m])
					df_cs[(ConfidenceScoreNames.BRIER,ix)][(st, m)] = brier_score_loss(conf_scores.truth, conf_scores[m])

		return df_cs.astype(float)


	def save_outputs(self, filename, relative_path, dataframe=None):
		"""
		Save outputs including plots and optional dataframe to specified location.

		This method saves matplotlib plots in both PNG and PDF formats and optionally
		saves a pandas DataFrame as an Excel file at the specified path.

		Parameters:
		-----------
		filename : str
			Base name for the output files (without extension)
		relative_path : Path or str
			Path relative to the config.output.path where files will be saved
		dataframe : pandas.DataFrame, optional
			DataFrame to save as an Excel file. If None, no DataFrame is saved

		Notes:
		------
		- Files are only saved if config.output.save is True
		- Directories are created if they don't exist
		- Plots are saved with 300 dpi and tight bounding box
		"""

		if self.config.output.save:

			# output path
			path = self.config.output.path / relative_path / filename
			path.mkdir(parents=True, exist_ok=True)

			# Save the plot
			for suffix in ['png', 'pdf']:
				plt.savefig(path / f'{filename}.{suffix}', format=suffix, dpi=300, bbox_inches='tight')

			# Save the sheet
			if dataframe is not None:
				LoadSaveFile( path / f'{filename}.xlsx' ).dump( dataframe, index=True )


	def figure_weight_quality_relation(self, aspect=1.5, font_scale=1, fontsize=12, relative_path='final_figures', height=4):
		"""
		Create and save a figure showing the relationship between worker strength and estimated weight.

		This function plots the relationship between workers' strength (probability threshold)
		and their estimated weight across different datasets using seaborn's lmplot with a
		third-order polynomial fit. The figure is organized in a grid of subplots by dataset.

		Parameters
		----------
		aspect : float, default=1.5
			The aspect ratio of each subplot.
		font_scale : float, default=1
			Scaling factor for font sizes.
		fontsize : int, default=12
			Base font size for labels and titles.
		relative_path : str, default='final_figures'
			Relative path where the figure will be saved.
		height : float, default=4
			Height (in inches) of each subplot.

		Returns
		-------
		None
			The function saves the figure and corresponding data to disk but does not return any value.

		Notes
		-----
		The function retrieves data using the `get_result` method with 'weight_strength_relation'
		as the metric name, creates a faceted plot with a polynomial regression line for each dataset,
		and saves both the figure and the underlying data.
		"""

		metric_name = 'weight_strength_relation'
		df: pd.DataFrame = self.get_result(metric_name=metric_name)  # type: ignore

		sns.set_theme( palette='colorblind', style='darkgrid', context='paper', font_scale=font_scale)

		# Create a facet_kws dictionary to pass deprecated arguments
		facet_kws = {'sharex': True, 'sharey': True, 'legend_out': False}

		p = sns.lmplot(data=df, legend=True, hue='Method', order=3, x="workers_strength", y="Weight", col='dataset_name', col_wrap=3, height=height, aspect=aspect, ci=None, facet_kws=facet_kws)

		p.set_xlabels(r"Probability Threshold ($\pi_\alpha^{(k)}$)" , fontsize=fontsize)
		p.set_ylabels(r"Estimated Weight ($\omega_\alpha^{(k)}$)"   , fontsize=fontsize)
		p.set_titles(col_template="{col_name}", fontweight='bold'   , fontsize=fontsize)
		p.fig.suptitle('Estimated Weight vs. Probability Threshold'  ,  fontsize=int(1.5*fontsize), fontweight='bold')
		p.tight_layout()
		sns.move_legend(p,"lower right", bbox_to_anchor=(0.75, 0.1) , bbox_transform=p.fig.transFigure)

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}', relative_path=relative_path, dataframe=df )


	def figure_metrics_mean_over_seeds_per_dataset_per_worker(self, metric: EvaluationMetricNames=EvaluationMetricNames.ACC, nl=3, figsize=(10, 10), font_scale=1.8, fontsize=20, relative_path='final_figures'):
		"""
		Creates and saves a figure displaying the mean of a specified evaluation metric across seeds for each dataset per worker.

		This function generates a heatmap that visualizes the mean performance metrics across different datasets and workers.
		The results are calculated by averaging across all random seeds for each dataset and worker combination.

		Parameters
		----------
		metric : EvaluationMetricNames, default=EvaluationMetricNames.ACC
			The evaluation metric to be plotted (e.g., accuracy, F1 score, etc.)
		nl : int, default=3
			The number of workers/labels per example to use for the analysis
		figsize : tuple, default=(10, 10)
			The size of the figure in inches (width, height)
		font_scale : float, default=1.8
			The scale factor to apply to the font size
		fontsize : int, default=20
			The base font size for the figure
		relative_path : str, default='final_figures'
			The relative path where the figure and dataframe will be saved

		Returns
		-------
		None
			The function saves the figure and the corresponding dataframe to the specified location
			but does not return any values.

		Notes
		-----
		- The function retrieves results using the 'metrics_mean_over_seeds_per_dataset_per_worker' metric name
		- The heatmap is created with the datasets as rows and workers as columns
		- Results are formatted with 2 decimal places in the heatmap annotations
		- The figure title includes the metric name and number of workers
		"""

		metric_name='metrics_mean_over_seeds_per_dataset_per_worker'

		# df = pd.DataFrame(columns=self.config.dataset.datasetNames)
		# for dt in self.config.dataset.datasetNames:
		df = pd.DataFrame({dt: self.get_result( metric_name=metric_name, dataset_name=dt, nl=f'NL{nl}')[metric] for dt in self.config.dataset.datasetNames})

		fig = plt.figure(figsize=figsize)
		sns.set(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')
		sns.heatmap(df.T, annot=True, fmt='.2f', cmap='Blues',  cbar=True, robust=True)

		fig.suptitle(f'{metric} for NL{nl} ({nl} Workers)', fontsize=int(1.5*fontsize), fontweight='bold')
		plt.tight_layout()

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}_{metric}', relative_path=relative_path, dataframe=df )


	def figure_metrics_all_datasets_workers(self, workers_list: list[str]=None, figsize=(15, 15), font_scale=1.8, fontsize=20, relative_path='final_figures'):
		"""
		Generate boxplots of evaluation metrics (Accuracy, AUC, F1) for all datasets across specified workers.

		This method visualizes performance metrics for each worker across all datasets, creating a grid of boxplots
		where each row represents a worker and each column represents a different metric.

		Parameters
		----------
		workers_list : list[str], optional
			List of worker identifiers to include in the visualization.
			If None, uses workers from config.simulation.workers_list with 'NL' prefix.
		figsize : tuple, default (15, 15)
			Figure size as (width, height) in inches.
		font_scale : float, default 1.8
			Scaling factor for font sizes in seaborn plots.
		fontsize : int, default 20
			Font size for axis labels and titles.
		relative_path : str, default 'final_figures'
			Relative path to save the output figure and data.

		Returns
		-------
		None
			The figure is displayed and saved to the specified path.

		Notes
		-----
		The method retrieves metrics from 'metrics_all_datasets_workers' result,
		creates boxplots for each worker-metric combination, and saves both the
		visualization and the underlying data.
		"""

		def get_axes(ix1, ix2):
			nonlocal axes
			if len(workers_list) == 1 or len(metrics_list) == 1:
				return axes[max( ix1, ix2 )]
			else:
				return axes[ix1, ix2]


		if workers_list is None:
			workers_list = [f'NL{i}' for i in self.config.simulation.workers_list]

		sns.set(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')

		metric_name  = 'metrics_all_datasets_workers'
		metrics_list = [EvaluationMetricNames.ACC, EvaluationMetricNames.AUC, EvaluationMetricNames.F1]

		df: pd.DataFrame = self.get_result(metric_name=metric_name) # type: ignore
		df = df.swaplevel(axis=1, i=1, j=2)

		fig, axes = plt.subplots(nrows=len(workers_list), ncols=len(metrics_list), figsize=figsize, sharex=True, sharey=True, squeeze=True)
		for i2, metric in enumerate(metrics_list):
			# df_per_nl = df[metric].T.groupby(level=1)
			for i1, nl in enumerate(workers_list):
				data = df[metric][nl].rename(columns=lambda c: c.value)
				# sns.boxplot(data=df_per_nl.get_group(nl).T, orient='h', ax=ax)
				sns.boxplot( data=data, orient='h', ax=get_axes( i1, i2 ) )

		i1 = 2 if len(workers_list) > 2 else len(workers_list)
		for i2, metric in enumerate(metrics_list):
			get_axes( i1, i2 ).set_xlabel( metric, fontsize=fontsize, fontweight='bold', labelpad=20 )

		for i1, nl in enumerate(workers_list):
			get_axes( i1, 0 ).set_ylabel( nl, fontsize=fontsize, fontweight='bold', labelpad=20 )

		fig.suptitle('Metrics Over All Datasets', fontsize=int(1.5*fontsize), fontweight='bold')
		plt.tight_layout()

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}', relative_path=relative_path, dataframe=df )


	def figure_F_heatmap(self, metric_name='F_eval_one_dataset_all_workers', dataset_name:DatasetNames=DatasetNames.IONOSPHERE, nl='NL3', fontsize=20, font_scale=1.8, figsize=(13, 11), relative_path='final_figures'):
		"""
		Create a heatmap visualization of confidence score evaluations.

		This method generates a 2x2 grid of heatmaps comparing confidence score performance
		metrics (ECE and Brier) between different strategies (FREQ and BETA) for either all
		workers on a specific dataset or all datasets for a specific worker level.

		Parameters
		----------
		metric_name : str, optional
			The type of evaluation to perform, either 'F_eval_one_dataset_all_workers'
			or 'F_eval_one_worker_all_datasets'. Default is 'F_eval_one_dataset_all_workers'.

		dataset_name : DatasetNames, optional
			Dataset to evaluate when using 'F_eval_one_dataset_all_workers'.
			Default is DatasetNames.IONOSPHERE.

		nl : str, optional
			Worker noise level to evaluate when using 'F_eval_one_worker_all_datasets'.
			Default is 'NL3'.

		fontsize : int, optional
			Base font size for the plot labels. Default is 20.

		font_scale : float, optional
			Scale factor for the seaborn font style. Default is 1.8.

		figsize : tuple, optional
			Figure dimensions (width, height) in inches. Default is (13, 11).

		relative_path : str, optional
			Directory path where the output figure and data will be saved.
			Default is 'final_figures'.

		Returns
		-------
		None
			The method saves the figure and associated data to disk but doesn't return any values.

		Notes
		-----
		The output visualization displays ECE (Expected Calibration Error) scores in the
		top row with a red colormap, and Brier scores in the bottom row with a blue colormap.
		FREQ strategy results are shown in the left column and BETA strategy results in the right column.
		"""

		def get_filename_subtitle():
			nonlocal filename, subtitle

			if metric_name == 'F_eval_one_dataset_all_workers':
				filename  = f'heatmap_F_evals_{dataset_name}_all_workers'
				subtitle = f'Confidence Score Evaluation for {dataset_name}'

			elif metric_name == 'F_eval_one_worker_all_datasets':
				filename  = f'heatmap_F_evals_all_datasets_{nl}'
				subtitle = f'Confidence Score Evaluation for {nl.split("NL")[1]} Workers'

			else:
				raise ValueError('metric_name does not exist')

			return filename, subtitle

		def create_heatmap(data, ax, cmap, cbar, title='', ylabel='', xlabel=''):
			sns.heatmap(data, ax=ax, annot=True, fmt='.2f', cmap=cmap, cbar=cbar, robust=True)
			ax.set_title(title, fontsize=fontsize, fontweight='bold')
			ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', labelpad=20)
			ax.set_xlabel(xlabel)

		def plot():
			"""
			Generates a 2x2 grid of heatmaps for different calibration scores and strategies.

			This function configures the plotting theme using seaborn and creates a figure with four subplots arranged in two rows and two columns. It then plots heatmaps using the preprocessed data from the nonlocal variable 'df' for two different confidence scores (e.g., ECE and BRIER) and two strategies (e.g., FREQ and BETA) by calling the helper function 'create_heatmap'. The heatmaps for each combination are styled with different color maps ('Reds' for ECE and 'Blues' for BRIER) and some include color bars as specified. Finally, the overall figure title is set using the nonlocal variable 'subtitle' with bold styling, and the layout is tightened for optimal display.

			Nonlocal Variables:
				df (DataFrame): Contains the data for the different calibration scores and strategies. It is expected to have keys corresponding to score names and strategy names.
				subtitle (str): The subtitle to be used as the figure title.
				Other nonlocal variables assumed to be present in the enclosing scope include:
					- font_scale (float): Scale of the font for the plot.
					- figsize (tuple): Size of the figure.
					- fontsize (int or float): Base size for the font used in the title.

			Returns:
				None. The function creates and displays a matplotlib figure with the generated heatmaps.
			"""
			nonlocal df, subtitle

			sns.set_theme(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')
			fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True, sharey=True, squeeze=True)

			# Create heatmaps in a loop
			create_heatmap(data=df[ConfidenceScoreNames.ECE].T[StrategyNames.FREQ],  ax=axes[0, 0], cmap='Reds', cbar=False, title=StrategyNames.FREQ.name, ylabel=ConfidenceScoreNames.ECE.name)
			create_heatmap(data=df[ConfidenceScoreNames.ECE].T[StrategyNames.BETA],  ax=axes[0, 1], cmap='Reds', cbar=True,  title=StrategyNames.BETA.name)
			create_heatmap(data=df[ConfidenceScoreNames.BRIER].T[StrategyNames.FREQ], ax=axes[1, 0], cmap='Blues', cbar=False, ylabel=ConfidenceScoreNames.BRIER.name)
			create_heatmap(data=df[ConfidenceScoreNames.BRIER].T[StrategyNames.BETA], ax=axes[1, 1], cmap='Blues', cbar=True)

			fig.suptitle(subtitle, fontsize=int(1.5*fontsize), fontweight='bold')
			plt.tight_layout()


		filename, subtitle = get_filename_subtitle()

		df = self.get_result(metric_name=metric_name, dataset_name=dataset_name, nl=nl).round(3)

		plot()

		self.save_outputs( filename=f'figure_{filename}', relative_path=relative_path, dataframe=df.T )


class AIM1_3_Plot:
	"""
	A class for plotting data in a DataFrame format, with options for smoothing and visualization.

	This class provides various methods for manipulating and visualizing data, including
	interpolation and smoothing of data points, customizing plot appearance, and displaying
	markers for specific techniques.

	Attributes:
		plot_data (pd.DataFrame): The data to be plotted.
		weight_strength_relation_interpolated (pd.DataFrame or None): Interpolated data after plotting.

	Methods:
		__init__(plot_data): Initializes the plotting class with the provided data.
		data_interpolation(x, y, smooth, interpolation_pt_count): Static method that performs
			data interpolation and smoothing.
		plot(xlabel, ylabel, xticks, title, legend, smooth, interpolation_pt_count, show_markers):
			Creates and displays the plot with the specified parameters.
		_show(x, xnew, y_smooth, xlabel, ylabel, xticks, title): Static method for configuring and
			displaying plot elements.
		_legend(legend, columns): Static method for adding legend to the plot.
		_fixing_x_axis(index): Static method for formatting the x-axis values.
		_show_markers(show_markers, columns, x, y): Static method for displaying markers on
			specific points in the plot.
	"""

	def __init__(self, plot_data: pd.DataFrame):

		self.weight_strength_relation_interpolated = None
		assert type(plot_data) == pd.DataFrame, 'plot_data must be a pandas DataFrame'

		self.plot_data = plot_data

	@staticmethod
	def data_interpolation(x, y, smooth=False, interpolation_pt_count=1000):
		"""
		Interpolate data points and optionally apply smoothing.

		Parameters
		----------
		x : numpy.ndarray
			The x-coordinates of the data points.
		y : numpy.ndarray
			The y-coordinates of the data points. Should have shape (n, m) where
			n is the number of data points and m is the number of dimensions.
		smooth : bool, optional
			Whether to apply smoothing to the data. Default is False.
		interpolation_pt_count : int, optional
			Number of points to use for interpolation if smoothing is applied. Default is 1000.

		Returns
		-------
		xnew : numpy.ndarray
			The x-coordinates of the interpolated data.
		y_smooth : numpy.ndarray
			The y-coordinates of the interpolated and smoothed data.

		Notes
		-----
		When smoothing is enabled, the function attempts to use a spline or convolution method.
		If an exception occurs during smoothing, the original data is returned unmodified.
		The 'kernel_regression' method is commented out in the current implementation.
		"""
		xnew, y_smooth = x, y

		if smooth:
			SMOOTH_METHOD = 'kernel_regression'

			try:

				if SMOOTH_METHOD == 'spline':

					xnew = np.linspace(x.min(), x.max(), interpolation_pt_count)
					spl = make_interp_spline(x, y, k=2)
					y_smooth = spl(xnew)

				elif SMOOTH_METHOD == 'conv':

					filter_size = 5
					filter_array = np.ones(filter_size) / filter_size
					xnew = x.copy()
					y_smooth = np.zeros(list(xnew.shape) + [2])
					for j in range(y.shape[1]):
						y_smooth[:, j] = np.convolve(y[:, j], filter_array, mode='same')

				# elif SMOOTH_METHOD == 'kernel_regression':

				#     xnew = np.linspace(thresh_technique.min(), thresh_technique.max(), interpolation_pt_count)
				#     y_smooth = np.zeros(list(xnew.shape) + [y.shape[1]])
				#     for j in range(y.shape[1]):
				#         kr = statsmodels.nonparametric.kernel_regression.KernelReg(y[:, j], thresh_technique, 'c')
				#         y_smooth[:, j], _ = kr.fit(xnew)

			except Exception as e:
				print(e)
				xnew, y_smooth = x, y

		return xnew, y_smooth

	def plot(self, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, interpolation_pt_count=1000, show_markers=ProposedTechniqueNames.PROPOSED):
		"""
		Plot the weight-strength relationship data.

		This method takes the plot data stored in the instance and creates a plot of the weight-strength relationship.
		It can interpolate and smooth the data, add markers for specific techniques, and set various plot attributes.

		Parameters:
		-----------
		xlabel : str, optional
			Label for the x-axis, by default ''
		ylabel : str, optional
			Label for the y-axis, by default ''
		xticks : bool, optional
			Whether to display x-axis ticks, by default True
		title : str, optional
			Title for the plot, by default ''
		legend : list or None, optional
			Custom legend labels. If None, uses column names from plot_data, by default None
		smooth : bool, optional
			Whether to smooth the plotted data, by default True
		interpolation_pt_count : int, optional
			Number of points to use for interpolation when smoothing, by default 1000
		show_markers : str or list, optional
			Which technique(s) to show markers for. Uses ProposedTechniqueNames.PROPOSED by default

			Returns:
		-------
		None
			The plot is displayed but not returned.

		Notes:
		------
		The interpolated data is stored in the weight_strength_relation_interpolated attribute as a pandas DataFrame.
		"""

		columns = self.plot_data.columns.to_list()
		y       = self.plot_data.values.astype(float)
		x       = self._fixing_x_axis(index=self.plot_data.index)

		xnew, y_smooth = AIM1_3_Plot.data_interpolation(x=x, y=y, smooth=smooth, interpolation_pt_count=interpolation_pt_count)

		self.weight_strength_relation_interpolated = pd.DataFrame(y_smooth, columns=columns, index=xnew)
		self.weight_strength_relation_interpolated.index.name = 'workers_strength'

		plt.plot(xnew, y_smooth)
		self._show_markers(show_markers=show_markers, columns=columns, x=x, y=y)

		self._show(x=x, xnew=xnew, y_smooth=y_smooth, xlabel=xlabel, ylabel=ylabel, xticks=xticks, title=title, )
		self._legend(legend=legend, columns=columns)

	@staticmethod
	def _show(x, xnew, y_smooth, xlabel, ylabel, xticks, title):

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.grid()

		if xticks:
			plt.xticks(xnew)

		plt.show()

		if xticks:
			plt.xticks(x)

		plt.ylim(y_smooth.min() - 0.1, max(1, y_smooth.max()) + 0.1)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.grid(True)

	@staticmethod
	def _legend(legend, columns):

		if legend is None:
			pass
		elif legend == 'empty':
			plt.legend()
		else:
			plt.legend(columns, **legend)

	@staticmethod
	def _fixing_x_axis(index):
		return index.map(lambda x: int(x.replace('NL', ''))) if isinstance(index[0], str) else index.to_numpy()

	@staticmethod
	def _show_markers(show_markers, columns, x, y):
		if show_markers in (ProposedTechniqueNames.PROPOSED, True):
			cl = [i for i, x in enumerate(columns) if (ProposedTechniqueNames.PROPOSED in x) or ('method' in x)]
			plt.plot(x, y[:, cl], 'o')

		elif show_markers == 'all':
			plt.plot(x, y, 'o')

