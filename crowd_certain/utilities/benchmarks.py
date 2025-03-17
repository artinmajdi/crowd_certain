
# @functools.cache
import functools
import multiprocessing
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from crowd_certain.utilities.config import params


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

	def __init__(self, crowd_labels: Dict[str, pd.DataFrame], ground_truth: Dict[str, pd.DataFrame]):
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
	def get_techniques(cls, benchmark_name: params.OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray):
		"""
		Apply a specific benchmark technique to the test data.

		Parameters
		----------
		benchmark_name (params.OtherBenchmarkNames): The benchmark technique to apply
		test : pd.DataFrame
			Test data containing worker labels
		test_unique : np.ndarray
			Unique items in the test set

		Returns
		-------
		function
			The appropriate function for the specified benchmark technique
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
	def wrapper(benchmark_name: params.OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray) -> Tuple[params.OtherBenchmarkNames, np.ndarray]:
		"""
		Wrapper function to apply a benchmark technique and return its name with results.

		This static method applies the specified benchmark technique and pairs its name
		with the resulting predictions.

		Parameters:
			benchmark_name (params.OtherBenchmarkNames): The benchmark technique to apply
			test (pd.DataFrame): The test data in the crowdkit format
			test_unique (np.ndarray): Unique task IDs in the test data

		Returns:
			tuple: A tuple containing (benchmark_name, aggregated_labels)
		"""
		return benchmark_name, BenchmarkTechniques.get_techniques( benchmark_name=benchmark_name, test=test, test_unique=test_unique )


	@staticmethod
	def objective_function(test: pd.DataFrame, test_unique: np.ndarray) -> Callable[[params.OtherBenchmarkNames], Tuple[params.OtherBenchmarkNames, np.ndarray]]:
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
	def apply(cls, true_labels: Dict[str, pd.DataFrame], use_parallelization_benchmarks: bool) -> pd.DataFrame:
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


	def calculate(self, use_parallelization_benchmarks: bool) -> pd.DataFrame:
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
			with multiprocessing.Pool(processes=len(params.OtherBenchmarkNames)) as pool:
				output = pool.map(function, list(params.OtherBenchmarkNames))

		else:
			output = [function(benchmark_name=m) for m in params.OtherBenchmarkNames]

		return pd.DataFrame({benchmark_name.value: aggregated_labels for benchmark_name, aggregated_labels in output})


	@staticmethod
	def reshape_dataframe_into_this_sdk_format(df_predicted_labels: pd.DataFrame) -> pd.DataFrame:
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

