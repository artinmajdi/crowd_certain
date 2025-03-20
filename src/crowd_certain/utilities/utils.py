# Standard library imports
import functools
import multiprocessing
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from crowd_certain.utilities.parameters.settings import Settings

USE_HD5F_STORAGE = True


class ClassifierTraining:
	def __init__(self, config: 'Settings'):
		self.config = config

	def train_classifier(self, sim_num: int):
		pass

	def get_classifier(self, sim_num: int, random_state: int=0):
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
			return self.get_random_forest_ensemble(random_state=random_state, sim_num=sim_num)

		elif self.config.simulation.simulation_methods is params.SimulationMethods.MULTIPLE_CLASSIFIERS:
			return self.classifiers_list[sim_num]

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
	def get_random_forest_ensemble(self, random_state: int, sim_num: int):
		return sk_ensemble.RandomForestClassifier(n_estimators=4, max_depth=4, random_state=random_state * sim_num) # n_estimators=4, max_depth=4

	@property
	def classifiers_list(self):

		from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
		from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
		from sklearn.naive_bayes import GaussianNB
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.neural_network import MLPClassifier
		from sklearn.svm import SVC
		from sklearn.tree import DecisionTreeClassifier

		return {
			1: KNeighborsClassifier(3),
			2: SVC(gamma=2, C=1),
			3: DecisionTreeClassifier(max_depth=5),
			4: RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
			5: MLPClassifier(alpha=1, max_iter=1000),
			6: AdaBoostClassifier(),
			7: GaussianNB(),
			8: QuadraticDiscriminantAnalysis()}


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

		df_uncertainties = pd.DataFrame(columns=[tech.value for tech in self.config.technique.uncertainty_techniques], index=df.index)

		for tech in self.config.technique.uncertainty_techniques:

			if tech is params.UncertaintyTechniques.STD:
				df_uncertainties[tech.value] = df.std( axis=1 )

			elif tech is params.UncertaintyTechniques.ENTROPY:
				# Normalize each row to sum to 1
				df_normalized = df.div(df.sum(axis=1) + epsilon, axis=0)
				# Calculate entropy
				entropy = - (df_normalized * np.log(df_normalized + epsilon)).sum(axis=1)
				# entropy = df.apply(lambda x: scipy.stats.entropy(x), axis=1).fillna(0)

				# normalizing entropy values to be between 0 and 1
				df_uncertainties[tech.value] = entropy / np.log(df.shape[1])

			elif tech is params.UncertaintyTechniques.CV:
				# The coefficient of variation (CoV) is a measure of relative variability. It is defined as the ratio of the standard deviation. CoV doesn't have an upper bound, but it's always non-negative. Normalizing CoV to a range of [0, 1] isn't straightforward because it can theoretically take any non-negative value. A common approach is to use a transformation that asymptotically approaches 1 as CoV increases. However, the choice of transformation can be somewhat arbitrary and may depend on the context of your data. One simple approach is to use a bounded function like the hyperbolic tangent:

				coefficient_of_variation = df.std(axis=1) / (df.mean(axis=1) + epsilon)
				df_uncertainties[tech.value] = np.tanh(coefficient_of_variation)

			elif tech is params.UncertaintyTechniques.PI:
				df_uncertainties[tech.value] = df.apply(lambda row: np.percentile(row.astype(int), 75) - np.percentile(row.astype(int), 25), axis=1)

			elif tech is params.UncertaintyTechniques.CI:
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
			upper_level = [tech.value for tech in self.config.technique.consistency_techniques]

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
			if tech is params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY:
				consistency[tech.value] = 1 - uncertainty

			elif tech is params.ConsistencyTechniques.ONE_DIVIDED_BY_UNCERTAINTY:
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

			The Series is indexed by values from the params.EvaluationMetricNames enum.
			If truth has no valid values or is not binary, the metrics will be NaN.

		Notes:
		------
		- The aggregated labels are thresholded at 0.5 to convert to binary predictions
		- Metrics are only calculated if there are non-null values in truth and truth contains exactly 2 unique values
		"""

		metrics = pd.Series(index=params.EvaluationMetricNames.values())

		non_null = ~truth.isnull()
		truth_notnull = truth[non_null].to_numpy()

		if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):

			yhat = (aggregated_labels > 0.5).astype(int)[non_null]
			metrics[params.EvaluationMetricNames.AUC.value] = sk_metrics.roc_auc_score( truth_notnull, yhat)
			metrics[params.EvaluationMetricNames.ACC.value] = sk_metrics.accuracy_score(truth_notnull, yhat)
			metrics[params.EvaluationMetricNames.F1.value ] = sk_metrics.f1_score( 	 truth_notnull, yhat)

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

				columns = pd.MultiIndex.from_product([workers_strength.index, [tech.value for tech in self.config.technique.uncertainty_techniques]], names=['worker', 'uncertainty_technique'])

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

				For params.SimulationMethods.RANDOM_STATES, it uses a RandomForestClassifier with varying random states.
				For params.SimulationMethods.MULTIPLE_CLASSIFIERS, it selects different classifier types from a predefined list.
				"""

				nonlocal predicted_labels_all_sims

				# training a random forest on the aformentioned labels
				classifier = self.classifier_training.get_classifier(sim_num=sim_num, random_state=self.seed)

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

				for sim_num in range(self.classifier_training.n_simulations):
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
														columns = pd.MultiIndex.from_product([worker_0, ...], params.UncertaintyTechniques.values()))

		Returns:
			weights = pd.DataFrame( index = pd.MultiIndex.from_product([params.ConsistencyTechniques.values(),
																		params.UncertaintyTechniques.values(),
																		params.ProposedTechniqueNames.values()]),
									columns = ['worker_0', .... ])
		"""

		# TODO This is the part where I should measure the prob_mv_binary for different # of workers instead of all of them
		prob_mv_binary = preds.mean(axis=1) > 0.5

		T1    = self.calculate_consistency( uncertainties )
		T2    = T1.copy()

		proposed_techniques = [tech.value for tech in params.ProposedTechniqueNames]
		w_hat = pd.DataFrame(index=proposed_techniques, columns=T1.columns, dtype=float)
		# w_hat2 = pd.Series(index=T1.columns)

		for worker in preds.columns:

			T2.loc[preds[worker].values != prob_mv_binary.values, worker] = 0

			w_hat[worker] = pd.DataFrame.from_dict({proposed_techniques[0]: T1[worker].mean(axis=0),
													proposed_techniques[1]: T2[worker].mean(axis=0)}, orient='index')

			# w_hat.loc[proposed_techniques[0],worker] = T1[worker].mean(axis=0)
			# w_hat[worker].iloc[proposed_techniques[1]] = T2[worker].mean(axis=0)

		# w_hat = pd.DataFrame([w_hat1, w_hat2], index=list(params.ProposedTechniqueNames)).T

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

			def get_I(row):
				return bdtrc(row['k'], row['n'], 0.5)

			out['I'] = out.apply(get_I, axis=1)

			# out['I'] = np.nan
			# for index in out['n'].index:
			# 	out['I'][index] = bdtrc(out['k'][index], out['n'][index], 0.5)

			def get_F_lambda(row):
				return max(row['I'], 1-row['I'])

			# F = I.copy()
			# F[I < 0.5] = (1 - F)[I < 0.5]
			return pd.DataFrame({'F': out.apply(get_F_lambda, axis=1), 'F_pos': out['I']})

		columns = pd.MultiIndex.from_product([params.StrategyNames.values(), ['F', 'F_pos']], names=['strategies', 'F_F_pos'])
		confidence_scores = pd.DataFrame(columns=columns, index=delta.index)
		confidence_scores[params.StrategyNames.FREQ.value] = get_freq()
		confidence_scores[params.StrategyNames.BETA.value] = get_beta()

		return confidence_scores


	def get_weights(self, workers_labels, preds, uncertainties, noisy_true_labels, n_workers) -> params.WeightType:
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
		weights_proposed = self.aim1_3_measuring_proposed_weights( preds=preds, uncertainties=uncertainties)

		# Benchmark accuracy measurement
		weights_Tao = self.measuring_Tao_weights_based_on_actual_labels( workers_labels=workers_labels, noisy_true_labels=noisy_true_labels, n_workers=n_workers)

		weights_Sheng = pd.DataFrame(1 / n_workers, index=weights_Tao.index, columns=weights_Tao.columns)

		return params.WeightType(PROPOSED=weights_proposed, TAO=weights_Tao, SHENG=weights_Sheng)


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

			confidence_scores = {}

			metrics = pd.DataFrame(columns=weights.PROPOSED.index, index=params.EvaluationMetricNames.values())

			for cln in weights.PROPOSED.index:
				agg_labels[cln] = (preds * weights.PROPOSED.T[cln]).sum(axis=1)

				confidence_scores[cln] = AIM1_3.calculate_confidence_scores(delta=preds, w=weights.PROPOSED.T[cln], n_workers=self.n_workers )

				# Measuring the metrics
				metrics[cln] = AIM1_3.get_AUC_ACC_F1(aggregated_labels=agg_labels[cln], truth=true_labels['test'].truth)

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

					confidence_scores[m.value] = AIM1_3.calculate_confidence_scores( delta=workers_labels, w=w, n_workers=self.n_workers )

				return agg_labels, confidence_scores

			v_benchmarks, F_benchmarks = get_v_F_Tao_sheng()

			v_other_benchmarks = BenchmarkTechniques.apply(true_labels=true_labels, use_parallelization_benchmarks=use_parallelization_benchmarks)

			v_benchmarks = pd.concat( [v_benchmarks, v_other_benchmarks], axis=1)

			# Measuring the metrics
			metrics_benchmarks = pd.DataFrame({cln: AIM1_3.get_AUC_ACC_F1(aggregated_labels=v_benchmarks[cln], truth=true_labels['test'].truth) for cln in v_benchmarks.columns})

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

		# def merge_worker_strengths_and_weights() -> pd.DataFrame:
		# 	nonlocal workers_strength

		# 	weights_Tao_mean  = weights.TAO.mean().to_frame(params.MainBenchmarks.TAO.value)

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

		return params.Result2Type( proposed		 = results_proposed,
							benchmark        = results_benchmarks,
							weight           = weights,
							workers_strength = workers_strength,
							n_workers        = n_workers,
							true_label       = true_labels )


	@staticmethod
	def wrapper(args: Dict, config: Settings, data, feature_columns) -> Tuple[Dict, params.Result2Type]:
		return args, AIM1_3.core_measurements(data=data, n_workers=args['nl'], config=config, feature_columns=feature_columns, seed=args['seed'], use_parallelization_benchmarks=False)


	@staticmethod
	def objective_function(config: Settings, data, feature_columns) -> Callable[[Dict], Tuple[Dict, params.Result2Type]]:
		return functools.partial(AIM1_3.wrapper, config=config, data=data, feature_columns=feature_columns)


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

			function = AIM1_3.objective_function(config=self.config, data=self.data, feature_columns=self.feature_columns)

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

			df = AIM1_3.core_measurements(**params_dict).workers_strength.set_index('workers_strength').sort_index()

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

