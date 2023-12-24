from collections import defaultdict
import enum
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import time
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from crowdkit import aggregation as crowdkit_aggregation
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.special import bdtrc
from sklearn import ensemble as sk_ensemble, metrics as sk_metrics
from tqdm import tqdm
import functools
import itertools
import scipy

from crowd_certain.utilities import load_data
from crowd_certain.utilities.params import ConfidenceScoreNames, DatasetNames, EvaluationMetricNames, MainBenchmarks, \
	OtherBenchmarkNames, ProposedTechniqueNames, StrategyNames, UncertaintyTechniques
from crowd_certain.utilities.settings import OutputModes, Settings


class LoadSaveFile:
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

	def __init__(self, crowd_labels, ground_truth): # type: (Dict, Dict) -> None
		self.ground_truth          = ground_truth
		self.crowd_labels          = crowd_labels
		self.crowd_labels_original = crowd_labels.copy()

		for mode in ['train', 'test']:
			self.crowd_labels[mode] = self.reshape_dataframe_into_this_sdk_format(self.crowd_labels[mode])

	@classmethod
	def get_techniques(cls, benchmark_name: OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray):

		def exception_handler(func):
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
				The algorithm operates on real-valued task messages $x_{i \rightarrow j}$  and
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
			return crowdkit_aggregation.MACE(n_iter=10).fit_predict_proba(test)[1]

		@exception_handler
		def MajorityVote():
			return crowdkit_aggregation.MajorityVote().fit_predict(test)

		@exception_handler
		def MMSR():
			return crowdkit_aggregation.MMSR().fit_predict(test)

		@exception_handler
		def Wawa():
			return crowdkit_aggregation.Wawa().fit_predict_proba(test)[1]

		@exception_handler
		def ZeroBasedSkill():
			return crowdkit_aggregation.ZeroBasedSkill().fit_predict_proba(test)[1]

		@exception_handler
		def GLAD():
			return crowdkit_aggregation.GLAD().fit_predict_proba(test)[1]

		@exception_handler
		def DawidSkene():
			return crowdkit_aggregation.DawidSkene().fit_predict(test)


		return eval(benchmark_name.value)()


	@staticmethod
	def wrapper(benchmark_name: OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray) -> Tuple[OtherBenchmarkNames, np.ndarray]:
		return (benchmark_name, BenchmarkTechniques.get_techniques(benchmark_name=benchmark_name, test=test, test_unique=test_unique))


	@staticmethod
	def objective_function(test, test_unique):
		return functools.partial(BenchmarkTechniques.wrapper, test=test, test_unique=test_unique)


	@classmethod
	def apply(cls, true_labels, use_parallelization_benchmarks) -> pd.DataFrame:
		ground_truth = {n: true_labels[n].truth.copy() 				     for n in ['train', 'test']}
		crowd_labels = {n: true_labels[n].drop(columns=['truth']).copy() for n in ['train', 'test']}
		return cls(crowd_labels=crowd_labels, ground_truth=ground_truth).calculate(use_parallelization_benchmarks)


	def calculate(self, use_parallelization_benchmarks) -> pd.DataFrame:
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

		aggregated_labels = {}
		for benchmark_name, values in output:
			aggregated_labels[benchmark_name] = values

		return pd.DataFrame(aggregated_labels) #, index=test_unique)


	# def calculate(self, use_parallelization_benchmarks) -> pd.DataFrame:
	# 	""" Apply all benchmarks to the input dataset """

	@staticmethod
	def reshape_dataframe_into_this_sdk_format(df_predicted_labels):
		"""  Preprocessing the data to adapt to the sdk structure:
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
class ResultsType:
	true_labels      : Dict[str, pd.DataFrame]
	F                : Dict[StrategyNames, Dict[enum.Enum, pd.DataFrame]]
	aggregated_labels: pd.DataFrame
	weights_proposed : pd.DataFrame
	weights_Tao      : pd.DataFrame
	workers_strength: pd.DataFrame
	n_workers       : int
	metrics 		 : pd.DataFrame


@dataclass
class ResultsComparisonsType:
	outputs                 : dict
	config                   : 'Settings'
	findings_confidence_score : dict
	weight_strength_relation : pd.DataFrame


class AIM1_3:

	def __init__(self):
		pass

	@staticmethod
	def get_accuracy(aggregated_labels, n_workers, delta_benchmark, truth):
		""" Measuring accuracy. This result in the same values as if I had measured a weighted majority voting using the "weights" multiplied by "delta" which is the binary predicted labels """

		accuracy = pd.DataFrame(index=[n_workers])
		for methods in [ProposedTechniqueNames, MainBenchmarks, OtherBenchmarkNames]:
			for m in methods:
				accuracy[m] = ((aggregated_labels[m] >= 0.5) == truth).mean(axis=0)

		accuracy['MV_Classifier'] = ( (delta_benchmark.mean(axis=1) >= 0.5) == truth).mean(axis=0)

		return accuracy


	@classmethod
	def core_measurements(cls, data, n_workers, config, feature_columns, seed, use_parallelization_benchmarks=False) -> ResultsType:
		""" Final pred labels & uncertainties for proposed technique
				dataframe = preds[train, test] * [mv]              <=> {rows: samples,  columns: workers}
				dataframe = uncertainties[train, test]  {rows: samples,  columns: workers}

			Final pred labels for proposed benchmarks
				dataframe = preds[train, test] * [simulation_0]    <=> {rows: samples,  columns: workers}

			Note: A separate boolean flag is used for use_parallelization_benchmarks. This will ensure we only apply parallelization when called through worker_weight_strength_relation function """

		def aim1_3_meauring_probs_uncertainties():
			""" Final pred labels & uncertainties for proposed technique
					dataframe = preds[train, test] * [mv]              <=> {rows: samples,  columns: workers}
					dataframe = uncertainties[train, test]  {rows: samples,  columns: workers}

				Final pred labels for proposed benchmarks
					dataframe = preds[train, test] * [simulation_0]    <=> {rows: samples,  columns: workers} """

			def getting_noisy_manual_labels_for_each_worker(true, l_strength: float=0.5, seed_num: int=1):

				# setting the random seed
				# np.random.seed(seed_num)

				# number of samples and workers/workers
				num_samples = true.shape[0]

				# finding a random number for each instance
				true_label_assignment_prob = np.random.random(num_samples)

				# samples that will have an inaccurate true label
				false_samples = true_label_assignment_prob < 1 - l_strength

				# measuring the new labels for each worker/worker
				worker_true = true > 0.5
				worker_true[false_samples] = ~ worker_true[false_samples]

				return worker_true

			def assigning_strengths_randomly_to_each_worker():

				workers_names = [f'worker_{j}' for j in range(n_workers)]

				workers_strength_array = np.random.uniform(low=config.simulation.low_dis, high=config.simulation.high_dis, size=n_workers)

				return pd.DataFrame({'workers_strength': workers_strength_array}, index=workers_names)

			def looping_over_all_workers(workers_strength):

				def calculate_uncertainties(df: pd.DataFrame, techniques: list[UncertaintyTechniques]) -> pd.DataFrame:
					""" Looping over all simulations. this is to measure uncertainties """

					# TODO: use different uncertainty measurement techniques for the SSIAI paper

					epsilon = np.finfo(float).eps
					df = df.astype(int)

					df_uncertainties = pd.DataFrame(columns=techniques, index=df.index)

					for tech in config.simulation.uncertainty_techniques:

						if tech is UncertaintyTechniques.STD:
							df_uncertainties[tech] = df.std( axis=1 )

						elif tech is UncertaintyTechniques.ENTROPY:
							# Normalize each row to sum to 1
							# >> df_normalized = df.div(df.sum(axis=1) + epsilon, axis=0)
							# Calculate entropy
							# >> uncertainties[mode][LB] = - (df_normalized * np.log(df_normalized + epsilon)).sum(axis=1)
							df_uncertainties[tech] = df.apply(lambda x: scipy.stats.entropy(x), axis=1).fillna(0)

						elif tech is UncertaintyTechniques.CV:
							df_uncertainties[tech] = df.std(axis=1) / (df.mean(axis=1) + epsilon)

						elif tech is UncertaintyTechniques.PI:
							df_uncertainties[tech] = df.apply(lambda row: np.percentile(row.astype(int), 75) - np.percentile(row.astype(int), 25), axis=1)

						elif tech is UncertaintyTechniques.CI:
							# Lower Bound: This is the first value in the tuple. It indicates the lower end of the range. If you have a 95% confidence interval, this means that you can be 95% confident that the true population mean is greater than or equal to this value.

							# Upper Bound: This is the second value in the tuple. It represents the upper end of the range. Continuing with the 95% confidence interval example, you can be 95% confident that the true population mean is less than or equal to this value.

							confidene_interval = df.apply(lambda row: scipy.stats.norm.interval(0.95, loc=np.mean(row), scale=scipy.stats.sem(row)) if np.std(row) > 0 else (np.nan, np.nan),axis=1).apply(pd.Series)

														# Calculate the width of the confidence interval (uncertainty score)
							weight = confidene_interval[1] - confidene_interval[0]

							# Optional: Normalize by the mean or another relevant value
							# For example, normalize by the midpoint of the interval
							# midpoint = (confidene_interval[1] + confidene_interval[0]) / 2

							df_uncertainties[tech] = weight.fillna(0) # / midpoint

					return df_uncertainties

				predicted_labels_all_sims = {'train': {}, 'test': {}}
				truth = { 'train': pd.DataFrame(), 'test': pd.DataFrame() }
				columns = pd.MultiIndex.from_product([workers_strength.index, config.simulation.uncertainty_techniques], names=['worker', 'uncertainty_technique'])
				uncertainties = { 'train': pd.DataFrame(columns=columns), 'test': pd.DataFrame(columns=columns) }

				for LB_index, LB in enumerate(workers_strength.index):

					# Initializationn
					for mode in ['train', 'test']:
						predicted_labels_all_sims[mode][LB] = {}
						truth[mode]['truth'] = data[mode].true.copy()

					# Extracting the simulated noisy manual labels based on the worker's strength
					truth['train'][LB] = getting_noisy_manual_labels_for_each_worker( 	seed_num=0,  # LB_index,
																						true=data['train'].true.values,
																						l_strength=workers_strength.T[LB].values )

					truth['test'][LB] = getting_noisy_manual_labels_for_each_worker( seed_num=1,  # LB_index,
																					 true=data['test'].true.values,
																					 l_strength=workers_strength.T[LB].values )

					SIMULATION_TYPE = 'random_state'

					if SIMULATION_TYPE == 'random_state':
						for sim_num in range(config.simulation.num_simulations):
							# training a random forest on the aformentioned labels
							RF = sk_ensemble.RandomForestClassifier(n_estimators=4, max_depth=4, random_state=sim_num)  # n_estimators=4, max_depth=4
							# RF = sklearn.tree.DecisionTreeClassifier(random_state=sim_num)

							RF.fit( X=data['train'][feature_columns], y=truth['train'][LB] )

							# predicting the labels using trained networks for both train and test data
							for mode in ['train', 'test']:
								sim_name = f'simulation_{sim_num}'
								predicted_labels_all_sims[mode][LB][sim_name] = RF.predict(data[mode][feature_columns])

					elif SIMULATION_TYPE == 'multiple_classifier':

						classifiers_list = [
							sklearn.neighbors.KNeighborsClassifier(3),  # type: ignore
							# SVC(kernel="linear", C=0.025),
							sklearn.svm.SVC(gamma=2, C=1),  # type: ignore
							# sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0)),
							sklearn.tree.DecisionTreeClassifier(max_depth=5),  # type: ignore
							sk_ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
							sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000),  # type: ignore
							sk_ensemble.AdaBoostClassifier(),
							sklearn.naive_bayes.GaussianNB(),  # type: ignore
							sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),  # type: ignore
						]

						for sim_num, classifier in enumerate(classifiers_list):

							classifier.fit( X=data['train'][feature_columns], y=truth['train'][LB] )

							# predicting the labels using trained networks for both train and test data
							for mode in ['train', 'test']:
								sim_name = f'simulation_{sim_num}'
								predicted_labels_all_sims[mode][LB][sim_name] = classifier.predict(data[mode][feature_columns])

					# Measuring the prediction and uncertainties values after MV over all simulations
					for mode in ['train', 'test']:
						# converting to dataframe
						predicted_labels_all_sims[mode][LB] = pd.DataFrame(predicted_labels_all_sims[mode][LB], index=data[mode].index)

						# uncertainties for each worker over all simulations
						uncertainties[mode][LB] = calculate_uncertainties(df=predicted_labels_all_sims[mode][LB], techniques=config.simulation.uncertainty_techniques)

						# predicted probability of each class after MV over all simulations
						predicted_labels_all_sims[mode][LB]['mv'] = ( predicted_labels_all_sims[mode][LB].mean(axis=1) > 0.5)

				# reshaping the dataframes
				preds = { 'train': {}, 'test': {} }
				for mode in ['train', 'test']:

					# reversing the order of simulations and workers. NOTE: for the final experiment I should use simulation_0. if I use the mv, then because the augmented truths keeps changing in each simulation, then with enough simulations, I'll end up witht perfect workers.
					for i in range(config.simulation.num_simulations + 1):

						SM = f'simulation_{i}' if i < config.simulation.num_simulations else 'mv'

						preds[mode][SM] = pd.DataFrame()
						for LB in [f'worker_{j}' for j in range(n_workers)]:
							preds[mode][SM][LB] = predicted_labels_all_sims[mode][LB][SM]

				return truth, uncertainties, preds

			def adding_accuracy_for_each_worker(workers_strength, predicted_labels, true_labels):

				workers_strength['accuracy-test-classifier'] = 0.0
				workers_strength['accuracy-test'] = 0.0

				for i in range(n_workers):
					LB = f'worker_{i}'

					# accuracy of classifier in simulation_0
					workers_strength.loc[LB, 'accuracy-test-classifier'] = ( predicted_labels['test']['simulation_0'][LB] == true_labels['test'].truth).mean()

					# accuracy of noisy true labels for each worker
					workers_strength.loc[LB, 'accuracy-test'] 		     = ( true_labels['test'][LB] == true_labels['test'].truth).mean()

				return workers_strength

			# setting a random strength for each worker/worker
			ls = assigning_strengths_randomly_to_each_worker()

			true_labels, uncertainty, predicted_labels = looping_over_all_workers(workers_strength=ls)

			workers_strength = adding_accuracy_for_each_worker(workers_strength=ls, predicted_labels=predicted_labels, true_labels=true_labels)

			return predicted_labels, uncertainty, true_labels, workers_strength

		def aim1_3_measuring_proposed_weights(predicted_labels, predicted_uncertainty):

			# weights       : n_workers thresh_technique num_methods
			# probs_weighted : num_samples thresh_technique n_workers

			# To-Do: This is the part where I should measure the prob_mv_binary for different # of workers instead of all of them
			prob_mv_binary = predicted_labels.mean(axis=1) > 0.5

			T1    = 1 - predicted_uncertainty
			T2    = T1.copy()
			w_hat1: dict[str, Any] = {}
			w_hat2: dict[str, Any] = {}

			for workers_name in predicted_labels.columns:
				# T1[workers_name] = 1 - predicted_uncertainty[workers_name]

				# T2[workers_name] = T1[workers_name].copy()
				T2[workers_name][predicted_labels[workers_name].values != prob_mv_binary.values] = 0

				w_hat1[workers_name] = T1[workers_name].mean(axis=0)
				w_hat2[workers_name] = T2[workers_name].mean(axis=0)

			w_hat = pd.DataFrame([w_hat1, w_hat2], index=list(ProposedTechniqueNames)).T

			# measuring average weight
			weights = w_hat.divide(w_hat.sum(axis=0), axis=1)

			probs_weighted = pd.DataFrame(columns=list(ProposedTechniqueNames), index=predicted_labels.index)
			for method in ProposedTechniqueNames:
				# probs_weighted[method] =( predicted_uncertainty * weights[method] ).sum(axis=1)
				probs_weighted[method] = (predicted_labels * weights[method]).sum( axis=1 )

			return weights, probs_weighted

		def measuring_Tao_weights_based_on_classifier_labels(delta, noisy_true_labels):
			"""
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

		def measuring_Tao_weights_based_on_actual_labels(delta, noisy_true_labels):
			"""
				tau          : 1 thresh_technique 1
				weights_Tao  : num_samples thresh_technique n_workers
				W_hat_Tao    : num_samples thresh_technique n_workers
				z            : num_samples thresh_technique 1
				gamma        : num_samples thresh_technique 1
			"""

			tau = (delta == noisy_true_labels).mean(axis=0)

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

			return W_hat_Tao.divide(z, axis=0)

		def get_AUC_ACC_F1(aggregated_labels: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:

			metrics = pd.DataFrame(index=list(EvaluationMetricNames), columns=aggregated_labels.columns)

			non_null = ~truth.isnull()
			truth_notnull = truth[non_null].to_numpy()

			if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):

				for m in aggregated_labels.columns:
					yhat = (aggregated_labels[m] > 0.5).astype(int)[non_null]
					metrics[m][EvaluationMetricNames.AUC] = sk_metrics.roc_auc_score( truth_notnull, yhat)
					metrics[m][EvaluationMetricNames.ACC] = sk_metrics.accuracy_score(truth_notnull, yhat)
					metrics[m][EvaluationMetricNames.F1 ] = sk_metrics.f1_score( 	  truth_notnull, yhat)

			return metrics

		def measuring_nu_and_confidence_score(n_workers, yhat_proposed_classifier, workers_labels, weights_proposed, weights_Tao) -> Tuple[Dict[StrategyNames, Dict[enum.Enum, pd.DataFrame]], pd.DataFrame]:

			def get_pred_and_weights(method_name: ProposedTechniqueNames):
				if method_name in ProposedTechniqueNames:
					# in the proposed method, unlike Tao and Sheng, we use the output of the trained classifiers as the predicted labels instead of the original worker's labels.
					return yhat_proposed_classifier, weights_proposed[method_name]

				elif method_name is MainBenchmarks.TAO:
					return workers_labels, weights_Tao / n_workers

				elif method_name is MainBenchmarks.SHENG:
					return workers_labels, pd.DataFrame(1 / n_workers, index=weights_Tao.index, columns=weights_Tao.columns)

				raise ValueError(f'Unknown method name: {method_name}')

			def get_F(conf_strategy, delta, weights) -> pd.DataFrame:
				""" Calculating the confidence score for each sample """

				out = pd.DataFrame( index=delta.index )
				# out['truth'] = true_labels['test'].truth

				# Delta is the binarized predicted probabilities
				out['P_pos'] = ( delta * weights).sum(axis=1)
				out['P_neg'] = (~delta * weights).sum(axis=1)

				if conf_strategy == StrategyNames.FREQ:
					out['F'] = out[['P_pos','P_neg']].max(axis=1)
					# F[out['P_pos'] < out['P_neg']] = out['P_neg'][out['P_pos'] < out['P_neg']]

				elif conf_strategy == StrategyNames.BETA:
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
					out['F'] = out.apply(get_F_lambda, axis=1)
					# F = I.copy()
					# F[I < 0.5] = (1 - F)[I < 0.5]

				return out

			F = {StrategyNames.FREQ:{}, StrategyNames.BETA:{}}
			aggregated_labels = pd.DataFrame(columns=list(ProposedTechniqueNames) + list(MainBenchmarks), index=yhat_proposed_classifier.index)

			for methods in [ProposedTechniqueNames, MainBenchmarks]:

				for m in methods:
					pred, weights = get_pred_and_weights(m)
					aggregated_labels[m] = (pred * weights).sum(axis=1)

					for strategy in StrategyNames:
						F[strategy][m] = get_F(strategy, pred, weights)

			return F, aggregated_labels

		# Setting the random seed
		np.random.seed(seed + 1)

		# calculating uncertainty and predicted probability values
		predicted_labels_all, uncertainty_all, true_labels, workers_strength = aim1_3_meauring_probs_uncertainties()

		yhat_proposed_classifier  = predicted_labels_all['test']['mv'          ]
		yhat_benchmark_classifier = predicted_labels_all['test']['simulation_0']

		# Measuring weights for the proposed technique
		weights_proposed, prob_weighted = aim1_3_measuring_proposed_weights( predicted_labels=yhat_proposed_classifier, predicted_uncertainty=uncertainty_all['test'])

		# Benchmark accuracy measurement
		weights_Tao = measuring_Tao_weights_based_on_actual_labels( delta=yhat_benchmark_classifier, noisy_true_labels=true_labels['test'].drop(columns=['truth']))

		workers_labels = true_labels['test'].drop(columns=['truth'])

		F, aggregated_labels = measuring_nu_and_confidence_score(n_workers=n_workers, workers_labels=workers_labels, yhat_proposed_classifier=yhat_proposed_classifier, weights_proposed=weights_proposed, weights_Tao=weights_Tao)

		# Get the results for other benchmarks
		aggregated_labels_benchmarks = BenchmarkTechniques.apply(true_labels=true_labels, use_parallelization_benchmarks=use_parallelization_benchmarks)

		aggregated_labels = pd.concat( [aggregated_labels, aggregated_labels_benchmarks.copy()], axis=1)

		# Measuring the metrics
		metrics = get_AUC_ACC_F1(aggregated_labels=aggregated_labels, truth=true_labels['test'].truth)

		# merge workers_strength and weights
		weights_Tao_mean  = weights_Tao.mean().to_frame().rename(columns={0: MainBenchmarks.TAO})
		workers_strength = pd.concat( [workers_strength, weights_proposed * n_workers, weights_Tao_mean], axis=1)

		return ResultsType(true_labels=true_labels, workers_strength=workers_strength, F=F, aggregated_labels=aggregated_labels, weights_proposed=weights_proposed, weights_Tao=weights_Tao, n_workers=n_workers, metrics=metrics)


	@staticmethod
	def wrapper(args: dict[str, Any], config, data, feature_columns) -> Tuple[dict[str, Any], ResultsType]:
		return args, AIM1_3.core_measurements(data=data, n_workers=args['nl'], config=config, feature_columns=feature_columns, seed=args['seed'], use_parallelization_benchmarks=False)


	@staticmethod
	def objective_function(config, data, feature_columns):
		return functools.partial(AIM1_3.wrapper, config=config, data=data, feature_columns=feature_columns)


	@classmethod
	def calculate_one_dataset(cls, config: 'Settings', dataset_name: DatasetNames=DatasetNames.IONOSPHERE) -> ResultsComparisonsType:

		def get_outputs(config, data=None, feature_columns=None) -> Dict[str, List[ResultsType]]:

			def calculate():

				input_list = [{'nl':nl, 'seed':seed} for nl in config.simulation.workers_list for seed in range(config.simulation.num_seeds)]

				function = cls.objective_function(config=config, data=data, feature_columns=feature_columns)

				outputs = {f'NL{nl}':np.full((config.simulation.num_seeds), np.nan).tolist()  for nl in config.simulation.workers_list}

				if config.simulation.use_parallelization:

					with multiprocessing.Pool(processes=min(config.simulation.max_parallel_workers, len(input_list))) as pool:
						core_results = pool.map(function, input_list)

					# with ProcessPoolExecutor(max_workers=config.simulation.max_parallel_workers) as executor:
					# 	future_to_seed = {executor.submit(function, args): args for args in input_list}
					# 	core_results = []
					# 	for future in tqdm(as_completed(future_to_seed), total=len(future_to_seed), desc='Processing n_workers & seeds'):
					# 		core_results.append(future.result())

				else:
					core_results = [function(args) for args in input_list]

				for args, values in core_results:
					outputs[f'NL{args["nl"]}'][args['seed']] = values

				return outputs

			path = config.output.path / 'outputs' / f'{config.dataset.dataset_name}.pkl'

			if config.output.mode is OutputModes.CALCULATE:
				outputs = calculate()

				if config.output.save:
					LoadSaveFile(path).dump(outputs)

			else:
				outputs = LoadSaveFile(path).load()

			return outputs

		def get_outputs_old(config, data=None, feature_columns=None) -> Dict[str, List[ResultsType]]:

			def get_core_results_not_parallel() -> List[ResultsType]:
				# This is not written as one line for loop on purpose (for debugging purposes)
				core_results = []
				for seed in range(config.simulation.num_seeds):
					core_results.append(aim1_3.get_core_measurements(seed=seed))

				return core_results


			path = config.output.path / 'outputs' / f'{config.dataset.dataset_name}.pkl'

			if config.output.mode is OutputModes.LOAD_LOCAL:
				return LoadSaveFile(path).load() # type: ignore

			elif config.output.mode is OutputModes.CALCULATE:

				outputs = {}
				for nl in tqdm(config.simulation.workers_list, desc='looping through different # workers'):
					aim1_3 = AIM1_3(config=config, n_workers=nl, data=data, feature_columns=feature_columns)

					# Get the core measurements for all seeds
					if config.simulation.use_parallelization:
						with multiprocessing.Pool(processes=config.simulation.num_seeds) as pool:
							outputs[f'NL{nl}'] = pool.map( aim1_3.get_core_measurements, list(range(config.simulation.num_seeds)))
					else:
						outputs[f'NL{nl}'] = get_core_results_not_parallel()

				# Saving the outputs locally
				if config.output.save:
					LoadSaveFile(path).dump(outputs)

				return outputs

			return None  # type: ignore

		def get_confidence_scores(outputs, config):

			path_main = config.output.path / f'confidence_score/{config.dataset.dataset_name}'

			DICT_KEYS = ['F_all', 'F_pos_all', 'F_mean_over_seeds', 'F_pos_mean_over_seeds']

			def get_Fs_per_nl_per_strategy(strategy, n_workers):

				def get(stuff):
					seeds_list   = list(range(config.simulation.num_seeds))
					# methods_list = ProposedTechniqueNames.members() + MainBenchmarks.members()
					methods_list = list(ProposedTechniqueNames) + list(MainBenchmarks)
					columns      = pd.MultiIndex.from_product([methods_list, seeds_list], names=['method', 'seed_ix'])
					df = pd.DataFrame(columns=columns)

					for (m, sx) in columns:
						df[(m, sx)] = outputs[n_workers][sx].F[strategy][m][stuff].squeeze()

					return df

				inF     = get( 'F' )
				inF_pos = get( 'P_pos' ) if strategy == StrategyNames.FREQ else get( 'I' )

				# inF_mean_over_seeds     = inF.T.groupby(level=0).mean().T
				# inF_pos_mean_over_seeds = inF_pos.T.groupby(level=0).mean().T

				techniques_list         = inF.columns.get_level_values(0)
				inF_mean_over_seeds     = pd.DataFrame({m: inF[m].mean(axis=1)     for m in techniques_list})
				inF_pos_mean_over_seeds = pd.DataFrame({m: inF_pos[m].mean(axis=1) for m in techniques_list})

				return inF, inF_pos, inF_mean_over_seeds, inF_pos_mean_over_seeds


			if config.output.mode is OutputModes.CALCULATE:

				F_dict = { key : {StrategyNames.FREQ:{}, StrategyNames.BETA:{}} for key in DICT_KEYS }

				for st in StrategyNames:
					for nl in [f'NL{x}' for x in config.simulation.workers_list]:

						F, F_pos, F_mean_over_seeds, F_pos_mean_over_seeds = get_Fs_per_nl_per_strategy( strategy=st, n_workers=nl )

						F_dict['F_all' 			 	  ][st][nl] = F.copy()
						F_dict['F_pos_all' 			  ][st][nl] = F_pos.copy()
						F_dict['F_mean_over_seeds'	  ][st][nl] = F_mean_over_seeds.copy()
						F_dict['F_pos_mean_over_seeds'][st][nl] = F_pos_mean_over_seeds.copy()


				if config.output.save:
					for name in DICT_KEYS:
						LoadSaveFile(path_main / f'{name}.pkl').dump(F_dict[name])

				return F_dict

			elif config.output.mode == OutputModes.LOAD_LOCAL:
				return { key: LoadSaveFile(path_main / f'{key}.pkl').load()  for key in DICT_KEYS }


			raise ValueError(f'Unknown config.output.mode: {config.output.mode}')

		def worker_weight_strength_relation(config, data, feature_columns, seed=0, n_workers=10) -> pd.DataFrame:

			metric_name = 'weight_strength_relation'
			path_main = config.output.path / metric_name / config.dataset.dataset_name.value

			if config.output.mode is OutputModes.CALCULATE:
				df = AIM1_3.core_measurements(n_workers=n_workers, config=config, data=data, feature_columns=feature_columns, seed=seed, use_parallelization_benchmarks=config.simulation.use_parallelization).workers_strength.set_index('workers_strength').sort_index()

				if config.output.save:
					LoadSaveFile(path_main / f'{metric_name}.xlsx').dump(df, index=True)

				return df

			else:
				return LoadSaveFile(path_main / f'{metric_name}.xlsx').load(header=0)


		np.random.seed(0)
		config.dataset.dataset_name = DatasetNames(dataset_name)

		# loading the dataset
		data, feature_columns = load_data.aim1_3_read_download_UCI_database(config=config) # type: ignore

		# getting the outputs
		outputs = get_outputs(config=config, data=data, feature_columns=feature_columns)

		# measuring the confidence scores
		findings_confidence_score = get_confidence_scores(outputs=outputs, config=config)

		# measuring the worker strength weight relationship for proposed and Tao
		weight_strength_relation = worker_weight_strength_relation(config=config, data=data, feature_columns=feature_columns, seed=0, n_workers=10)

		return ResultsComparisonsType(weight_strength_relation=weight_strength_relation, findings_confidence_score=findings_confidence_score, outputs=outputs, config=config)


	@classmethod
	def calculate_all_datasets(cls, config: 'Settings') -> Dict[DatasetNames, ResultsComparisonsType]:
		return {dt: cls.calculate_one_dataset(dataset_name=dt, config=config) for dt in config.dataset.datasetNames}


class Aim1_3_Data_Analysis_Results:

	RENAME_MAPS={ProposedTechniqueNames.PROPOSED_PENALIZED:'Crowd-Certain' , ProposedTechniqueNames.PROPOSED:'Crowd-Certain Without Penalization'}

	def __init__(self, config):

		self.outputs  = None
		self.accuracy = dict(freq=pd.DataFrame(), beta=pd.DataFrame())
		self.config   = config

	def update(self) -> 'Aim1_3_Data_Analysis_Results':
		self.results_all_datasets  = AIM1_3.calculate_all_datasets(config=self.config)
		return self


	def get_result(self, metric_name='F_all', dataset_name: DatasetNames=DatasetNames.MUSHROOM, strategy=StrategyNames.FREQ , nl='NL3', seed_ix=0, method_name=ProposedTechniqueNames.PROPOSED, data_mode='test'):

		metrics_list = [EvaluationMetricNames.AUC, EvaluationMetricNames.ACC, EvaluationMetricNames.F1]

		def drop_proposed_rename_crowd_certain(df, orient='columns'):

			if orient == 'columns':
				return df.drop(columns=[ProposedTechniqueNames.PROPOSED])

			else:
				return df.drop(index=[ProposedTechniqueNames.PROPOSED])

		def get_metrics_mean_over_seeds(dataset_name1: DatasetNames, n_workers) -> pd.DataFrame:
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
			value_vars = list(self.RENAME_MAPS.values()) + [MainBenchmarks.TAO.value]
			rename_columns = self.RENAME_MAPS
			rename_columns.update({MainBenchmarks.TAO:MainBenchmarks.TAO.value})

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

		from sklearn.metrics import brier_score_loss

		def ece_score(y_true, conf_scores_per_method, n_bins=10):
			"""Compute ECE"""
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
					accuracy_in_bin = y_true[in_bin].mean()
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

		metric_name = 'weight_strength_relation'
		df: pd.DataFrame = self.get_result(metric_name=metric_name)  # type: ignore

		sns.set( palette='colorblind', style='darkgrid', context='paper', font_scale=font_scale)

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
		def get_axes(i1, i2):
			nonlocal axes
			if len(workers_list) == 1 or len(metrics_list) == 1:
				return axes[max(i1, i2)]
			else:
				return axes[i1, i2]


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
				sns.boxplot(data=data, orient='h', ax=get_axes(i1, i2))

		i1 = 2 if len(workers_list) > 2 else len(workers_list)
		for i2, metric in enumerate(metrics_list):
			get_axes(i1, i2).set_xlabel(metric, fontsize=fontsize, fontweight='bold', labelpad=20)

		for i1, nl in enumerate(workers_list):
			get_axes(i1,0).set_ylabel(nl, fontsize=fontsize, fontweight='bold', labelpad=20)

		fig.suptitle('Metrics Over All Datasets', fontsize=int(1.5*fontsize), fontweight='bold')
		plt.tight_layout()

		# Saving the Figure & Sheet
		self.save_outputs( filename=f'figure_{metric_name}', relative_path=relative_path, dataframe=df )


	def figure_F_heatmap(self, metric_name='F_eval_one_dataset_all_workers', dataset_name:DatasetNames=DatasetNames.IONOSPHERE, nl='NL3', fontsize=20, font_scale=1.8, figsize=(13, 11), relative_path='final_figures'):

		def get_filename_subtitle():

			if metric_name == 'F_eval_one_dataset_all_workers':
				filename  = f'heatmap_F_evals_{dataset_name}_all_workers'
				suptitle = f'Confidence Score Evaluation for {dataset_name}'

			elif metric_name == 'F_eval_one_worker_all_datasets':
				filename  = f'heatmap_F_evals_all_datasets_{nl}'
				suptitle = f'Confidence Score Evaluation for {nl.split("NL")[1]} Workers'

			else:
				raise ValueError('metric_name does not exist')

			return filename, suptitle

		def create_heatmap(data, ax, cmap, cbar, title='', ylabel='', xlabel=''):
			sns.heatmap(data, ax=ax, annot=True, fmt='.2f', cmap=cmap, cbar=cbar, robust=True)
			ax.set_title(title, fontsize=fontsize, fontweight='bold')
			ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', labelpad=20)
			ax.set_xlabel(xlabel)

		def plot(df, filename, subtitle):

			sns.set(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')
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
		plot(df, filename, subtitle)

		self.save_outputs( filename=f'figure_{filename}', relative_path=relative_path, dataframe=df.T )


class AIM1_3_Plot:
	""" Plotting the results"""

	def __init__(self, plot_data: pd.DataFrame):

		self.weight_strength_relation_interpolated = None
		assert type(plot_data) == pd.DataFrame, 'plot_data must be a pandas DataFrame'

		self.plot_data = plot_data

	@staticmethod
	def data_interpolation(x, y, smooth=False, interpolation_pt_count=1000):
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

