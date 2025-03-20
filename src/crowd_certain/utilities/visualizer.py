# Third-party imports
from typing import Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

# Local imports
from crowd_certain.utilities.io.dataset_loader import LoadSaveFile
from crowd_certain.utilities.io.hdf5_storage import HDF5Storage
from crowd_certain.utilities.parameters import params
from crowd_certain.utilities.utils import AIM1_3

USE_HD5F_STORAGE = True


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


	def get_result(self, metric_name='F_all', dataset_name: params.DatasetNames=params.DatasetNames.MUSHROOM, strategy=params.StrategyNames.FREQ , nl='NL3', seed_ix=0, method_name=params.ProposedTechniqueNames.PROPOSED, data_mode='test'):
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

		dataset_name : params.DatasetNames, default=params.DatasetNames.MUSHROOM
			The dataset to retrieve results for.

		strategy : params.StrategyNames, default=params.StrategyNames.FREQ
			The strategy used for aggregation or confidence calculation.

		nl : str, default='NL3'
			Worker count identifier (e.g., 'NL3' for 3 workers).

		seed_ix : int, default=0
			The simulation seed index to use.

		method_name : params.ProposedTechniqueNames, default=params.ProposedTechniqueNames.PROPOSED
			The method to retrieve results for.

		data_mode : str, default='test'
			Whether to use 'train' or 'test' data.

		Returns
		-------
		pandas.DataFrame
			The requested metrics or data according to the specified parameters.
		"""

		metrics_list = [params.EvaluationMetricNames.AUC, params.EvaluationMetricNames.ACC, params.EvaluationMetricNames.F1]

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
					return dataframe.drop( columns=[params.ProposedTechniqueNames.PROPOSED] )

				else:
					return dataframe.drop( index=[params.ProposedTechniqueNames.PROPOSED] )

		def get_metrics_mean_over_seeds(dataset_name1: params.DatasetNames, n_workers) -> pd.DataFrame:
			"""
			Calculate the mean of metrics across different random seeds for a specific dataset and number of workers.

			This function retrieves metrics for each seed from the simulation results, organizes them in a DataFrame,
			and then computes the mean value for each metric across all seeds.

			Parameters:
			-----------
			dataset_name1 : params.DatasetNames
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

			if strategy == params.StrategyNames.FREQ:
				df = 1 - df

			df['truth'] = self.results_all_datasets[dataset_name].outputs[nl][seed_ix].true_labels[data_mode].truth
			return df

		elif metric_name in ['F_eval_one_dataset_all_workers', 'F_eval_one_worker_all_datasets']:
			return self.get_evaluation_metrics_for_confidence_scores(metric_name=metric_name, dataset_name=dataset_name, nl=nl)

		elif metric_name == 'weight_strength_relation': # 'df_weight_stuff'
			techniques: list[Any] = list(params.ProposedTechniqueNames) + [params.MainBenchmarks.TAO]
			value_vars     = [n.value    for n in techniques]
			rename_columns = {n: n.value for n in techniques}

			wwr = pd.DataFrame()
			for dt in self.config.dataset.datasetNames:
				if USE_HD5F_STORAGE:
					# Create a path for the HDF5 file and use HDF5Storage to load the data
					path_main = self.config.output.path / 'weight_strength_relation' / str(dt)
					h5s = HDF5Storage(path_main / 'weight_strength_relation.h5')
					df_w = h5s.load_dataframe('/weight_strength_relation')

					if df_w is not None:
						df_w = df_w.reset_index()
						df_w['dataset_name'] = dt
						wwr = pd.concat([wwr, df_w])
				else:
					df = (self.results_all_datasets[dt].weight_strength_relation
							.rename(columns=rename_columns)
							.reset_index()
							.melt(id_vars=['workers_strength'], value_vars=value_vars, var_name='Method', value_name='Weight'))

					wwr = pd.concat([wwr, df.assign(dataset_name=dt.value)], axis=0)

			if USE_HD5F_STORAGE:
				# Format the data for visualization
				wwr = pd.melt(
					wwr,
					id_vars=['workers_strength', 'dataset_name'],
					value_vars=value_vars,
					var_name='Method',
					value_name='Weight'
				)

				# Rename the techniques for better readability
				wwr['Method'] = wwr['Method'].replace(rename_columns)

			return wwr

		elif metric_name in ['F', 'aggregated_labels','true_labels']:

			if metric_name == 'F':
				assert method_name in list(params.ProposedTechniqueNames) + list(params.MainBenchmarks)
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
				columns = pd.MultiIndex.from_product(
					[metrics_list, self.config.dataset.datasetNames, workers_list],
					names=['metric', 'dataset', 'workers']
				)

				# Create DataFrame using dictionary comprehension
				data = {
					(metric, dt, nl): drop_proposed_rename_crowd_certain(
						get_metrics_mean_over_seeds(dt, nl),
						orient='index'
					)[metric]
					for dt in self.config.dataset.datasetNames
					for nl in workers_list
					for metric in metrics_list
				}

				return pd.DataFrame(data, columns=columns)

		raise ValueError(f"Invalid metric name: {metric_name}")

	def get_evaluation_metrics_for_confidence_scores(self, metric_name='F_eval_one_dataset_all_workers', dataset_name: params.DatasetNames=params.DatasetNames.IONOSPHERE, nl='NL3'):
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

		dataset_name : params.DatasetNames, default=params.DatasetNames.IONOSPHERE
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
			def get_params(i):
				return dict(nl=i, dataset_name=dataset_name)

		elif metric_name == 'F_eval_one_worker_all_datasets':
			target_list = self.config.dataset.datasetNames
			target_name = 'dataset_name'
			def get_params(i):
				return dict(nl=nl, dataset_name=i)

		else:
			raise ValueError('metric_name should be either workers or datasets')


		index   = pd.MultiIndex.from_product([list(params.StrategyNames), [params.ProposedTechniqueNames.PROPOSED_PENALIZED] + list(params.MainBenchmarks)], names=['strategy', 'technique'])

		columns = pd.MultiIndex.from_product([list(params.ConfidenceScoreNames), target_list], names=['metric', target_name])

		df_cs   = pd.DataFrame(columns=columns, index=index)

		techniques = [params.ProposedTechniqueNames.PROPOSED_PENALIZED, params.MainBenchmarks.TAO, params.MainBenchmarks.SHENG]

		for st in params.StrategyNames:
			for ix in target_list:
				conf_scores = self.get_result(metric_name='F_pos_mean_over_seeds', strategy=st, **get_params(ix))
				conf_scores = conf_scores[techniques + ['truth']]

				for m in techniques:
					df_cs[(params.ConfidenceScoreNames.ECE  ,ix)][(st, m)] = ece_score(conf_scores.truth, conf_scores[m])
					df_cs[(params.ConfidenceScoreNames.BRIER,ix)][(st, m)] = brier_score_loss(conf_scores.truth, conf_scores[m])

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


	def figure_metrics_mean_over_seeds_per_dataset_per_worker(self, metric: params.EvaluationMetricNames=params.EvaluationMetricNames.ACC, nl=3, figsize=(10, 10), font_scale=1.8, fontsize=20, relative_path='final_figures'):
		"""
		Creates and saves a figure displaying the mean of a specified evaluation metric across seeds for each dataset per worker.

		This function generates a heatmap that visualizes the mean performance metrics across different datasets and workers.
		The results are calculated by averaging across all random seeds for each dataset and worker combination.

		Parameters
		----------
		metric : params.EvaluationMetricNames, default=params.EvaluationMetricNames.ACC
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


	def figure_metrics_all_datasets_workers(self, workers_list: Optional[list[str]]=None, figsize=(15, 15), font_scale=1.8, fontsize=20, relative_path='final_figures'):
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

		if workers_list is None:
			workers_list = [f'NL{i}' for i in self.config.simulation.workers_list]

		def get_axes(ix1, ix2):
			nonlocal axes
			if len(workers_list) == 1 or len(metrics_list) == 1:
				return axes[max( ix1, ix2 )]
			else:
				return axes[ix1, ix2]

		sns.set_theme(font_scale=font_scale, palette='colorblind', style='darkgrid', context='paper')

		metric_name  = 'metrics_all_datasets_workers'
		emn = params.EvaluationMetricNames
		metrics_list = [emn.ACC, emn.AUC, emn.F1]

		df: pd.DataFrame = self.get_result(metric_name=metric_name)
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


	def figure_F_heatmap(self, metric_name='F_eval_one_dataset_all_workers', dataset_name:params.DatasetNames=params.DatasetNames.IONOSPHERE, nl='NL3', fontsize=20, font_scale=1.8, figsize=(13, 11), relative_path='final_figures'):
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

		dataset_name : params.DatasetNames, optional
			Dataset to evaluate when using 'F_eval_one_dataset_all_workers'.
			Default is params.DatasetNames.IONOSPHERE.

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
			create_heatmap(data=df[params.ConfidenceScoreNames.ECE].T[params.StrategyNames.FREQ],  ax=axes[0, 0], cmap='Reds', cbar=False, title=params.StrategyNames.FREQ.name, ylabel=params.ConfidenceScoreNames.ECE.name)
			create_heatmap(data=df[params.ConfidenceScoreNames.ECE].T[params.StrategyNames.BETA],  ax=axes[0, 1], cmap='Reds', cbar=True,  title=params.StrategyNames.BETA.name)
			create_heatmap(data=df[params.ConfidenceScoreNames.BRIER].T[params.StrategyNames.FREQ], ax=axes[1, 0], cmap='Blues', cbar=False, ylabel=params.ConfidenceScoreNames.BRIER.name)
			create_heatmap(data=df[params.ConfidenceScoreNames.BRIER].T[params.StrategyNames.BETA], ax=axes[1, 1], cmap='Blues', cbar=True)

			fig.suptitle(subtitle, fontsize=int(1.5*fontsize), fontweight='bold')
			plt.tight_layout()


		filename, subtitle = get_filename_subtitle()

		df = self.get_result(metric_name=metric_name, dataset_name=dataset_name, nl=nl).round(3)

		plot()

		self.save_outputs( filename=f'figure_{filename}', relative_path=relative_path, dataframe=df.T )


# class AIM1_3_Plot:
# 	"""
# 	A class for plotting data in a DataFrame format, with options for smoothing and visualization.

# 	This class provides various methods for manipulating and visualizing data, including
# 	interpolation and smoothing of data points, customizing plot appearance, and displaying
# 	markers for specific techniques.

# 	Attributes:
# 		plot_data (pd.DataFrame): The data to be plotted.
# 		weight_strength_relation_interpolated (pd.DataFrame or None): Interpolated data after plotting.

# 	Methods:
# 		__init__(plot_data): Initializes the plotting class with the provided data.
# 		data_interpolation(x, y, smooth, interpolation_pt_count): Static method that performs
# 			data interpolation and smoothing.
# 		plot(xlabel, ylabel, xticks, title, legend, smooth, interpolation_pt_count, show_markers):
# 			Creates and displays the plot with the specified parameters.
# 		_show(x, xnew, y_smooth, xlabel, ylabel, xticks, title): Static method for configuring and
# 			displaying plot elements.
# 		_legend(legend, columns): Static method for adding legend to the plot.
# 		_fixing_x_axis(index): Static method for formatting the x-axis values.
# 		_show_markers(show_markers, columns, x, y): Static method for displaying markers on
# 			specific points in the plot.
# 	"""

# 	def __init__(self, plot_data: pd.DataFrame):

# 		self.weight_strength_relation_interpolated = None
# 		assert isinstance(plot_data, pd.DataFrame), 'plot_data must be a pandas DataFrame'

# 		self.plot_data = plot_data

# 	@staticmethod
# 	def data_interpolation(x, y, smooth=False, interpolation_pt_count=1000):
# 		"""
# 		Interpolate data points and optionally apply smoothing.

# 		Parameters
# 		----------
# 		x : numpy.ndarray
# 			The x-coordinates of the data points.
# 		y : numpy.ndarray
# 			The y-coordinates of the data points. Should have shape (n, m) where
# 			n is the number of data points and m is the number of dimensions.
# 		smooth : bool, optional
# 			Whether to apply smoothing to the data. Default is False.
# 		interpolation_pt_count : int, optional
# 			Number of points to use for interpolation if smoothing is applied. Default is 1000.

# 		Returns
# 		-------
# 		xnew : numpy.ndarray
# 			The x-coordinates of the interpolated data.
# 		y_smooth : numpy.ndarray
# 			The y-coordinates of the interpolated and smoothed data.

# 		Notes
# 		-----
# 		When smoothing is enabled, the function attempts to use a spline or convolution method.
# 		If an exception occurs during smoothing, the original data is returned unmodified.
# 		The 'kernel_regression' method is commented out in the current implementation.
# 		"""
# 		xnew, y_smooth = x, y

# 		if smooth:
# 			SMOOTH_METHOD = 'kernel_regression'

# 			try:

# 				if SMOOTH_METHOD == 'spline':

# 					xnew = np.linspace(x.min(), x.max(), interpolation_pt_count)
# 					spl = make_interp_spline(x, y, k=2)
# 					y_smooth = spl(xnew)

# 				elif SMOOTH_METHOD == 'conv':

# 					filter_size = 5
# 					filter_array = np.ones(filter_size) / filter_size
# 					xnew = x.copy()
# 					y_smooth = np.zeros(list(xnew.shape) + [2])
# 					for j in range(y.shape[1]):
# 						y_smooth[:, j] = np.convolve(y[:, j], filter_array, mode='same')

# 				# elif SMOOTH_METHOD == 'kernel_regression':

# 				#     xnew = np.linspace(thresh_technique.min(), thresh_technique.max(), interpolation_pt_count)
# 				#     y_smooth = np.zeros(list(xnew.shape) + [y.shape[1]])
# 				#     for j in range(y.shape[1]):
# 				#         kr = statsmodels.nonparametric.kernel_regression.KernelReg(y[:, j], thresh_technique, 'c')
# 				#         y_smooth[:, j], _ = kr.fit(xnew)

# 			except Exception as e:
# 				print(e)
# 				xnew, y_smooth = x, y

# 		return xnew, y_smooth

# 	def plot(self, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, interpolation_pt_count=1000, show_markers=params.ProposedTechniqueNames.PROPOSED):
# 		"""
# 		Plot the weight-strength relationship data.

# 		This method takes the plot data stored in the instance and creates a plot of the weight-strength relationship.
# 		It can interpolate and smooth the data, add markers for specific techniques, and set various plot attributes.

# 		Parameters:
# 		-----------
# 		xlabel : str, optional
# 			Label for the x-axis, by default ''
# 		ylabel : str, optional
# 			Label for the y-axis, by default ''
# 		xticks : bool, optional
# 			Whether to display x-axis ticks, by default True
# 		title : str, optional
# 			Title for the plot, by default ''
# 		legend : list or None, optional
# 			Custom legend labels. If None, uses column names from plot_data, by default None
# 		smooth : bool, optional
# 			Whether to smooth the plotted data, by default True
# 		interpolation_pt_count : int, optional
# 			Number of points to use for interpolation when smoothing, by default 1000
# 		show_markers : str or list, optional
# 			Which technique(s) to show markers for. Uses params.ProposedTechniqueNames.PROPOSED by default

# 			Returns:
# 		-------
# 		None
# 			The plot is displayed but not returned.

# 		Notes:
# 		------
# 		The interpolated data is stored in the weight_strength_relation_interpolated attribute as a pandas DataFrame.
# 		"""

# 		columns = self.plot_data.columns.to_list()
# 		y       = self.plot_data.values.astype(float)
# 		x       = self._fixing_x_axis(index=self.plot_data.index)

# 		xnew, y_smooth = AIM1_3_Plot.data_interpolation(x=x, y=y, smooth=smooth, interpolation_pt_count=interpolation_pt_count)

# 		self.weight_strength_relation_interpolated = pd.DataFrame(y_smooth, columns=columns, index=xnew)
# 		self.weight_strength_relation_interpolated.index.name = 'workers_strength'

# 		plt.plot(xnew, y_smooth)
# 		self._show_markers(show_markers=show_markers, columns=columns, x=x, y=y)

# 		self._show(x=x, xnew=xnew, y_smooth=y_smooth, xlabel=xlabel, ylabel=ylabel, xticks=xticks, title=title, )
# 		self._legend(legend=legend, columns=columns)

# 	@staticmethod
# 	def _show(x, xnew, y_smooth, xlabel, ylabel, xticks, title):

# 		plt.xlabel(xlabel)
# 		plt.ylabel(ylabel)
# 		plt.title(title)
# 		plt.grid()

# 		if xticks:
# 			plt.xticks(xnew)

# 		plt.show()

# 		if xticks:
# 			plt.xticks(x)

# 		plt.ylim(y_smooth.min() - 0.1, max(1, y_smooth.max()) + 0.1)
# 		plt.xlabel(xlabel)
# 		plt.ylabel(ylabel)
# 		plt.title(title)
# 		plt.grid(True)

# 	@staticmethod
# 	def _legend(legend, columns):

# 		if legend is None:
# 			pass
# 		elif legend == 'empty':
# 			plt.legend()
# 		else:
# 			plt.legend(columns, **legend)

# 	@staticmethod
# 	def _fixing_x_axis(index):
# 		return index.map(lambda x: int(x.replace('NL', ''))) if isinstance(index[0], str) else index.to_numpy()

# 	@staticmethod
# 	def _show_markers(show_markers, columns, x, y):
# 		if show_markers in (params.ProposedTechniqueNames.PROPOSED, True):
# 			cl = [i for i, x in enumerate(columns) if (params.ProposedTechniqueNames.PROPOSED in x) or ('method' in x)]
# 			plt.plot(x, y[:, cl], 'o')

# 		elif show_markers == 'all':
# 			plt.plot(x, y, 'o')

