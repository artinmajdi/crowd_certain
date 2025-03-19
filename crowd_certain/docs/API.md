# API Reference

[← Back to Main README](../README.md) | [← Back to Documentation Index](index.md) | [← Previous: Usage Guide](USAGE.md)

---

This document provides detailed information about the classes and methods available in the Crowd-Certain library.

## Table of Contents

- [Core Classes](#core-classes)
  - [AIM1_3](#aim1_3)
  - [BenchmarkTechniques](#benchmarktechniques)
  - [LoadSaveFile](#loadsavefile)
  - [Aim1_3_Data_Analysis_Results](#aim1_3_data_analysis_results)
  - [AIM1_3_Plot](#aim1_3_plot)
- [Data Types](#data-types)
  - [ResultType](#resulttype)
  - [WeightType](#weighttype)
  - [Result2Type](#result2type)
  - [ResultComparisonsType](#resultcomparisonstype)
- [Enumerations](#enumerations)

## Core Classes

### AIM1_3

Main class for implementing uncertainty-based worker weight calculation and confidence scoring for crowdsourced label aggregation.

#### Constructor

```python
AIM1_3(data, feature_columns, config, n_workers=3, seed=0)
```

**Parameters:**
- `data` (dict): Dictionary containing training and test data
- `feature_columns` (list): List of feature column names
- `config` (Settings): Configuration object containing simulation parameters
- `n_workers` (int, optional): Number of workers in the simulation. Default is 3.
- `seed` (int, optional): Random seed for reproducibility. Default is 0.

#### Methods

##### calculate_uncertainties

```python
calculate_uncertainties(df: pd.DataFrame) -> pd.DataFrame
```

Calculate various uncertainty metrics for a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing values for which uncertainty needs to be calculated.

**Returns:**
- pd.DataFrame: DataFrame containing calculated uncertainty values for each row in the input DataFrame.

##### calculate_consistency

```python
calculate_consistency(uncertainty: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame
```

Calculates consistency metrics from uncertainty values.

**Parameters:**
- `uncertainty` (Union[pd.DataFrame, pd.Series, np.ndarray]): The uncertainty values to convert to consistency.

**Returns:**
- pd.DataFrame: A DataFrame containing the calculated consistency values.

##### aim1_3_meauring_probs_uncertainties

```python
aim1_3_meauring_probs_uncertainties()
```

Simulate workers with varying skill levels and measure their predictions and uncertainties.

**Returns:**
- tuple: A tuple containing:
  - preds (dict): Predictions organized by mode ('train'/'test'), simulation, and worker
  - uncertainties (dict): Uncertainty measures for each worker and uncertainty technique
  - truth (dict): The true labels and worker-annotated labels
  - workers_strength (pd.DataFrame): Worker strength information

##### aim1_3_measuring_proposed_weights

```python
aim1_3_measuring_proposed_weights(preds, uncertainties) -> pd.DataFrame
```

Calculates proposed weights for worker predictions based on consistency and uncertainty metrics.

**Parameters:**
- `preds` (pd.DataFrame): Worker predictions with data indices as rows and workers as columns
- `uncertainties` (pd.DataFrame): Uncertainty metrics for each worker and technique

**Returns:**
- pd.DataFrame: Calculated weights for each worker across different consistency techniques, uncertainty techniques, and proposed weighting methods.

##### get_weights

```python
get_weights(workers_labels, preds, uncertainties, noisy_true_labels, n_workers) -> WeightType
```

Calculate weights for different methods (proposed, TAO, and SHENG).

**Parameters:**
- `workers_labels` (pd.DataFrame): Matrix of labels assigned by workers to items.
- `preds` (pd.Series or np.ndarray): Predictions (estimated true labels).
- `uncertainties` (pd.Series or np.ndarray): Uncertainty values associated with predictions.
- `noisy_true_labels` (pd.Series or np.ndarray): Ground truth labels (possibly with noise).
- `n_workers` (int): Number of workers who provided labels.

**Returns:**
- WeightType: Named tuple containing weights for three methods.

##### measuring_nu_and_confidence_score

```python
measuring_nu_and_confidence_score(weights: WeightType, preds_all, true_labels, use_parallelization_benchmarks: bool=False) -> Tuple[ResultType, ResultType]
```

Calculates the confidence scores (nu) and evaluation metrics for proposed methods and benchmarks.

**Parameters:**
- `weights` (WeightType): Object containing weights for proposed methods and benchmarks.
- `preds_all`: Dictionary containing workers' predictions for different datasets and methods.
- `true_labels`: Ground truth labels for evaluation.
- `use_parallelization_benchmarks` (bool, optional): Whether to use parallelization for benchmark calculation. Default is False.

**Returns:**
- Tuple[ResultType, ResultType]: A tuple containing results for proposed methods and benchmark methods.

##### worker_weight_strength_relation

```python
worker_weight_strength_relation(seed=0, n_workers=10) -> pd.DataFrame
```

Analyzes the relationship between worker strength and weights.

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility. Default is 0.
- `n_workers` (int, optional): Number of workers to simulate. Default is 10.

**Returns:**
- pd.DataFrame: DataFrame containing the relationship between worker strength and weights.

##### get_outputs

```python
get_outputs() -> Dict[str, List[ResultType]]
```

Retrieves the outputs for the calculation or loads them from file based on the current configuration.

**Returns:**
- Dict[str, List[ResultType]]: A dictionary mapping output keys to lists containing the results for each seed.

#### Static Methods

##### get_AUC_ACC_F1

```python
@staticmethod
get_AUC_ACC_F1(aggregated_labels: pd.Series, truth: pd.Series) -> pd.Series
```

Calculate AUC, accuracy, and F1 score metrics between aggregated labels and ground truth.

**Parameters:**
- `aggregated_labels` (pd.Series): The aggregated (predicted) probability labels, typically between 0 and 1.
- `truth` (pd.Series): The ground truth labels with the same index as aggregated_labels.

**Returns:**
- pd.Series: A pandas Series containing AUC, Accuracy, and F1 score.

##### calculate_confidence_scores

```python
@staticmethod
calculate_confidence_scores(delta, w: Union[pd.DataFrame, pd.Series], n_workers) -> pd.DataFrame
```

Calculate confidence scores for each item using multiple strategies.

**Parameters:**
- `delta` (pd.DataFrame): Binary matrix of worker responses
- `w` (Union[pd.DataFrame, pd.Series]): Worker weights used to compute weighted sum of responses.
- `n_workers` (int): Number of workers who provided classifications

**Returns:**
- pd.DataFrame: A DataFrame with multi-level columns containing confidence scores.

##### measuring_Tao_weights_based_on_classifier_labels

```python
@staticmethod
measuring_Tao_weights_based_on_classifier_labels(delta, noisy_true_labels)
```

Calculates normalized Tao weights based on classifier labels.

**Parameters:**
- `delta` (pd.DataFrame): Worker responses with shape (num_samples, n_workers)
- `noisy_true_labels` (pd.DataFrame or Series): The estimated true labels from a classifier

**Returns:**
- pd.DataFrame: Normalized Tao weights with shape (num_samples, n_workers)

##### measuring_Tao_weights_based_on_actual_labels

```python
@staticmethod
measuring_Tao_weights_based_on_actual_labels(workers_labels, noisy_true_labels, n_workers)
```

Calculate worker weights based on their accuracy relative to actual labels.

**Parameters:**
- `workers_labels` (pd.DataFrame): DataFrame where each row represents a sample and each column represents a worker's labels.
- `noisy_true_labels` (pd.DataFrame): DataFrame containing the noisy ground truth labels for each sample.
- `n_workers` (int): Number of workers.

**Returns:**
- pd.DataFrame: Normalized weights for each worker for each sample.

##### core_measurements

```python
@classmethod
core_measurements(cls, data, n_workers, config, feature_columns, seed, use_parallelization_benchmarks=False) -> Result2Type
```

Performs core measurements for the AIM1_3 algorithm.

**Parameters:**
- `data` (dict): Dictionary containing training and test data
- `n_workers` (int): Number of workers in the simulation
- `config` (Settings): Configuration object containing simulation parameters
- `feature_columns` (list): List of feature column names
- `seed` (int): Random seed for reproducibility
- `use_parallelization_benchmarks` (bool, optional): Whether to use parallelization for benchmark calculation. Default is False.

**Returns:**
- Result2Type: Object containing results for proposed methods, benchmark methods, weights, and worker strength information.

##### calculate_one_dataset

```python
@classmethod
calculate_one_dataset(cls, config: 'Settings', dataset_name: DatasetNames=DatasetNames.IONOSPHERE) -> ResultComparisonsType
```

Calculates various output metrics and analyses for a given dataset using the provided configuration.

**Parameters:**
- `config` (Settings): Configuration settings that include parameters, dataset details, and other experiment configurations.
- `dataset_name` (DatasetNames, optional): The identifier for the dataset to be processed. Default is DatasetNames.IONOSPHERE.

**Returns:**
- ResultComparisonsType: An object containing weight_strength_relation, outputs, and config.

##### calculate_all_datasets

```python
@classmethod
calculate_all_datasets(cls, config: 'Settings') -> Dict[DatasetNames, ResultComparisonsType]
```

Calculates results for all datasets specified in the configuration.

**Parameters:**
- `config` (Settings): Configuration settings.

**Returns:**
- Dict[DatasetNames, ResultComparisonsType]: Dictionary mapping dataset names to their results.

### BenchmarkTechniques

Class for benchmarking various crowd-sourcing techniques.

#### Constructor

```python
BenchmarkTechniques(crowd_labels, ground_truth)
```

**Parameters:**
- `crowd_labels` (Dict): Dictionary containing worker labels
- `ground_truth` (Dict): Dictionary containing ground truth labels

#### Methods

##### apply

```python
@classmethod
apply(cls, true_labels, use_parallelization_benchmarks) -> pd.DataFrame
```

Apply all benchmark techniques to the given labels.

**Parameters:**
- `true_labels` (dict): Dictionary containing true labels
- `use_parallelization_benchmarks` (bool): Whether to use parallelization for benchmark calculation

**Returns:**
- pd.DataFrame: DataFrame containing the results of all benchmark techniques

##### calculate

```python
calculate(self, use_parallelization_benchmarks) -> pd.DataFrame
```

Calculate the results of all benchmark techniques.

**Parameters:**
- `use_parallelization_benchmarks` (bool): Whether to use parallelization for benchmark calculation

**Returns:**
- pd.DataFrame: DataFrame containing the results of all benchmark techniques

#### Static Methods

##### get_techniques

```python
@classmethod
get_techniques(cls, benchmark_name: OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray)
```

Get the implementation of a specific benchmark technique.

**Parameters:**
- `benchmark_name` (OtherBenchmarkNames): Name of the benchmark technique
- `test` (pd.DataFrame): Test data
- `test_unique` (np.ndarray): Unique test data

**Returns:**
- function: Function implementing the specified benchmark technique

##### wrapper

```python
@staticmethod
wrapper(benchmark_name: OtherBenchmarkNames, test: pd.DataFrame, test_unique: np.ndarray) -> Tuple[OtherBenchmarkNames, np.ndarray]
```

Wrapper function for benchmark techniques.

**Parameters:**
- `benchmark_name` (OtherBenchmarkNames): Name of the benchmark technique
- `test` (pd.DataFrame): Test data
- `test_unique` (np.ndarray): Unique test data

**Returns:**
- Tuple[OtherBenchmarkNames, np.ndarray]: Tuple containing the benchmark name and the result

##### objective_function

```python
@staticmethod
objective_function(test, test_unique)
```

Objective function for benchmark techniques.

**Parameters:**
- `test` (pd.DataFrame): Test data
- `test_unique` (np.ndarray): Unique test data

**Returns:**
- function: Function implementing the objective function

##### reshape_dataframe_into_this_sdk_format

```python
@staticmethod
reshape_dataframe_into_this_sdk_format(df_predicted_labels)
```

Reshape a DataFrame into the format required by the SDK.

**Parameters:**
- `df_predicted_labels` (pd.DataFrame): DataFrame containing predicted labels

**Returns:**
- pd.DataFrame: Reshaped DataFrame

### LoadSaveFile

Utility class for loading and saving files.

#### Constructor

```python
LoadSaveFile(path)
```

**Parameters:**
- `path` (str or Path): Path to the file

#### Methods

##### load

```python
load(self, index_col=None, header=None)
```

Load a file.

**Parameters:**
- `index_col` (int or list, optional): Column(s) to use as the row labels of the DataFrame. Default is None.
- `header` (int, list, or None, optional): Row number(s) to use as the column names. Default is None.

**Returns:**
- object: Loaded file content

##### dump

```python
dump(self, file, index=False)
```

Save a file.

**Parameters:**
- `file` (object): Object to save
- `index` (bool, optional): Whether to include the index in the saved file. Default is False.

### Aim1_3_Data_Analysis_Results

Class for analyzing and visualizing the results of the AIM1_3 algorithm.

#### Constructor

```python
Aim1_3_Data_Analysis_Results(config)
```

**Parameters:**
- `config` (Settings): Configuration object containing simulation parameters

#### Methods

##### update

```python
update() -> 'Aim1_3_Data_Analysis_Results'
```

Update the analysis results with the latest data.

**Returns:**
- Aim1_3_Data_Analysis_Results: Self for method chaining

##### get_result

```python
get_result(metric_name='F_all', dataset_name: DatasetNames=DatasetNames.MUSHROOM, strategy=StrategyNames.FREQ, nl='NL3', seed_ix=0, method_name=ProposedTechniqueNames.PROPOSED, data_mode='test')
```

Get a specific result from the analysis.

**Parameters:**
- `metric_name` (str, optional): Name of the metric to retrieve. Default is 'F_all'.
- `dataset_name` (DatasetNames, optional): Name of the dataset. Default is DatasetNames.MUSHROOM.
- `strategy` (StrategyNames, optional): Name of the strategy. Default is StrategyNames.FREQ.
- `nl` (str, optional): Number of workers. Default is 'NL3'.
- `seed_ix` (int, optional): Seed index. Default is 0.
- `method_name` (ProposedTechniqueNames, optional): Name of the method. Default is ProposedTechniqueNames.PROPOSED.
- `data_mode` (str, optional): Data mode ('train' or 'test'). Default is 'test'.

**Returns:**
- object: The requested result

##### get_evaluation_metrics_for_confidence_scores

```python
get_evaluation_metrics_for_confidence_scores(metric_name='F_eval_one_dataset_all_workers', dataset_name: DatasetNames=DatasetNames.IONOSPHERE, nl='NL3')
```

Get evaluation metrics for confidence scores.

**Parameters:**
- `metric_name` (str, optional): Name of the metric to retrieve. Default is 'F_eval_one_dataset_all_workers'.
- `dataset_name` (DatasetNames, optional): Name of the dataset. Default is DatasetNames.IONOSPHERE.
- `nl` (str, optional): Number of workers. Default is 'NL3'.

**Returns:**
- pd.DataFrame: DataFrame containing evaluation metrics for confidence scores

##### save_outputs

```python
save_outputs(self, filename, relative_path, dataframe=None)
```

Save outputs to a file.

**Parameters:**
- `filename` (str): Name of the file
- `relative_path` (str): Relative path to save the file
- `dataframe` (pd.DataFrame, optional): DataFrame to save. Default is None.

##### figure_weight_quality_relation

```python
figure_weight_quality_relation(self, aspect=1.5, font_scale=1, fontsize=12, relative_path='final_figures', height=4)
```

Generate a figure showing the relationship between worker weight and quality.

**Parameters:**
- `aspect` (float, optional): Aspect ratio of the figure. Default is 1.5.
- `font_scale` (float, optional): Font scale. Default is 1.
- `fontsize` (int, optional): Font size. Default is 12.
- `relative_path` (str, optional): Relative path to save the figure. Default is 'final_figures'.
- `height` (int, optional): Height of the figure. Default is 4.

##### figure_metrics_mean_over_seeds_per_dataset_per_worker

```python
figure_metrics_mean_over_seeds_per_dataset_per_worker(self, metric: EvaluationMetricNames=EvaluationMetricNames.ACC, nl=3, figsize=(10, 10), font_scale=1.8, fontsize=20, relative_path='final_figures')
```

Generate a figure showing metrics averaged over seeds for each dataset and worker.

**Parameters:**
- `metric` (EvaluationMetricNames, optional): Metric to visualize. Default is EvaluationMetricNames.ACC.
- `nl` (int, optional): Number of workers. Default is 3.
- `figsize` (tuple, optional): Figure size. Default is (10, 10).
- `font_scale` (float, optional): Font scale. Default is 1.8.
- `fontsize` (int, optional): Font size. Default is 20.
- `relative_path` (str, optional): Relative path to save the figure. Default is 'final_figures'.

##### figure_metrics_all_datasets_workers

```python
figure_metrics_all_datasets_workers(self, workers_list: list[str]=None, figsize=(15, 15), font_scale=1.8, fontsize=20, relative_path='final_figures')
```

Generate a figure showing metrics for all datasets and workers.

**Parameters:**
- `workers_list` (list[str], optional): List of workers to include. Default is None.
- `figsize` (tuple, optional): Figure size. Default is (15, 15).
- `font_scale` (float, optional): Font scale. Default is 1.8.
- `fontsize` (int, optional): Font size. Default is 20.
- `relative_path` (str, optional): Relative path to save the figure. Default is 'final_figures'.

##### figure_F_heatmap

```python
figure_F_heatmap(self, metric_name='F_eval_one_dataset_all_workers', dataset_name:DatasetNames=DatasetNames.IONOSPHERE, nl='NL3', fontsize=20, font_scale=1.8, figsize=(13, 11), relative_path='final_figures')
```

Generate a heatmap of confidence scores.

**Parameters:**
- `metric_name` (str, optional): Name of the metric to visualize. Default is 'F_eval_one_dataset_all_workers'.
- `dataset_name` (DatasetNames, optional): Name of the dataset. Default is DatasetNames.IONOSPHERE.
- `nl` (str, optional): Number of workers. Default is 'NL3'.
- `fontsize` (int, optional): Font size. Default is 20.
- `font_scale` (float, optional): Font scale. Default is 1.8.
- `figsize` (tuple, optional): Figure size. Default is (13, 11).
- `relative_path` (str, optional): Relative path to save the figure. Default is 'final_figures'.

### AIM1_3_Plot

Class for plotting results from the AIM1_3 algorithm.

#### Constructor

```python
AIM1_3_Plot(plot_data: pd.DataFrame)
```

**Parameters:**
- `plot_data` (pd.DataFrame): Data to plot

#### Methods

##### plot

```python
plot(self, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, interpolation_pt_count=1000, show_markers=ProposedTechniqueNames.PROPOSED)
```

Plot the data.

**Parameters:**
- `xlabel` (str, optional): Label for the x-axis. Default is ''.
- `ylabel` (str, optional): Label for the y-axis. Default is ''.
- `xticks` (bool, optional): Whether to show x-ticks. Default is True.
- `title` (str, optional): Title of the plot. Default is ''.
- `legend` (list, optional): Legend labels. Default is None.
- `smooth` (bool, optional): Whether to smooth the plot. Default is True.
- `interpolation_pt_count` (int, optional): Number of points for interpolation. Default is 1000.
- `show_markers` (ProposedTechniqueNames, optional): Which technique to show markers for. Default is ProposedTechniqueNames.PROPOSED.

#### Static Methods

##### data_interpolation

```python
@staticmethod
data_interpolation(x, y, smooth=False, interpolation_pt_count=1000)
```

Interpolate data for smoother plotting.

**Parameters:**
- `x` (array-like): x-coordinates
- `y` (array-like): y-coordinates
- `smooth` (bool, optional): Whether to smooth the data. Default is False.
- `interpolation_pt_count` (int, optional): Number of points for interpolation. Default is 1000.

**Returns:**
- tuple: Tuple containing the interpolated x and y coordinates

## Data Types

### ResultType

Data class for storing results of the AIM1_3 algorithm.

```python
@dataclass
class ResultType:
    confidence_scores: dict[str, pd.DataFrame]
    aggregated_labels: pd.DataFrame
    metrics: pd.DataFrame
```

### WeightType

Data class for storing weights calculated by different methods.

```python
@dataclass
class WeightType:
    PROPOSED: pd.DataFrame
    TAO: pd.DataFrame
    SHENG: pd.DataFrame
```

### Result2Type

Data class for storing comprehensive results of the AIM1_3 algorithm.

```python
@dataclass
class Result2Type:
    proposed: ResultType
    benchmark: ResultType
    weight: WeightType
    workers_strength: pd.DataFrame
    n_workers: int
    true_label: dict[str, pd.DataFrame]
```

### ResultComparisonsType

Data class for storing comparison results across different datasets.

```python
@dataclass
class ResultComparisonsType:
    outputs: dict
    config: 'Settings'
    weight_strength_relation: pd.DataFrame
```

## Enumerations

The library uses several enumerations to define constants and options:

- **DatasetNames**: Names of datasets available for analysis
- **UncertaintyTechniques**: Techniques for calculating uncertainty
- **ConsistencyTechniques**: Techniques for calculating consistency
- **ProposedTechniqueNames**: Names of proposed techniques
- **MainBenchmarks**: Names of main benchmark techniques
- **OtherBenchmarkNames**: Names of other benchmark techniques
- **StrategyNames**: Names of strategies for confidence scoring
- **EvaluationMetricNames**: Names of evaluation metrics
- **SimulationMethods**: Methods for simulation
- **OutputModes**: Modes for output handling

---
