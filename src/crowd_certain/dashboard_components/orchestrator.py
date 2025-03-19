"""
Orchestrator module for crowd-certain.

This module contains the Orchestrator class which coordinates the high-level workflow
for crowd-sourcing simulations, including worker simulation, weight calculation,
and confidence score computation.
"""

import functools
import multiprocessing
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from crowd_certain.utilities.parameters import params
from crowd_certain.utilities.io.hdf5_storage import HDF5Storage
from crowd_certain.utilities.parameters.settings import Settings
from crowd_certain.utilities._components.worker_simulation import WorkerSimulator
from crowd_certain.utilities._components.weighting_schemes import WeightingSchemes
from crowd_certain.utilities._components.confidence_scoring import ConfidenceScorer
from crowd_certain.utilities._components.metrics import MetricsCalculator

# Whether to use HDF5 storage for saving/loading results
USE_HD5F_STORAGE = True

class Orchestrator:
    """
    Class for coordinating the high-level workflow for crowd-sourcing simulations.

    This class manages the process of simulating workers, calculating weights,
    computing confidence scores, and evaluating results.
    """

    def __init__(self, config: Settings, data: dict, feature_columns: list):
        """
        Initialize the Orchestrator.

        Parameters
        ----------
        config : Settings
            Configuration object containing simulation parameters
        data : dict
            Dictionary containing training and test data
        feature_columns : list
            List of feature column names
        """
        self.config = config
        self.data = data
        self.feature_columns = feature_columns

    def core_measurements(self, n_workers: int, seed: int,
                        use_parallelization_benchmarks: bool = False) -> params.Result2Type:
        """
        Main method that orchestrates the calculation pipeline.

        Parameters
        ----------
        n_workers : int
            Number of workers in the simulation
        seed : int
            Random seed for reproducibility
        use_parallelization_benchmarks : bool, default=False
            Whether to use parallelization for benchmark calculations

        Returns
        -------
        params.Result2Type
            Results of the simulation including proposed methods, benchmarks,
            weights, worker strengths, and true labels
        """
        # Create worker simulator
        worker_sim = WorkerSimulator(
            config=self.config,
            data=self.data,
            feature_columns=self.feature_columns,
            n_workers=n_workers,
            seed=seed
        )

        # Simulate workers and get results
        preds_all, uncertainties_all, true_labels, workers_strength = worker_sim.simulate_workers()

        # Calculate weights for proposed techniques and benchmarks
        weighting = WeightingSchemes(self.config)
        weights = weighting.get_weights(
            workers_labels=preds_all['test']['simulation_0'],
            preds=preds_all['test']['mv'],
            uncertainties=uncertainties_all['test'],
            noisy_true_labels=true_labels['test'].drop(columns=['truth']),
            n_workers=n_workers
        )

        # Calculate results for proposed techniques and benchmarks
        results_proposed, results_benchmarks = self._calculate_results(
            weights=weights,
            preds_all=preds_all,
            true_labels=true_labels,
            n_workers=n_workers,
            use_parallelization_benchmarks=use_parallelization_benchmarks
        )

        # Return all results
        return params.Result2Type(
            proposed=results_proposed,
            benchmark=results_benchmarks,
            weight=weights,
            workers_strength=workers_strength,
            n_workers=n_workers,
            true_label=true_labels
        )

    def _calculate_results(self, weights: params.WeightType, preds_all: dict,
                          true_labels: dict, n_workers: int,
                          use_parallelization_benchmarks: bool = False) -> Tuple[params.ResultType, params.ResultType]:
        """
        Calculate confidence scores and aggregated labels using the provided weights.

        Parameters
        ----------
        weights : params.WeightType
            Object containing weights for proposed and benchmark methods
        preds_all : dict
            Dictionary containing worker predictions
        true_labels : dict
            Dictionary containing true labels
        n_workers : int
            Number of workers in the simulation
        use_parallelization_benchmarks : bool, default=False
            Whether to use parallelization for benchmark calculations

        Returns
        -------
        Tuple[params.ResultType, params.ResultType]
            Results for proposed methods and benchmark methods
        """
        # Calculate results for proposed methods
        results_proposed = self._calculate_proposed_results(
            weights=weights,
            preds=preds_all['test']['mv'],
            true_labels=true_labels
        )

        # Calculate results for benchmark methods
        results_benchmarks = self._calculate_benchmark_results(
            weights=weights,
            workers_labels=preds_all['test']['simulation_0'],
            true_labels=true_labels,
            n_workers=n_workers,
            use_parallelization_benchmarks=use_parallelization_benchmarks
        )

        return results_proposed, results_benchmarks

    def _calculate_proposed_results(self, weights: params.WeightType, preds: pd.DataFrame,
                                  true_labels: dict) -> params.ResultType:
        """
        Calculate results for proposed methods.

        Parameters
        ----------
        weights : params.WeightType
            Object containing weights for proposed methods
        preds : pd.DataFrame
            DataFrame containing worker predictions
        true_labels : dict
            Dictionary containing true labels

        Returns
        -------
        params.ResultType
            Results for proposed methods
        """
        # Initialize output DataFrames
        agg_labels = pd.DataFrame(columns=weights.PROPOSED.index, index=preds.index)
        confidence_scores = {}
        metrics = pd.DataFrame(
            columns=weights.PROPOSED.index,
            index=params.EvaluationMetricNames.values()
        )

        # For each proposed technique
        for technique in weights.PROPOSED.index:
            # Calculate aggregated labels
            agg_labels[technique] = (preds * weights.PROPOSED.T[technique]).sum(axis=1)

            # Calculate confidence scores
            confidence_scores[technique] = ConfidenceScorer.calculate_confidence_scores(
                delta=preds,
                w=weights.PROPOSED.T[technique],
                n_workers=len(preds.columns)
            )

            # Calculate evaluation metrics
            metrics[technique] = MetricsCalculator.get_AUC_ACC_F1(
                aggregated_labels=agg_labels[technique],
                truth=true_labels['test'].truth
            )

        return params.ResultType(
            aggregated_labels=agg_labels,
            confidence_scores=confidence_scores,
            metrics=metrics
        )

    def _calculate_benchmark_results(self, weights: params.WeightType, workers_labels: pd.DataFrame,
                                   true_labels: dict, n_workers: int,
                                   use_parallelization_benchmarks: bool = False) -> params.ResultType:
        """
        Calculate results for benchmark methods.

        Parameters
        ----------
        weights : params.WeightType
            Object containing weights for benchmark methods
        workers_labels : pd.DataFrame
            DataFrame containing worker labels
        true_labels : dict
            Dictionary containing true labels
        n_workers : int
            Number of workers in the simulation
        use_parallelization_benchmarks : bool, default=False
            Whether to use parallelization for benchmark calculations

        Returns
        -------
        params.ResultType
            Results for benchmark methods
        """
        # Calculate aggregated labels for TAO and SHENG methods
        agg_labels = pd.DataFrame(index=workers_labels.index, columns=params.MainBenchmarks.values())
        confidence_scores = {}

        for m in params.MainBenchmarks:
            # Select appropriate weights
            w = weights.TAO if m is params.MainBenchmarks.TAO else weights.SHENG

            # Calculate aggregated labels
            agg_labels[m.value] = (workers_labels * w).sum(axis=1)

            # Calculate confidence scores
            confidence_scores[m.value] = ConfidenceScorer.calculate_confidence_scores(
                delta=workers_labels,
                w=w,
                n_workers=n_workers
            )

        # Add other benchmarks
        if hasattr(self, '_calculate_other_benchmarks'):
            other_benchmarks = self._calculate_other_benchmarks(
                true_labels=true_labels,
                use_parallelization=use_parallelization_benchmarks
            )
            agg_labels = pd.concat([agg_labels, other_benchmarks], axis=1)

        # Calculate metrics for all benchmark methods
        metrics = pd.DataFrame({
            col: MetricsCalculator.get_AUC_ACC_F1(
                aggregated_labels=agg_labels[col],
                truth=true_labels['test'].truth
            ) for col in agg_labels.columns
        })

        return params.ResultType(
            aggregated_labels=agg_labels,
            confidence_scores=confidence_scores,
            metrics=metrics
        )

    def get_outputs(self) -> Dict[str, List[params.ResultType]]:
        """
        Retrieves the outputs for the calculation or loads them from file.

        Returns
        -------
        Dict[str, List[params.ResultType]]
            A dictionary mapping output keys to lists containing results for each seed
        """
        # Set up storage path
        if USE_HD5F_STORAGE:
            path = self.config.output.path / 'outputs' / f'{self.config.dataset.dataset_name}.h5'
            h5s = HDF5Storage(path)
        else:
            path = self.config.output.path / 'outputs' / f'{self.config.dataset.dataset_name}.pkl'
            from crowd_certain.utilities.io.dataset_loader import LoadSaveFile
            lsf = LoadSaveFile(path)

        # If loading from file
        if self.config.output.mode is params.OutputModes.LOAD:
            if USE_HD5F_STORAGE:
                return h5s.load(f'/{self.config.dataset.dataset_name}/outputs')
            return lsf.load()

        # If calculating
        # Create input list with all worker-seed combinations
        input_list = [
            {'nl': nl, 'seed': seed}
            for nl in self.config.simulation.workers_list
            for seed in range(self.config.simulation.num_seeds)
        ]

        # Define function to calculate results for each combination
        function = self._create_objective_function()

        # Calculate results with or without parallelization
        if self.config.simulation.use_parallelization:
            with multiprocessing.Pool(
                processes=min(
                    self.config.simulation.max_parallel_workers,
                    len(input_list)
                )
            ) as pool:
                core_results = pool.map(function, input_list)
        else:
            core_results = [function(args) for args in input_list]

        # Organize results
        outputs = {
            f'NL{nl}': np.full(self.config.simulation.num_seeds, np.nan).tolist()
            for nl in self.config.simulation.workers_list
        }

        for args, values in core_results:
            outputs[f'NL{args["nl"]}'][args['seed']] = values

        # Save results if configured
        if self.config.output.save:
            if USE_HD5F_STORAGE:
                h5s.save(data=outputs, group_path='/outputs')
            else:
                lsf.dump(outputs)

        return outputs

    def _create_objective_function(self) -> Callable:
        """
        Create objective function for calculating results.

        Returns
        -------
        Callable
            Function that calculates results for given arguments
        """
        return functools.partial(
            self._wrapper,
            config=self.config,
            data=self.data,
            feature_columns=self.feature_columns
        )

    @staticmethod
    def _wrapper(args: Dict, config: Settings, data,
               feature_columns) -> Tuple[Dict, params.Result2Type]:
        """
        Wrapper function for calculating results.

        Parameters
        ----------
        args : Dict
            Arguments for calculation
        config : Settings
            Configuration object
        data : dict
            Data dictionary
        feature_columns : list
            Feature column names

        Returns
        -------
        Tuple[Dict, params.Result2Type]
            Arguments and results
        """
        orchestrator = Orchestrator(config=config, data=data, feature_columns=feature_columns)
        result = orchestrator.core_measurements(
            n_workers=args['nl'],
            seed=args['seed'],
            use_parallelization_benchmarks=False
        )
        return args, result
