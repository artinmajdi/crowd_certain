"""
Main AIM1_3 class module for crowd-certain.

This module contains a simplified version of the AIM1_3 class that uses
components from other modules to perform crowd-sourcing simulations and analyses.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from crowd_certain.utilities.parameters import params
from crowd_certain.utilities.io import dataset_loader, HDF5Storage, LoadSaveFile
from crowd_certain.utilities.parameters.settings import Settings
from crowd_certain.utilities._components.orchestrator import Orchestrator

# Whether to use HDF5 storage for saving/loading results
USE_HD5F_STORAGE = True

class AIM1_3:
    """
    Main class for AIM1_3 implementation that handles the calculation of worker weights,
    uncertainties, and confidence scores for crowdsourced label aggregation.

    This class is a simplified version that uses components from other modules.
    It serves as the main entry point for running simulations on datasets.
    """

    def __init__(self, data: dict, feature_columns: list, config: Settings, n_workers: int = 3, seed: int = 0):
        """
        Initialize the AIM1_3 class.

        Parameters
        ----------
        data : dict
            Dictionary containing training and test data
        feature_columns : list
            List of feature column names
        config : Settings
            Configuration object containing simulation parameters
        n_workers : int, default=3
            Number of workers in the simulation
        seed : int, default=0
            Random seed for reproducibility
        """
        self.data = data
        self.feature_columns = feature_columns
        self.config = config
        self.n_workers = n_workers
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed + 1)

        # Create orchestrator
        self.orchestrator = Orchestrator(
            config=config,
            data=data,
            feature_columns=feature_columns
        )

    def get_outputs(self) -> Dict[str, list]:
        """
        Retrieve outputs for the simulation.

        Returns
        -------
        Dict[str, list]
            Dictionary containing simulation outputs for each worker count and seed
        """
        return self.orchestrator.get_outputs()

    def worker_weight_strength_relation(self, seed: int = 0, n_workers: int = 10) -> pd.DataFrame:
        """
        Calculate and retrieve the relationship between worker strength and weights.

        This method either calculates the relationship using the core_measurements function,
        or loads it from a saved file, depending on the configuration output mode.

        Parameters
        ----------
        seed : int, default=0
            Random seed for reproducibility
        n_workers : int, default=10
            Number of workers to simulate

        Returns
        -------
        pd.DataFrame
            DataFrame containing worker strength-weight relationship
        """
        metric_name = 'weight_strength_relation'
        path_main = self.config.output.path / metric_name / str(self.config.dataset.dataset_name)

        if USE_HD5F_STORAGE:
            h5s = HDF5Storage(path_main / f'{metric_name}.h5')
        else:
            lsf = LoadSaveFile(path_main / f'{metric_name}.xlsx')

        if self.config.output.mode is params.OutputModes.CALCULATE:
            # Calculate the relationship using Orchestrator
            df = self.orchestrator.core_measurements(
                n_workers=n_workers,
                seed=seed,
                use_parallelization_benchmarks=self.config.simulation.use_parallelization
            ).workers_strength.set_index('workers_strength').sort_index()

            # Save if configured
            if self.config.output.save:
                if USE_HD5F_STORAGE:
                    h5s.save_dataframe(df, '/weight_strength_relation')
                else:
                    lsf.dump(df)

            return df

        # Load from storage
        if USE_HD5F_STORAGE:
            return h5s.load_dataframe('/weight_strength_relation')
        return lsf.load(header=0)

    @classmethod
    def calculate_one_dataset(cls, config: Settings,
                            dataset_name: Optional[params.DatasetNames] = None) -> params.ResultComparisonsType:
        """
        Calculate results for a single dataset.

        Parameters
        ----------
        config : Settings
            Configuration settings
        dataset_name : Optional[params.DatasetNames], default=None
            Dataset name to use (overrides config.dataset.dataset_name)

        Returns
        -------
        params.ResultComparisonsType
            Results of the calculation
        """
        # Update dataset name if provided
        if dataset_name is not None:
            config.dataset.dataset_name = dataset_name

        # Check if results exist in storage
        if USE_HD5F_STORAGE and config.output.mode is params.OutputModes.LOAD:
            h5_path = config.output.path / 'results' / f'{config.dataset.dataset_name}.h5'
            h5s = HDF5Storage(h5_path)
            result = h5s.load_result_comparisons(config.dataset.dataset_name.value)
            if result is not None:
                return result

        # Load dataset
        data, feature_columns = dataset_loader.load_dataset(config=config)

        # Create AIM1_3 instance
        aim1_3 = cls(data=data, feature_columns=feature_columns, config=config)

        # Get outputs and weight-strength relation
        outputs = aim1_3.get_outputs()
        weight_strength_relation = aim1_3.worker_weight_strength_relation(seed=0, n_workers=10)

        # Create result object
        result = params.ResultComparisonsType(
            weight_strength_relation=weight_strength_relation,
            outputs=outputs,
            config=config
        )

        # Save result if configured
        if USE_HD5F_STORAGE and config.output.save:
            h5_path = config.output.path / 'results' / f'{config.dataset.dataset_name}.h5'
            h5s = HDF5Storage(h5_path)
            h5s.save_result_comparisons(result, config.dataset.dataset_name.value)

        return result

    @classmethod
    def calculate_all_datasets(cls, config: Settings) -> Dict[params.DatasetNames, params.ResultComparisonsType]:
        """
        Calculate results for all datasets in the configuration.

        Parameters
        ----------
        config : Settings
            Configuration settings

        Returns
        -------
        Dict[params.DatasetNames, params.ResultComparisonsType]
            Dictionary mapping dataset names to their results
        """
        # Check if we should load from storage
        if USE_HD5F_STORAGE and config.output.mode is params.OutputModes.LOAD:
            h5_path = config.output.path / 'results' / 'all_datasets.h5'
            h5s = HDF5Storage(h5_path)
            results = h5s.load_all_datasets_results(config.dataset.datasetNames)
            if results and len(results) == len(config.dataset.datasetNames):
                return results

        # Calculate results for each dataset
        results = {}
        for dt in config.dataset.datasetNames:
            results[dt] = cls.calculate_one_dataset(dataset_name=dt, config=config)

        # Save results if configured
        if USE_HD5F_STORAGE and config.output.save:
            h5_path = config.output.path / 'results' / 'all_datasets.h5'
            h5s = HDF5Storage(h5_path)
            h5s.save_all_datasets_results(results)

        return results
