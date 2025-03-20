"""
Worker simulation module for crowd-certain.

This module provides functionality for generating simulated workers with varying skill levels
and creating noisy labels for crowdsourcing simulations.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import ensemble as sk_ensemble

from crowd_certain.utilities.parameters import params
from crowd_certain.utilities.utils import ClassifierTraining, CrowdCertainOrchestrator


class WorkerSimulator:
    """
    A class for simulating workers with varying skill levels and their label predictions.
    """

    def __init__(self, config, data, feature_columns, n_workers, seed=0):
        """
        Initialize the WorkerSimulator.

        Parameters
        ----------
        config : Settings
            Configuration object containing simulation parameters
        data : dict
            Dictionary containing training and test data
        feature_columns : list
            List of feature column names
        n_workers : int
            Number of workers to simulate
        seed : int, default=0
            Random seed for reproducibility
        """
        self.config = config
        self.data = data
        self.feature_columns = feature_columns
        self.n_workers = n_workers
        self.seed = seed
        np.random.seed(self.seed + 1)

    def generate_workers(self) -> pd.DataFrame:
        """
        Generates a dataframe of workers with randomly assigned strength values.

        The function creates a list of worker names and assigns a random strength value
        to each worker from a uniform distribution defined by the configuration parameters.

        Returns
        -------
        pd.DataFrame
            A dataframe with worker names as the index and their assigned
            strength values in a column named 'workers_strength'.
        """
        workers_names = [f'worker_{j}' for j in range(self.n_workers)]

        workers_strength_array = np.random.uniform(
            low=self.config.simulation.low_dis,
            high=self.config.simulation.high_dis,
            size=self.n_workers
        )

        return pd.DataFrame({'workers_strength': workers_strength_array}, index=workers_names)

    def generate_noisy_labels(self, truth_array: np.ndarray, worker_strength: float) -> np.ndarray:
        """
        Generate noisy labels by flipping true labels with a probability determined by labeling strength.

        This function simulates worker annotations by introducing noise to ground truth labels.
        Each sample has a probability of (1 - worker_strength) to have its true label flipped.

        Parameters
        ----------
        truth_array : np.ndarray
            Array of ground truth probabilities/scores for each sample.

        worker_strength : float
            Labeling strength parameter in range [0,1].
            Higher values mean more accurate labels (less noise).
            At worker_strength=1, no labels are flipped.
            At worker_strength=0, all labels are flipped.

        Returns
        -------
        np.ndarray
            Binary array of noisy labels where some true labels are flipped based on worker_strength.
        """
        # number of samples
        num_samples = truth_array.shape[0]

        # finding a random number for each instance
        true_label_assignment_prob = np.random.random(num_samples)

        # samples that will have an inaccurate true label
        false_samples = true_label_assignment_prob < 1 - worker_strength

        # measuring the new labels for each worker
        worker_truth = truth_array > 0.5
        worker_truth[false_samples] = ~worker_truth[false_samples]

        return worker_truth

    def simulate_workers(self) -> Tuple[Dict, Dict, Dict, pd.DataFrame]:
        """
        Simulate workers with varying skill levels and measure their predictions and uncertainties.

        This method performs a simulation where:
        1. Workers with different strengths (skills) are generated randomly
        2. These workers provide noisy labels for the data based on their strength
        3. Classifiers are trained on these noisy labels to make predictions
        4. Different simulation methods can be used (random states or multiple classifiers)

        Returns
        -------
        tuple
            A tuple containing:
            - preds (dict): Predictions organized by mode ('train'/'test'), simulation, and worker
            - uncertainties (dict): Uncertainty measures for each worker and uncertainty technique
            - truth (dict): The true labels and worker-annotated labels
            - workers_strength (pd.DataFrame): Worker strength information including:
                * workers_strength: The randomly assigned strength value
                * accuracy-test-classifier: Accuracy of each worker's classifier on test data
                * accuracy-test: Accuracy of each worker's noisy labels compared to ground truth
        """
        # Generate worker strengths
        workers_strength = self.generate_workers()

        # Simulate worker responses and predictions
        truth, uncertainties, preds = self._simulate_worker_responses(workers_strength)

        # Calculate accuracy metrics for each worker
        self._add_worker_accuracy_metrics(workers_strength, preds, truth)

        return preds, uncertainties, truth, workers_strength

    def _simulate_worker_responses(self, workers_strength) -> Tuple[Dict, Dict, Dict]:
        """
        Performs simulations to evaluate worker performance in crowdsourced labeling tasks.

        This function executes a series of simulations that:
        1. Initializes dataframes for predicted labels, ground truth, and uncertainties
        2. Generates simulated noisy labels for each worker based on their strength
        3. Trains classifiers on these noisy labels and makes predictions
        4. Calculates uncertainty metrics across multiple simulations
        5. Aggregates results across workers and simulations

        Parameters
        ----------
        workers_strength : pd.DataFrame
            DataFrame containing worker strength values

        Returns
        -------
        tuple
            Contains (truth, uncertainties, preds)
            - truth: Dict with train/test DataFrames containing ground truth and noisy labels
            - uncertainties: Dict with train/test DataFrames containing uncertainty metrics per worker
            - preds: Dict with predictions reorganized by simulation rather than by worker
        """
        predicted_labels_all_sims = {'train': {}, 'test': {}}
        truth = {'train': pd.DataFrame(), 'test': pd.DataFrame()}

        # Set up columns for uncertainty metrics
        columns = pd.MultiIndex.from_product(
            [workers_strength.index, [l.value for l in self.config.technique.uncertainty_techniques]],
            names=['worker', 'uncertainty_technique']
        )

        uncertainties = {'train': pd.DataFrame(columns=columns), 'test': pd.DataFrame(columns=columns)}

        # Loop through all workers to generate noisy labels and predictions
        for worker_index, worker in enumerate(workers_strength.index):
            # Initialize
            for mode in ['train', 'test']:
                predicted_labels_all_sims[mode][worker] = {}
                if worker_index == 0:  # Only set truth once
                    truth[mode]['truth'] = self.data[mode].true.copy()

            # Generate noisy labels
            truth['train'][worker] = self.generate_noisy_labels(
                truth_array=self.data['train'].true.values,
                worker_strength=workers_strength.loc[worker, 'workers_strength']
            )

            truth['test'][worker] = self.generate_noisy_labels(
                truth_array=self.data['test'].true.values,
                worker_strength=workers_strength.loc[worker, 'workers_strength']
            )

            # Train classifiers and make predictions
            n_simulations = self._get_n_simulations()
            for sim_num in range(n_simulations):
                self._update_predicted_labels(
                    predicted_labels_all_sims=predicted_labels_all_sims,
                    worker=worker,
                    sim_num=sim_num,
                    truth=truth
                )

            # Calculate uncertainties for this worker
            for mode in ['train', 'test']:
                # Convert to dataframe
                predicted_labels_all_sims[mode][worker] = pd.DataFrame(
                    predicted_labels_all_sims[mode][worker],
                    index=self.data[mode].index
                )

                # Get uncertainties for this worker using the UncertaintyCalculator
                from crowd_certain.utilities._components.uncertainty import (
                    UncertaintyCalculator,
                )
                calculator = UncertaintyCalculator(self.config)
                uncertainties[mode][worker] = calculator.calculate_uncertainties(
                    df=predicted_labels_all_sims[mode][worker]
                )

                # Add majority vote prediction
                predicted_labels_all_sims[mode][worker]['mv'] = (
                    predicted_labels_all_sims[mode][worker].mean(axis=1) > 0.5
                )

        # Reorganize predictions by simulation instead of by worker
        preds = self._swap_prediction_axes(predicted_labels_all_sims)

        return truth, uncertainties, preds

    def _update_predicted_labels(self, predicted_labels_all_sims, worker, sim_num, truth):
        """
        Trains a classifier and updates predictions for a specific worker and simulation.

        Parameters
        ----------
        predicted_labels_all_sims : dict
            Dictionary to store predictions
        worker : str
            Worker identifier
        sim_num : int
            Simulation number
        truth : dict
            Dictionary containing true and noisy labels
        """
        classifier = self._get_classifier(sim_num)

        # Train classifier on noisy labels for this worker
        classifier.fit(
            X=self.data['train'][self.feature_columns],
            y=truth['train'][worker]
        )

        # Make predictions for both train and test sets
        for mode in ['train', 'test']:
            predicted_labels_all_sims[mode][worker][f'simulation_{sim_num}'] = \
                classifier.predict(self.data[mode][self.feature_columns])

    def _get_classifier(self, sim_num):
        """
        Returns a classifier based on the simulation method configuration.

        Parameters
        ----------
        sim_num : int
            Simulation number

        Returns
        -------
        sklearn estimator
            A classifier instance configured according to the simulation method.
        """
        if self.config.simulation.simulation_methods is params.SimulationMethods.RANDOM_STATES:
            return sk_ensemble.RandomForestClassifier(
                n_estimators=4,
                max_depth=4,
                random_state=self.seed * sim_num
            )
        elif self.config.simulation.simulation_methods is params.SimulationMethods.MULTIPLE_CLASSIFIERS:
            return ClassifierTraining.classifiers_list[sim_num]

    def _get_n_simulations(self):
        """
        Determines the number of simulations to be run based on the simulation method.

        Returns
        -------
        int
            Number of simulations to run
        """
        if self.config.simulation.simulation_methods is params.SimulationMethods.RANDOM_STATES:
            return self.config.simulation.num_simulations
        elif self.config.simulation.simulation_methods is params.SimulationMethods.MULTIPLE_CLASSIFIERS:
            return len(self.config.simulation.classifiers_list)

    def _swap_prediction_axes(self, predicted_labels_all_sims):
        """
        Swaps the axes of the predicted labels dataframe organization.

        Transforms the structure from {mode: {worker: {simulation: predictions}}} to
        {mode: {simulation: dataframe}}, where each dataframe has workers as columns.

        Parameters
        ----------
        predicted_labels_all_sims : dict
            Nested dictionary containing predictions

        Returns
        -------
        dict
            Reorganized predictions
        """
        from collections import defaultdict

        # Reshaping the dataframes
        preds_swapped = {'train': defaultdict(pd.DataFrame), 'test': defaultdict(pd.DataFrame)}

        for mode in ['train', 'test']:
            for i in range(self.config.simulation.num_simulations + 1):
                sim = f'simulation_{i}' if i < self.config.simulation.num_simulations else 'mv'

                preds_swapped[mode][sim] = pd.DataFrame()
                for worker in workers_strength.index:
                    preds_swapped[mode][sim][worker] = predicted_labels_all_sims[mode][worker][sim]

        return preds_swapped

    def _add_worker_accuracy_metrics(self, workers_strength, preds, truth):
        """
        Calculate and add accuracy measures for each worker.

        This function computes two types of accuracy metrics for each worker:
        1. 'accuracy-test-classifier': The accuracy of the classifier's predictions against true labels
        2. 'accuracy-test': The accuracy of each worker's noisy labels against true labels

        Parameters
        ----------
        workers_strength : pd.DataFrame
            DataFrame to update with accuracy metrics
        preds : dict
            Dictionary containing predictions
        truth : dict
            Dictionary containing true labels
        """
        workers_strength['accuracy-test-classifier'] = 0.0
        workers_strength['accuracy-test'] = 0.0

        for worker in workers_strength.index:
            # Accuracy of classifier in simulation_0
            workers_strength.loc[worker, 'accuracy-test-classifier'] = (
                (preds['test']['simulation_0'][worker] == truth['test'].truth).mean()
            )

            # Accuracy of noisy true labels
            workers_strength.loc[worker, 'accuracy-test'] = (
                (truth['test'][worker] == truth['test'].truth).mean()
            )
