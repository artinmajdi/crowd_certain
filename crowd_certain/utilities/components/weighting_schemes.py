"""
Weighting schemes module for crowd-certain.

This module contains classes and functions for calculating various weighting schemes
for crowd workers, including proposed techniques and benchmark methods like TAO and SHENG.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union

from crowd_certain.utilities import params
from crowd_certain.utilities.components.uncertainty import calculate_consistency

class WeightingSchemes:
    """
    Class for calculating worker weights using different schemes.
    """

    def __init__(self, config):
        """
        Initialize WeightingSchemes.

        Parameters
        ----------
        config : Settings
            Configuration object containing parameters for weighting schemes
        """
        self.config = config

    def calculate_proposed_weights(self, preds: pd.DataFrame, uncertainties: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates proposed weights for worker predictions based on consistency and uncertainty metrics.

        This method computes two sets of weights based on different techniques:
        1. The first technique uses raw consistency scores (T1)
        2. The second technique zeros out consistency scores for incorrect predictions (T2)

        The weights are then normalized by the mean weight across all workers.

        Parameters
        ----------
        preds : pd.DataFrame
            Worker predictions with data indices as rows and workers as columns
            Shape: (n_samples, n_workers)
        uncertainties : pd.DataFrame
            Uncertainty metrics for each worker and technique
            Shape: (n_samples, n_workers * n_uncertainty_techniques) with MultiIndex columns [worker, uncertainty_technique]

        Returns
        -------
        pd.DataFrame
            Calculated weights for each worker across different consistency techniques,
            uncertainty techniques, and proposed weighting methods.
            The DataFrame has a MultiIndex with levels:
            [ConsistencyTechnique, UncertaintyTechnique, ProposedTechniqueName]
            and columns representing workers.
        """
        # Calculate majority vote binary prediction
        prob_mv_binary = preds.mean(axis=1) > 0.5

        # Calculate consistency scores from uncertainties
        T1 = calculate_consistency(uncertainties, self.config)
        T2 = T1.copy()

        proposed_techniques = [l.value for l in params.ProposedTechniqueNames]
        w_hat = pd.DataFrame(index=proposed_techniques, columns=T1.columns, dtype=float)

        for worker in preds.columns:
            # For T2, zero out consistency scores where worker disagrees with majority vote
            T2.loc[preds[worker].values != prob_mv_binary.values, worker] = 0

            # Calculate weights for both proposed techniques
            w_hat[worker] = pd.DataFrame.from_dict({
                proposed_techniques[0]: T1[worker].mean(axis=0),
                proposed_techniques[1]: T2[worker].mean(axis=0)
            }, orient='index')

        # Normalize weights by mean across workers
        w_hat_mean_over_workers = w_hat.T.groupby(level=[1, 2]).sum().T

        weights = pd.DataFrame().reindex_like(w_hat)
        for worker in preds.columns:
            weights[worker] = w_hat[worker].divide(w_hat_mean_over_workers)

        # Unstack weights to return MultiIndex DataFrame
        weights = weights.unstack().unstack(level='worker')

        return weights

    @staticmethod
    def calculate_tao_weights(delta: pd.DataFrame, noisy_true_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates normalized Tao weights based on classifier labels.

        This function computes weights for worker responses using the agreement
        between worker responses and the classifier's estimated true labels.

        Parameters
        ----------
        delta : pd.DataFrame
            Worker responses with shape (num_samples, n_workers), where each cell contains
            a binary response (True/False) or (1/0) from a worker for a specific sample.

        noisy_true_labels : pd.DataFrame or pd.Series
            The estimated true labels from a classifier with shape (num_samples, 1).

        Returns
        -------
        pd.DataFrame
            Normalized Tao weights with shape (num_samples, n_workers), where each value
            represents the weight assigned to a worker's response for a specific sample.
        """
        tau = (delta == noisy_true_labels).mean(axis=0)

        # Number of workers
        M = len(delta.columns)

        # Number of true and false labels for each class and sample
        true_counts = delta.sum(axis=1)
        false_counts = M - true_counts

        # Measuring the "specific quality of instances"
        s = delta.multiply(true_counts - 1, axis=0) + (~delta).multiply(false_counts - 1, axis=0)
        gamma = (1 + s ** 2) * tau
        W_hat_Tao = gamma.apply(lambda x: 1 / (1 + np.exp(-x)))
        z = W_hat_Tao.mean(axis=1)

        return W_hat_Tao.divide(z, axis=0)

    @staticmethod
    def calculate_tao_weights_with_actual_labels(workers_labels: pd.DataFrame,
                                                noisy_true_labels: pd.DataFrame,
                                                n_workers: int) -> pd.DataFrame:
        """
        Calculate worker weights based on their accuracy relative to actual labels.

        This function evaluates how well each worker's labels align with the noisy true labels
        and computes weights that give more importance to workers who are more accurate.

        Parameters
        ----------
        workers_labels : pd.DataFrame
            DataFrame where each row represents a sample and each column represents a worker's labels.

        noisy_true_labels : pd.DataFrame
            DataFrame containing the noisy ground truth labels for each sample.

        n_workers : int
            Number of workers.

        Returns
        -------
        pd.DataFrame
            Normalized weights for each worker for each sample.
        """
        tau = (workers_labels == noisy_true_labels).mean(axis=0)

        # Number of workers
        M = len(noisy_true_labels.columns)

        # Number of true and false labels for each class and sample
        true_counts = noisy_true_labels.sum(axis=1)
        false_counts = M - true_counts

        # Measuring the "specific quality of instances"
        s = noisy_true_labels.multiply(true_counts - 1, axis=0) + (~noisy_true_labels).multiply(false_counts - 1, axis=0)
        gamma = (1 + s ** 2) * tau
        W_hat_Tao = gamma.apply(lambda x: 1 / (1 + np.exp(-x)))
        z = W_hat_Tao.mean(axis=1)

        return W_hat_Tao.divide(z, axis=0) / n_workers

    @staticmethod
    def calculate_sheng_weights(shape: tuple, n_workers: int) -> pd.DataFrame:
        """
        Calculate equal weights for the SHENG method.

        Parameters
        ----------
        shape : tuple
            Shape to match for the weights DataFrame (rows, cols)
        n_workers : int
            Number of workers

        Returns
        -------
        pd.DataFrame
            DataFrame with equal weights (1/n_workers) for all workers
        """
        index, columns = shape
        return pd.DataFrame(1 / n_workers, index=index, columns=columns)

    def get_weights(self, workers_labels: pd.DataFrame, preds: pd.DataFrame,
                   uncertainties: pd.DataFrame, noisy_true_labels: pd.DataFrame,
                   n_workers: int) -> params.WeightType:
        """
        Calculate weights for different methods (proposed, TAO, and SHENG).

        Parameters
        ----------
        workers_labels : pd.DataFrame
            Matrix of labels assigned by workers to items.
        preds : pd.DataFrame
            Predictions (estimated true labels).
        uncertainties : pd.DataFrame
            Uncertainty values associated with predictions.
        noisy_true_labels : pd.DataFrame
            Ground truth labels (possibly with noise).
        n_workers : int
            Number of workers who provided labels.

        Returns
        -------
        params.WeightType
            Named tuple containing weights for three methods:
            - PROPOSED: Weights calculated using the proposed technique
            - TAO: Weights calculated based on Tao's method
            - SHENG: Equal weights (1/n_workers) for all worker-item pairs
        """
        # Measuring weights for the proposed technique
        weights_proposed = self.calculate_proposed_weights(preds=preds, uncertainties=uncertainties)

        # Benchmark accuracy measurement
        weights_Tao = self.calculate_tao_weights_with_actual_labels(
            workers_labels=workers_labels,
            noisy_true_labels=noisy_true_labels,
            n_workers=n_workers
        )

        # Equal weights for SHENG method
        weights_Sheng = self.calculate_sheng_weights(
            shape=(weights_Tao.index, weights_Tao.columns),
            n_workers=n_workers
        )

        return params.WeightType(PROPOSED=weights_proposed, TAO=weights_Tao, SHENG=weights_Sheng)
