"""
Metrics module for crowd-certain.

This module provides functions and classes for calculating evaluation metrics
for aggregated labels from crowd workers.
"""

import pandas as pd
import numpy as np
from sklearn import metrics as sk_metrics

from crowd_certain.utilities.config import params

class MetricsCalculator:
    """
    A class for calculating various evaluation metrics for crowd-sourcing techniques.
    """

    @staticmethod
    def get_accuracy(aggregated_labels: pd.DataFrame, n_workers: int,
                     delta_benchmark: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
        """
        Calculates the accuracy of various crowdsourcing aggregation methods.

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
        """
        accuracy = pd.DataFrame(index=[n_workers])

        for methods in [params.ProposedTechniqueNames, params.MainBenchmarks, params.OtherBenchmarkNames]:
            for m in methods:
                accuracy[m] = ((aggregated_labels[m] >= 0.5) == truth).mean(axis=0)

        accuracy['MV_Classifier'] = ((delta_benchmark.mean(axis=1) >= 0.5) == truth).mean(axis=0)

        return accuracy

    @staticmethod
    def get_AUC_ACC_F1(aggregated_labels: pd.Series, truth: pd.Series) -> pd.Series:
        """
        Calculate AUC, accuracy, and F1 score metrics between aggregated labels and ground truth.

        Parameters
        ----------
        aggregated_labels : pd.Series
            The aggregated (predicted) probability labels, typically between 0 and 1.

        truth : pd.Series
            The ground truth labels with the same index as aggregated_labels.
            Can contain null values which will be filtered out.

        Returns
        -------
        pd.Series
            A pandas Series containing the following metrics:
            - AUC (Area Under the ROC Curve)
            - Accuracy
            - F1 score
        """
        metrics = pd.Series(index=params.EvaluationMetricNames.values())

        # Filter out null values from truth
        non_null = ~truth.isnull()
        truth_notnull = truth[non_null].to_numpy()

        # Only calculate metrics if there are valid values and truth is binary
        if (len(truth_notnull) > 0) and (np.unique(truth_notnull).size == 2):
            # Convert aggregated labels to binary predictions
            yhat = (aggregated_labels > 0.5).astype(int)[non_null]

            # Calculate AUC, accuracy, and F1 score
            metrics[params.EvaluationMetricNames.AUC.value] = sk_metrics.roc_auc_score(truth_notnull, yhat)
            metrics[params.EvaluationMetricNames.ACC.value] = sk_metrics.accuracy_score(truth_notnull, yhat)
            metrics[params.EvaluationMetricNames.F1.value] = sk_metrics.f1_score(truth_notnull, yhat)

        return metrics
