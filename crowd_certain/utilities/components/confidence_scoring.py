"""
Confidence scoring module for crowd-certain.

This module contains functions and classes for calculating confidence scores
for aggregated labels from multiple crowd workers.
"""

import numpy as np
import pandas as pd
from scipy.special import bdtrc

from crowd_certain.utilities.config import params

class ConfidenceScorer:
    """
    A class for calculating confidence scores using different strategies.
    """

    @staticmethod
    def calculate_confidence_scores(delta: pd.DataFrame, w: pd.DataFrame, n_workers: int) -> pd.DataFrame:
        """
        Calculate confidence scores for each item using multiple strategies.

        This function computes confidence scores for binary classifications based on worker responses
        and their weights. It implements two strategies: frequency-based and beta distribution-based.

        Parameters
        ----------
        delta : pd.DataFrame
            Binary matrix of worker responses where rows are items and columns are workers
        w : pd.DataFrame
            Worker weights used to compute weighted sum of responses
        n_workers : int
            Number of workers who provided classifications

        Returns
        -------
        pd.DataFrame
            A DataFrame with multi-level columns containing confidence scores:
            - First level: Strategy names (FREQ, BETA)
            - Second level: 'F' (overall confidence score) and 'F_pos' (confidence in positive class)
        """
        # Calculate weighted positive and negative probabilities
        P_pos = (delta * w).sum(axis=1)
        P_neg = (~delta * w).sum(axis=1)

        # Get confidence scores using frequency strategy
        freq_scores = ConfidenceScorer._get_freq_scores(P_pos, P_neg)

        # Get confidence scores using beta strategy
        beta_scores = ConfidenceScorer._get_beta_scores(P_pos, P_neg, n_workers)

        # Create output DataFrame with MultiIndex columns
        columns = pd.MultiIndex.from_product(
            [params.StrategyNames.values(), ['F', 'F_pos']],
            names=['strategies', 'F_F_pos']
        )

        confidence_scores = pd.DataFrame(columns=columns, index=delta.index)
        confidence_scores[params.StrategyNames.FREQ.value] = freq_scores
        confidence_scores[params.StrategyNames.BETA.value] = beta_scores

        return confidence_scores

    @staticmethod
    def _get_freq_scores(P_pos: pd.Series, P_neg: pd.Series) -> pd.DataFrame:
        """
        Calculate frequency-based confidence scores.

        Parameters
        ----------
        P_pos : pd.Series
            Weighted sum of positive responses
        P_neg : pd.Series
            Weighted sum of negative responses

        Returns
        -------
        pd.DataFrame
            DataFrame with 'F' and 'F_pos' columns
        """
        out = pd.DataFrame({'P_pos': P_pos, 'P_neg': P_neg})
        return pd.DataFrame({'F': out.max(axis=1), 'F_pos': P_pos})

    @staticmethod
    def _get_beta_scores(P_pos: pd.Series, P_neg: pd.Series, n_workers: int) -> pd.DataFrame:
        """
        Calculate beta distribution-based confidence scores.

        Parameters
        ----------
        P_pos : pd.Series
            Weighted sum of positive responses
        P_neg : pd.Series
            Weighted sum of negative responses
        n_workers : int
            Number of workers

        Returns
        -------
        pd.DataFrame
            DataFrame with 'F' and 'F_pos' columns
        """
        out = pd.DataFrame({'P_pos': P_pos, 'P_neg': P_neg})

        # Calculate parameters for beta distribution
        out['l_alpha'] = 1 + out['P_pos'] * n_workers
        out['u_beta'] = 1 + out['P_neg'] * n_workers
        out['k'] = out['l_alpha'] - 1
        out['n'] = ((out['l_alpha'] + out['u_beta']) - 1)

        # Calculate incomplete beta function
        out['I'] = out.apply(lambda row: bdtrc(row['k'], row['n'], 0.5), axis=1)

        # Return confidence scores
        return pd.DataFrame({
            'F': out.apply(lambda row: max(row['I'], 1-row['I']), axis=1),
            'F_pos': out['I']
        })
