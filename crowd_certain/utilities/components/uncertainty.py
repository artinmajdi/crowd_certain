"""
Uncertainty calculation module for crowd-certain.

This module contains functions and classes for calculating various uncertainty metrics
for predictions, including standard deviation, entropy, coefficient of variation,
prediction interval, and confidence interval.
"""

import numpy as np
import pandas as pd
import scipy

from crowd_certain.utilities import params

class UncertaintyCalculator:
    """
    A class for calculating various uncertainty metrics for predictions.
    """
    def __init__(self, config):
        """
        Initialize the UncertaintyCalculator with configuration.

        Parameters
        ----------
        config : Settings
            Configuration object containing uncertainty techniques to use
        """
        self.config = config

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

        df_uncertainties = pd.DataFrame(columns=[l.value for l in self.config.technique.uncertainty_techniques], index=df.index)

        for tech in self.config.technique.uncertainty_techniques:

            if tech is params.UncertaintyTechniques.STD:
                df_uncertainties[tech.value] = df.std(axis=1)

            elif tech is params.UncertaintyTechniques.ENTROPY:
                # Normalize each row to sum to 1
                df_normalized = df.div(df.sum(axis=1) + epsilon, axis=0)
                # Calculate entropy
                entropy = -(df_normalized * np.log(df_normalized + epsilon)).sum(axis=1)

                # normalizing entropy values to be between 0 and 1
                df_uncertainties[tech.value] = entropy / np.log(df.shape[1])

            elif tech is params.UncertaintyTechniques.CV:
                # The coefficient of variation (CoV) is a measure of relative variability.
                # Normalizing using hyperbolic tangent to bound between 0 and 1
                coefficient_of_variation = df.std(axis=1) / (df.mean(axis=1) + epsilon)
                df_uncertainties[tech.value] = np.tanh(coefficient_of_variation)

            elif tech is params.UncertaintyTechniques.PI:
                df_uncertainties[tech.value] = df.apply(lambda row: np.percentile(row.astype(int), 75) - np.percentile(row.astype(int), 25), axis=1)

            elif tech is params.UncertaintyTechniques.CI:
                # Calculate confidence interval
                confidence_interval = df.apply(
                    lambda row: scipy.stats.norm.interval(0.95, loc=np.mean(row), scale=scipy.stats.sem(row))
                    if np.std(row) > 0 else (np.nan, np.nan),
                    axis=1
                ).apply(pd.Series)

                # Calculate the width of the confidence interval (uncertainty score)
                width = confidence_interval[1] - confidence_interval[0]

                df_uncertainties[tech.value] = width.fillna(0)

        return df_uncertainties

    def calculate_consistency(self, uncertainty: pd.DataFrame) -> pd.DataFrame:
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
            - If input was DataFrame: Returns multi-level columned DataFrame with consistency techniques as the second level
            - If input was Series/ndarray: Returns DataFrame with consistency techniques as columns
        """
        from collections import OrderedDict

        def initialize_consistency():
            nonlocal consistency
            upper_level = [l.value for l in self.config.technique.consistency_techniques]

            if isinstance(uncertainty, pd.DataFrame):
                # The use of OrderedDict helps preserving the order of columns.
                levels = [list(OrderedDict.fromkeys(uncertainty.columns.get_level_values(i))) for i in range(uncertainty.columns.nlevels)]

                new_columns      = [upper_level] + levels
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


# Function aliases for backward compatibility and convenience
def calculate_uncertainties(df: pd.DataFrame, config) -> pd.DataFrame:
    """Convenience function for calculating uncertainties"""
    calculator = UncertaintyCalculator(config)
    return calculator.calculate_uncertainties(df)


def calculate_consistency(uncertainty, config) -> pd.DataFrame:
    """Convenience function for calculating consistency"""
    calculator = UncertaintyCalculator(config)
    return calculator.calculate_consistency(uncertainty)
