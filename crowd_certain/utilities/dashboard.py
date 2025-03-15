"""
Crowd-Certain Dashboard

A Streamlit-based dashboard for visualizing and interacting with the Crowd-Certain framework.
This dashboard allows users to run simulations, analyze results, and explore the performance
of different crowd-sourced label aggregation techniques.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import directly from the utilities module since we're now in the same package
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities import params
from crowd_certain.utilities.settings import Settings, get_settings, find_config_file, revert_to_default_config
from crowd_certain.utilities.utils import ResultComparisonsType
from crowd_certain.utilities.dataset_loader import find_dataset_path

class DashboardStyles:
    """Manages CSS styles for the dashboard."""

    @staticmethod
    def apply_styles() -> None:
        """Apply custom CSS styles to the dashboard."""
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #4B8BBE;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: bold;
                color: #306998;
                margin-bottom: 0.5rem;
            }
            .info-box {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .result-box {
                background-color: #e6f3ff;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)


class SidebarConfig:
    """Manages the sidebar configuration options."""

    def __init__(self):
        """Initialize the sidebar configuration from existing config if available."""
        # Load existing configuration if available
        try:
            self.config = get_settings()
            # print("Loaded existing configuration from config.json")

        except Exception as e:
            print(f"Could not load existing configuration: {str(e)}. Using default values.")
            self.config = Settings(
                dataset=dict(
                    dataset_name = params.DatasetNames.IONOSPHERE,
                    datasetNames = [params.DatasetNames.IONOSPHERE],
                ),
                simulation=dict(
                    low_dis   = 0.4,
                    high_dis  = 1.0,
                    num_seeds = 3,
                    n_workers_min_max    = [3, 8],
                    use_parallelization  = True,
                    max_parallel_workers = min(4, os.cpu_count() or 4)
                ),
                technique=dict(
                    uncertainty_techniques = [params.UncertaintyTechniques.STD],
                    consistency_techniques = [params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
                ),
                output=dict(
                    mode=params.OutputModes.CALCULATE,
                    save=False,
                )
            )

        # Setup options dictionaries
        self.dataset_options     = {name.value: name for name in params.DatasetNames}
        self.uncertainty_options = {tech.value: tech for tech in params.UncertaintyTechniques}
        self.consistency_options = {tech.value: tech for tech in params.ConsistencyTechniques}

        # Set initial values from config
        self.selected_dataset_values = [str(ds) for ds in self.config.dataset.datasetNames]

        # Get simulation parameters
        self.n_workers_min = self.config.simulation.n_workers_min_max[0]
        self.n_workers_max = self.config.simulation.n_workers_min_max[1]
        self.low_quality   = self.config.simulation.low_dis
        self.high_quality  = self.config.simulation.high_dis
        self.num_seeds     = self.config.simulation.num_seeds

        # Get parallelization settings
        self.use_parallelization  = getattr(self.config.simulation, 'use_parallelization', True)
        self.max_parallel_workers = getattr(self.config.simulation, 'max_parallel_workers', min(4, os.cpu_count() or 4))

        # Get technique settings
        self.selected_uncertainty_values = [tech.value for tech in self.config.technique.uncertainty_techniques]
        self.selected_consistency_values = [tech.value for tech in self.config.technique.consistency_techniques]

        # Get output settings
        self.auto_download = True  # Default to True for better user experience

        # Get the correct dataset path using the shared function
        self.dataset_path = find_dataset_path(getattr(self.config.dataset, 'path_all_datasets', Path(__file__).parent.parent / 'datasets'))

    def save_config(self, config_path=None):
        """Save the current configuration to a file."""
        # Save the config with current UI values
        config = self.update_config()

        # Determine the file path
        if config_path is None:
            config_path = find_config_file(config_path=config_path)

        # Save the configuration
        try:
            config.save(config_path)
            return True, f"Configuration saved to {config_path}"

        except Exception as e:
            return False, f"Error saving configuration: {str(e)}"

    def revert_to_default_config(self):
        """Revert to the default configuration."""
        try:
            # Use the revert_to_default_config function
            success, config_path = revert_to_default_config()

            if not success:
                return False, f"Failed to revert to default configuration"

            # Reload the configuration
            self.config = get_settings()

            # Update UI values from the new config
            self.selected_dataset_values = [str(ds) for ds in self.config.dataset.datasetNames]
            self.n_workers_min           = self.config.simulation.n_workers_min_max[0]
            self.n_workers_max           = self.config.simulation.n_workers_min_max[1]
            self.low_quality             = self.config.simulation.low_dis
            self.high_quality            = self.config.simulation.high_dis
            self.num_seeds               = self.config.simulation.num_seeds
            self.use_parallelization  = getattr(self.config.simulation, 'use_parallelization', True)
            self.max_parallel_workers = getattr(self.config.simulation, 'max_parallel_workers', min(4, os.cpu_count() or 4))
            self.selected_uncertainty_values = [tech.value for tech in self.config.technique.uncertainty_techniques]
            self.selected_consistency_values = [tech.value for tech in self.config.technique.consistency_techniques]

            return True, f"Configuration reverted to default"
        except Exception as e:
            return False, f"Error reverting to default configuration: {str(e)}"

    def render(self) -> None:
        """Render the sidebar configuration options."""

        def get_number_of_workers():
            # Number of workers - now with text input
            st.sidebar.markdown("##### Number of Workers", help="Minimum and maximum number of workers in the simulation")

            col1, col2 = st.sidebar.columns(2)
            # Minimum number of workers
            with col1:
                self.n_workers_min = st.number_input(
                    "Minimum",
                    value=self.n_workers_min,
                    min_value=3,
                    max_value=10,
                    step=1,
                )

            # Maximum number of workers
            with col2:
                self.n_workers_max = st.number_input(
                    "Maximum",
                    value=self.n_workers_max,
                    min_value=self.n_workers_min,
                    max_value=20,
                    step=1,
                )

        def get_worker_quality_range():
            st.sidebar.markdown("##### Worker Quality Range", help="Minimum and maximum worker quality (0: Miles Morales   .  1: Always Correct)")
            worker_quality_col1, worker_quality_col2 = st.sidebar.columns(2)
            with worker_quality_col1:
                self.low_quality = st.number_input(
                    "Minimum",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.low_quality,
                    step=0.05,
                )
            with worker_quality_col2:
                self.high_quality = st.number_input(
                    "Maximum",
                    min_value=self.low_quality,
                    max_value=1.0,
                    value=self.high_quality,
                    step=0.05,
                )

        def selecting_datasets():
            # Dataset selection - now with multi-select
            self.selected_dataset_values = st.sidebar.multiselect(
                "Select Datasets",
                options=list(self.dataset_options.keys()),
                default=self.selected_dataset_values,
                help="Choose one or more datasets to run the simulation on"
            )


            # Advanced options
            with st.sidebar.expander("Advanced Options"):
                # Dataset options
                st.markdown("##### Dataset Settings")
                self.auto_download = st.checkbox(
                    "Auto-download Datasets",
                    value=self.auto_download,
                    help="Automatically download datasets if they are not found locally"
                )

                # Add option to manually specify dataset path
                use_custom_path = st.checkbox(
                    "Use Custom Dataset Path",
                    value=False,
                    help="Specify a custom path to the datasets directory"
                )

                if use_custom_path:
                    custom_path = st.text_input(
                        "Custom Dataset Path",
                        value=str(self.dataset_path),
                        help="Full path to the directory containing the datasets"
                    )
                    if custom_path:
                        self.dataset_path = Path(custom_path)
                        st.info(f"Using custom dataset path: {self.dataset_path}")

                        # Check if path exists
                        if not self.dataset_path.exists():
                            st.warning(f"Warning: The specified path does not exist. It will be created when needed.")
                        else:
                            # List available datasets at this path
                            available_datasets = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
                            if available_datasets:
                                st.success(f"Found dataset directories: {', '.join(available_datasets)}")
                            else:
                                st.warning("No dataset directories found at this path.")


        st.sidebar.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

        # Dataset selection
        selecting_datasets()
        st.sidebar.markdown("---")


        # Number of workers input
        get_number_of_workers()
        st.sidebar.markdown("---")

        # Worker quality range input
        get_worker_quality_range()
        st.sidebar.markdown("---")

        # Number of seeds input
        self.num_seeds = st.sidebar.number_input(
            "Number of Random Seeds",
            value=self.num_seeds,
            min_value=1,
            max_value=10,
            step=1,
            help="Number of random seeds to use for the simulation (must be an integer ≥ 1)"
        )

        # Uncertainty techniques selection
        with st.sidebar.expander("Uncertainty Techniques", expanded=False):
            # Use list comprehension to create checkboxes and gather selected values
            self.selected_uncertainty_values = [
            uncertainty_technique for uncertainty_technique in self.uncertainty_options
            if st.checkbox(
                uncertainty_technique,
                value=uncertainty_technique in self.selected_uncertainty_values,
                key=f"uncertainty_{uncertainty_technique}"
            )
            ]

            # Ensure at least one technique is selected
            if not self.selected_uncertainty_values:
                self.selected_uncertainty_values = [params.UncertaintyTechniques.STD.value]
                st.warning("At least one uncertainty technique must be selected")


        # Consistency techniques selection
        with st.sidebar.expander("Consistency Techniques", expanded=False):
            # Use list comprehension for consistency techniques
            self.selected_consistency_values = [
            consistency_technique for consistency_technique in self.consistency_options
            if st.checkbox(
                consistency_technique,
                value=consistency_technique in self.selected_consistency_values,
                key=f"consistency_{consistency_technique}"
            )
            ]

            # Ensure at least one technique is selected
            if not self.selected_consistency_values:
                self.selected_consistency_values = [params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY.value]
                st.warning("At least one consistency technique must be selected")

        # Advanced options
        with st.sidebar.expander("Parallelization Settings"):
            self.use_parallelization = st.checkbox(
                "Enable Parallelization",
                value=self.use_parallelization,
                help="Use parallel processing to speed up simulations (recommended for multiple datasets or workers)"
            )

            if self.use_parallelization:
                self.max_parallel_workers = st.number_input(
                    "Maximum Parallel Workers",
                    min_value=1,
                    max_value=os.cpu_count() or 4,
                    value=self.max_parallel_workers,
                    step=1,
                    help=f"Maximum number of parallel processes to use (your system has {os.cpu_count() or 'unknown'} CPU cores)"
                )
            else:
                self.max_parallel_workers = 1

        # Update the config with values from the UI
        self.update_config()

        # Configuration management buttons
        st.sidebar.markdown("---")
        st.sidebar.markdown("##### Configuration Management")

        # Create two columns for the buttons
        col1, col2 = st.sidebar.columns(2)

        # Save Configuration button
        with col1:
            if st.button("Save Config", type="primary"):
                success, message = self.save_config()
                if success:
                    st.success(message)
                else:
                    st.error(message)

        # Revert to Default button
        with col2:
            if st.button("Revert to Default", type="secondary"):
                success, message = self.revert_to_default_config()
                if success:
                    st.success(message)
                else:
                    st.error(message)

    def get_selected_datasets(self) -> List[params.DatasetNames]:
        """Get the selected datasets as DatasetNames enums."""
        return [self.dataset_options[val] for val in self.selected_dataset_values]

    def get_selected_uncertainty(self) -> List[params.UncertaintyTechniques]:
        """Get the selected uncertainty techniques as UncertaintyTechniques enums."""
        return [self.uncertainty_options[val] for val in self.selected_uncertainty_values]

    def get_selected_consistency(self) -> List[params.ConsistencyTechniques]:
        """Get the selected consistency techniques as ConsistencyTechniques enums."""
        return [self.consistency_options[val] for val in self.selected_consistency_values]

    def update_config(self) -> Settings:
        """Update the existing config with values from the UI and return it."""
        # Update dataset settings
        self.config.dataset.dataset_name      = self.get_selected_datasets()[0]  # Use first dataset as primary
        self.config.dataset.datasetNames      = self.get_selected_datasets()     # All selected datasets
        self.config.dataset.path_all_datasets = self.dataset_path

        # Update simulation settings
        self.config.simulation.n_workers_min_max    = [self.n_workers_min, self.n_workers_max]
        self.config.simulation.low_dis              = self.low_quality
        self.config.simulation.high_dis             = self.high_quality
        self.config.simulation.num_seeds            = self.num_seeds
        self.config.simulation.use_parallelization  = self.use_parallelization
        self.config.simulation.max_parallel_workers = self.max_parallel_workers

        # Update technique settings
        self.config.technique.uncertainty_techniques = self.get_selected_uncertainty()
        self.config.technique.consistency_techniques = self.get_selected_consistency()

        # Update output settings
        self.config.output.mode = params.OutputModes.CALCULATE
        self.config.output.save = False  # Don't save results by default in the dashboard

        return self.config


class SimulationRunner:
    """Handles running simulations and displaying results."""

    def __init__(self, config: SidebarConfig):
        """Initialize the simulation runner with configuration."""
        self.config = config

    def run_simulation(self) -> None:
        """Run the simulation with the current configuration for all selected datasets."""
        # Check if any datasets are selected
        if not self.config.selected_dataset_values:
            st.error("Please select at least one dataset to run the simulation.")
            return

        # Initialize results dictionary in session state if it doesn't exist
        if 'results_by_dataset' not in st.session_state:
            st.session_state.results_by_dataset = {}

        # Create base configuration
        settings = self.config.update_config()

        # Track overall progress
        progress_bar = st.progress(0)
        total_datasets = len(self.config.selected_dataset_values)

        # Run simulation for each selected dataset
        for i, dataset_value in enumerate(self.config.selected_dataset_values):
            dataset_name = self.config.dataset_options[dataset_value]

            with st.spinner(f"Running simulation for dataset: {dataset_value} ({i+1}/{total_datasets})..."):
                # Update settings for this specific dataset
                settings.dataset.dataset_name = dataset_name

                # Check if dataset exists and show appropriate message
                dataset_dir = settings.dataset.path_all_datasets / dataset_value
                dataset_file = dataset_dir / f"{dataset_value}.csv"

                if not dataset_file.exists():
                    if self.config.auto_download:
                        st.info(f"Dataset {dataset_value} not found locally. Will attempt to download...")
                    else:
                        st.warning(f"Dataset {dataset_value} not found locally and auto-download is disabled. The simulation may fail.")

                # Run the simulation for this dataset
                try:
                    results: ResultComparisonsType = AIM1_3.calculate_one_dataset(config=settings, dataset_name=dataset_name)

                    # Store results for this dataset
                    st.session_state.results_by_dataset[dataset_value] = results

                    st.success(f"Simulation completed successfully for {dataset_value} dataset!")

                except Exception as e:
                    st.error(f"Error running simulation for {dataset_value}: {str(e)}")
                    st.exception(e)

                    # Provide helpful message for common errors
                    error_str = str(e)
                    if "Could not load dataset" in error_str:
                        st.warning(
                            "The dataset could not be found or downloaded. Please check your internet connection "
                            "and try again. If the problem persists, you can manually download the dataset from "
                            "the UCI Machine Learning Repository and place it in the appropriate directory: "
                            f"{settings.dataset.path_all_datasets / dataset_value}"
                        )

            # Update progress bar
            progress_bar.progress((i + 1) / total_datasets)

        # Set the last dataset as the current result for backward compatibility
        if self.config.selected_dataset_values and self.config.selected_dataset_values[-1] in st.session_state.results_by_dataset:
            st.session_state.results = st.session_state.results_by_dataset[self.config.selected_dataset_values[-1]]

        # Final success message
        if len(self.config.selected_dataset_values) > 1:
            st.success(f"Completed simulations for {len(self.config.selected_dataset_values)} datasets!")


class ResultsTab:
    """Manages the Results tab content."""

    @staticmethod
    def render() -> None:
        """Render the Results tab content."""
        st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)

        # Check if any results are available
        if 'results_by_dataset' not in st.session_state or not st.session_state.results_by_dataset:
            st.info("No simulation results available. Please run a simulation first.")
            return

        # Get available datasets
        available_datasets = list(st.session_state.results_by_dataset.keys())

        # Create tabs for each dataset
        dataset_tabs = st.tabs(available_datasets)

        # Display results for each dataset in its respective tab
        for i, dataset_name in enumerate(available_datasets):
            with dataset_tabs[i]:
                # Get the results for this dataset
                results = st.session_state.results_by_dataset[dataset_name]

                # Display metrics
                st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)

                # Get the first result to display metrics
                first_nl_key = list(results.outputs.keys())[0]
                first_result = results.outputs[first_nl_key][0]

                # Display metrics for proposed methods and benchmarks
                st.markdown("#### Proposed Methods")
                proposed_metrics = first_result.proposed.metrics
                st.dataframe(proposed_metrics)

                st.markdown("#### Benchmark Methods")
                benchmark_metrics = first_result.benchmark.metrics
                st.dataframe(benchmark_metrics)

                # Plot worker strength vs weight relationship
                ResultsTab._plot_worker_strength_weight_relation(results)

                # Display confidence scores
                ResultsTab._display_confidence_scores(first_result)

    @staticmethod
    def _plot_worker_strength_weight_relation(results) -> None:
        """Plot the worker strength vs weight relationship."""
        st.markdown('<div class="sub-header">Worker Strength vs Weight Relationship</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        weight_strength_df = results.weight_strength_relation

        # Plot the data
        sns.scatterplot(data=weight_strength_df, x=weight_strength_df.index, y=weight_strength_df.columns[0], ax=ax, label=weight_strength_df.columns[0])

        # Add more columns if they exist
        for col in weight_strength_df.columns[1:]:
            sns.scatterplot(data=weight_strength_df, x=weight_strength_df.index, y=col, ax=ax, label=col)

        ax.set_xlabel("Worker Strength")
        ax.set_ylabel("Weight")
        ax.set_title("Worker Strength vs Weight Relationship")
        ax.legend()

        st.pyplot(fig)

    @staticmethod
    def _display_confidence_scores(first_result) -> None:
        """Display confidence scores for the first result."""
        st.markdown('<div class="sub-header">Confidence Scores</div>', unsafe_allow_html=True)

        # Get confidence scores for the first result
        confidence_scores = first_result.proposed.confidence_scores

        # Create tabs for each confidence score method
        if confidence_scores:
            # Add debug information
            st.write("Confidence score keys types:", [type(key).__name__ for key in confidence_scores.keys()])

            # Convert keys to strings to avoid StreamlitAPIException
            confidence_tabs = st.tabs([str(key) for key in confidence_scores.keys()])

            for i, (method, scores) in enumerate(confidence_scores.items()):
                with confidence_tabs[i]:
                    st.dataframe(scores)

                    # Plot histogram of confidence scores
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Get the F scores (first level)
                    f_scores = scores.iloc[:, scores.columns.get_level_values(0) == 'F']

                    # Plot histogram
                    for col in f_scores.columns:
                        sns.histplot(f_scores[col], kde=True, ax=ax, label=col)

                    ax.set_xlabel("Confidence Score")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of Confidence Scores for {method}")
                    ax.legend()

                    st.pyplot(fig)


class WorkerAnalysisTab:
    """Manages the Worker Analysis tab content."""

    @staticmethod
    def render() -> None:
        """Render the Worker Analysis tab content."""
        st.markdown('<div class="sub-header">Worker Analysis</div>', unsafe_allow_html=True)

        # Check if any results are available
        if 'results_by_dataset' not in st.session_state or not st.session_state.results_by_dataset:
            st.info("No simulation results available. Please run a simulation first.")
            return

        # Get available datasets
        available_datasets = list(st.session_state.results_by_dataset.keys())

        # Create tabs for each dataset
        dataset_tabs = st.tabs(available_datasets)

        # Display worker analysis for each dataset in its respective tab
        for i, dataset_name in enumerate(available_datasets):
            with dataset_tabs[i]:
                # Get the results for this dataset
                results = st.session_state.results_by_dataset[dataset_name]

                # Get the first result to display worker information
                first_nl_key = list(results.outputs.keys())[0]
                first_result = results.outputs[first_nl_key][0]

                # Display worker strength information
                WorkerAnalysisTab._display_worker_strength(first_result)

                # Display worker weights
                WorkerAnalysisTab._display_worker_weights(first_result)

    @staticmethod
    def _display_worker_strength(first_result) -> None:
        """Display worker strength information."""
        st.markdown("#### Worker Strength")
        worker_strength = first_result.workers_strength
        st.dataframe(worker_strength)

        # Plot worker strength distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(worker_strength['workers_strength'], kde=True, ax=ax)
        ax.set_xlabel("Worker Strength")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Worker Strength")
        st.pyplot(fig)

    @staticmethod
    def _display_worker_weights(first_result) -> None:
        """Display worker weights for different methods."""
        st.markdown("#### Worker Weights")

        # Create tabs for different weight methods
        weight_tabs = st.tabs(["Proposed", "TAO", "SHENG"])

        with weight_tabs[0]:
            st.dataframe(first_result.weight.PROPOSED)

            # Plot heatmap of weights
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(first_result.weight.PROPOSED, cmap="viridis", ax=ax)
            ax.set_title("Proposed Method Weights")
            st.pyplot(fig)

        with weight_tabs[1]:
            st.dataframe(first_result.weight.TAO)

            # Plot heatmap of weights
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(first_result.weight.TAO, cmap="viridis", ax=ax)
            ax.set_title("TAO Method Weights")
            st.pyplot(fig)

        with weight_tabs[2]:
            st.dataframe(first_result.weight.SHENG)

            # Plot heatmap of weights
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(first_result.weight.SHENG, cmap="viridis", ax=ax)
            ax.set_title("SHENG Method Weights")
            st.pyplot(fig)


class AboutTab:
    """Manages the About tab content."""

    @staticmethod
    def render() -> None:
        """Render the About tab content."""
        st.markdown('<div class="sub-header">About Crowd-Certain</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <p>Crowd-Certain is a comprehensive framework for aggregating labels from multiple annotators (crowd workers)
        while estimating the uncertainty and confidence in the aggregated labels.</p>

        <p>The framework implements methods for:</p>
        <ul>
            <li>Calculating uncertainties using different techniques (standard deviation, entropy, coefficient of variation, etc.)</li>
            <li>Converting uncertainties to consistency scores</li>
            <li>Calculating weights for proposed techniques and benchmark methods</li>
            <li>Measuring accuracy metrics for different aggregation techniques</li>
            <li>Simulating worker strengths and noisy labels</li>
            <li>Generating confidence scores for aggregated labels</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        AboutTab._display_key_features()
        AboutTab._display_benchmark_methods()
        AboutTab._display_dataset_structure()
        AboutTab._display_contact()

    @staticmethod
    def _display_key_features() -> None:
        """Display key features of the framework."""
        st.markdown('<div class="sub-header">Key Features</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Uncertainty Techniques
            - Standard Deviation
            - Entropy
            - Coefficient of Variation
            - Prediction Interval
            - Confidence Interval
            """)

        with col2:
            st.markdown("""
            #### Consistency Techniques
            - One Minus Uncertainty
            - One Divided By Uncertainty
            """)

    @staticmethod
    def _display_benchmark_methods() -> None:
        """Display benchmark methods information."""
        st.markdown('<div class="sub-header">Benchmark Methods</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Main Benchmarks
            - Tao
            - Sheng
            """)

        with col2:
            st.markdown("""
            #### Other Benchmarks
            - KOS
            - MACE
            - MMSR
            - Wawa
            - ZeroBasedSkill
            - MajorityVote
            - DawidSkene
            """)

    @staticmethod
    def _display_dataset_structure() -> None:
        """Display information about the dataset structure."""
        st.markdown('<div class="sub-header">Dataset Structure</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <p>Datasets are stored in a standardized structure:</p>
        <pre>
        datasets/
        ├── ionosphere/
        │   └── ionosphere.csv
        ├── chess/
        │   └── chess.csv
        ├── mushroom/
        │   └── mushroom.csv
        └── ...
        </pre>

        <p>Each dataset is stored in its own directory with the same name as the dataset.
        Within each directory, the dataset file is named using the same pattern: <code>{dataset_name}.csv</code>.</p>

        <p>When you run a simulation, the system will:</p>
        <ol>
            <li>Look for the dataset in the local cache using the structure above</li>
            <li>If not found, download it from the UCI Machine Learning Repository</li>
            <li>Process and save it in the standardized format for future use</li>
        </ol>

        <p>You can specify a custom dataset path in the Advanced Options section if needed.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _display_contact() -> None:
        """Display contact information."""
        st.markdown('<div class="sub-header">Contact</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <p>Artin Majdi - msm2024@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)


class CrowdCertainDashboard:
    """Main dashboard class that orchestrates the entire application."""

    def __init__(self):
        """Initialize the dashboard."""
        # Set page configuration
        st.set_page_config(
            page_title="Crowd-Certain Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Apply styles
        DashboardStyles.apply_styles()

        # Initialize components
        self.sidebar_config = SidebarConfig()
        self.simulation_runner = SimulationRunner(self.sidebar_config)

    def run(self) -> None:
        """Run the dashboard application."""
        # Display header
        self._display_header()

        # Render sidebar
        self.sidebar_config.render()

        # Add run button
        run_simulation = st.sidebar.button("Run Simulation", type="primary")

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Simulation Results", "Worker Analysis", "About"])

        # Run simulation if button is clicked
        if run_simulation:
            self.simulation_runner.run_simulation()

        # Render tab contents
        with tab1:
            ResultsTab.render()

        with tab2:
            WorkerAnalysisTab.render()

        with tab3:
            AboutTab.render()

    def _display_header(self) -> None:
        """Display the dashboard header."""
        st.markdown('<div class="main-header">Crowd-Certain: Crowd-Sourced Label Aggregation</div>', unsafe_allow_html=True)
        st.markdown("""
        Crowd-Certain is a comprehensive framework for aggregating labels from multiple annotators (crowd workers)
        while estimating the uncertainty and confidence in the aggregated labels.
        """)


# Entry point for the application
if __name__ == "__main__":
    dashboard = CrowdCertainDashboard()
    dashboard.run()
