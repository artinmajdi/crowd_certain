"""
Crowd-Certain Dashboard

A Streamlit-based dashboard for visualizing and interacting with the Crowd-Certain framework.
This dashboard allows users to run simulations, analyze results, and explore the performance
of different crowd-sourced label aggregation techniques.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import directly from the utilities module since we're now in the same package
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques, ReadMode
from crowd_certain.utilities.settings import Settings, OutputModes


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
        """Initialize the sidebar configuration."""
        self.dataset_options = {name.value: name for name in DatasetNames}
        self.uncertainty_options = {tech.value: tech for tech in UncertaintyTechniques}
        self.consistency_options = {tech.value: tech for tech in ConsistencyTechniques}

        # Default values
        self.selected_dataset_value = "ionosphere"
        self.n_workers_min = 3
        self.n_workers_max = 8
        self.low_quality = 0.4
        self.high_quality = 1.0
        self.selected_uncertainty_values = [UncertaintyTechniques.STD.value]
        self.selected_consistency_values = [ConsistencyTechniques.ONE_MINUS_UNCERTAINTY.value]
        self.num_seeds = 3
        self.auto_download = True
        self.read_mode = ReadMode.AUTO.value

        # Get the correct dataset path
        self.dataset_path = self._get_correct_dataset_path()

    def _get_correct_dataset_path(self) -> Path:
        """
        Get the correct dataset path without duplication.

        Returns:
            Path object pointing to the datasets directory
        """
        # Start with the current file's directory
        current_dir = Path(__file__).parent.parent

        # Create path to datasets directory
        datasets_dir = current_dir / "datasets"

        # If running from the installed package, the path might be different
        if not datasets_dir.exists():
            # Try alternative paths
            alt_path = Path.cwd() / "datasets"
            if alt_path.exists():
                return alt_path

            # Another common location
            alt_path = Path.cwd() / "crowd_certain" / "datasets"
            if alt_path.exists():
                return alt_path

            # Create the directory if it doesn't exist
            print(f"Creating datasets directory at: {datasets_dir}")
            datasets_dir.mkdir(parents=True, exist_ok=True)

        # Verify the structure by checking for at least one dataset directory
        if datasets_dir.exists():
            # Check if any dataset directories exist
            dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir() and d.name in [name.value for name in DatasetNames]]
            if dataset_dirs:
                print(f"Found {len(dataset_dirs)} dataset directories in {datasets_dir}")
            else:
                print(f"No dataset directories found in {datasets_dir}. Will create them as needed.")

        return datasets_dir

    def render(self) -> None:
        """Render the sidebar configuration options."""
        st.sidebar.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

        # Dataset selection
        self.selected_dataset_value = st.sidebar.selectbox(
            "Select Dataset",
            options=list(self.dataset_options.keys()),
            index=list(self.dataset_options.keys()).index(self.selected_dataset_value),
            help="Choose a dataset to run the simulation on"
        )

        # Number of workers
        self.n_workers_min = st.sidebar.slider(
            "Minimum Number of Workers",
            min_value=2,
            max_value=10,
            value=self.n_workers_min,
            help="Minimum number of workers in the simulation"
        )

        self.n_workers_max = st.sidebar.slider(
            "Maximum Number of Workers",
            min_value=self.n_workers_min,
            max_value=20,
            value=self.n_workers_max,
            help="Maximum number of workers in the simulation"
        )

        # Worker quality range
        worker_quality_col1, worker_quality_col2 = st.sidebar.columns(2)
        with worker_quality_col1:
            self.low_quality = st.number_input(
                "Minimum Worker Quality",
                min_value=0.0,
                max_value=1.0,
                value=self.low_quality,
                step=0.05,
                help="Minimum worker quality (0.0 = random guessing, 1.0 = perfect)"
            )
        with worker_quality_col2:
            self.high_quality = st.number_input(
                "Maximum Worker Quality",
                min_value=self.low_quality,
                max_value=1.0,
                value=self.high_quality,
                step=0.05,
                help="Maximum worker quality (0.0 = random guessing, 1.0 = perfect)"
            )

        # Uncertainty techniques
        self.selected_uncertainty_values = st.sidebar.multiselect(
            "Uncertainty Techniques",
            options=list(self.uncertainty_options.keys()),
            default=self.selected_uncertainty_values,
            help="Techniques to measure uncertainty in worker predictions"
        )

        # Consistency techniques
        self.selected_consistency_values = st.sidebar.multiselect(
            "Consistency Techniques",
            options=list(self.consistency_options.keys()),
            default=self.selected_consistency_values,
            help="Techniques to convert uncertainty to consistency scores"
        )

        # Number of seeds
        self.num_seeds = st.sidebar.slider(
            "Number of Random Seeds",
            min_value=1,
            max_value=10,
            value=self.num_seeds,
            help="Number of random seeds to use for the simulation"
        )

        # Advanced options
        with st.sidebar.expander("Advanced Options"):
            self.auto_download = st.checkbox(
                "Auto-download Datasets",
                value=self.auto_download,
                help="Automatically download datasets if they are not found locally"
            )

            self.read_mode = st.selectbox(
                "Read Mode",
                options=[mode.value for mode in ReadMode],
                index=[mode.value for mode in ReadMode].index(self.read_mode),
                help="Method to read datasets"
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

    def get_selected_dataset(self) -> DatasetNames:
        """Get the selected dataset as a DatasetNames enum."""
        return self.dataset_options[self.selected_dataset_value]

    def get_selected_uncertainty(self) -> List[UncertaintyTechniques]:
        """Get the selected uncertainty techniques as UncertaintyTechniques enums."""
        return [self.uncertainty_options[val] for val in self.selected_uncertainty_values]

    def get_selected_consistency(self) -> List[ConsistencyTechniques]:
        """Get the selected consistency techniques as ConsistencyTechniques enums."""
        return [self.consistency_options[val] for val in self.selected_consistency_values]

    def create_config(self) -> Settings:
        """Create a Settings object from the current configuration."""
        return Settings(
            dataset=dict(
                dataset_name=self.get_selected_dataset(),
                datasetNames=[self.get_selected_dataset()],
                read_mode=ReadMode(self.read_mode),
                path_all_datasets=self.dataset_path
            ),
            simulation=dict(
                n_workers_min_max=[self.n_workers_min, self.n_workers_max],
                low_dis=self.low_quality,
                high_dis=self.high_quality,
                num_seeds=self.num_seeds,
            ),
            technique=dict(
                uncertainty_techniques=self.get_selected_uncertainty(),
                consistency_techniques=self.get_selected_consistency(),
            ),
            output=dict(
                mode=OutputModes.CALCULATE,
                save=False,
            )
        )


class SimulationRunner:
    """Handles running simulations and displaying results."""

    def __init__(self, config: SidebarConfig):
        """Initialize the simulation runner with configuration."""
        self.config = config

    def run_simulation(self) -> None:
        """Run the simulation with the current configuration."""
        with st.spinner("Running simulation..."):
            # Create configuration
            settings = self.config.create_config()

            # Check if dataset exists and show appropriate message
            dataset_dir = settings.dataset.path_all_datasets / self.config.selected_dataset_value
            dataset_file = dataset_dir / f"{self.config.selected_dataset_value}.csv"

            # Display dataset path information (can be removed in production)
            st.info(f"Looking for dataset at: {dataset_file}")
            st.info(f"Dataset base directory: {settings.dataset.path_all_datasets}")

            if not dataset_file.exists():
                if self.config.auto_download:
                    st.info(f"Dataset {self.config.selected_dataset_value} not found locally. Will attempt to download...")
                else:
                    st.warning(f"Dataset {self.config.selected_dataset_value} not found locally and auto-download is disabled. The simulation may fail.")

            # Run the simulation
            try:
                results = AIM1_3.calculate_one_dataset(config=settings)
                st.session_state.results = results
                st.success(f"Simulation completed successfully for {self.config.selected_dataset_value} dataset!")
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
                st.exception(e)

                # Provide helpful message for common errors
                error_str = str(e)
                if "Could not load dataset" in error_str:
                    st.warning(
                        "The dataset could not be found or downloaded. Please check your internet connection "
                        "and try again. If the problem persists, you can manually download the dataset from "
                        "the UCI Machine Learning Repository and place it in the appropriate directory: "
                        f"{settings.dataset.path_all_datasets / self.config.selected_dataset_value}"
                    )


class ResultsTab:
    """Manages the Results tab content."""

    @staticmethod
    def render() -> None:
        """Render the Results tab content."""
        st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)

        if 'results' not in st.session_state:
            return

        results = st.session_state.results

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

        if 'results' not in st.session_state:
            return

        results = st.session_state.results

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
