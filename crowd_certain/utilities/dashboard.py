import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Import directly from the utilities module since we're now in the same package
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.params import DatasetNames, UncertaintyTechniques, ConsistencyTechniques
from crowd_certain.utilities.settings import Settings, OutputModes
from crowd_certain.utilities.params import ReadMode

# Set page configuration
st.set_page_config(
    page_title="Crowd-Certain Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
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

# Header
st.markdown('<div class="main-header">Crowd-Certain: Crowd-Sourced Label Aggregation</div>', unsafe_allow_html=True)
st.markdown("""
Crowd-Certain is a comprehensive framework for aggregating labels from multiple annotators (crowd workers)
while estimating the uncertainty and confidence in the aggregated labels.
""")

# Sidebar
st.sidebar.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

# Dataset selection
dataset_options = {name.value: name for name in DatasetNames}
selected_dataset_value = st.sidebar.selectbox(
    "Select Dataset",
    options=list(dataset_options.keys()),
    index=list(dataset_options.keys()).index("ionosphere"),
    help="Choose a dataset to run the simulation on"
)
selected_dataset = dataset_options[selected_dataset_value]

# Number of workers
n_workers_min = st.sidebar.slider(
    "Minimum Number of Workers",
    min_value=2,
    max_value=10,
    value=3,
    help="Minimum number of workers in the simulation"
)

n_workers_max = st.sidebar.slider(
    "Maximum Number of Workers",
    min_value=n_workers_min,
    max_value=20,
    value=8,
    help="Maximum number of workers in the simulation"
)

# Worker quality range
worker_quality_col1, worker_quality_col2 = st.sidebar.columns(2)
with worker_quality_col1:
    low_quality = st.number_input(
        "Minimum Worker Quality",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Minimum worker quality (0.0 = random guessing, 1.0 = perfect)"
    )
with worker_quality_col2:
    high_quality = st.number_input(
        "Maximum Worker Quality",
        min_value=low_quality,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Maximum worker quality (0.0 = random guessing, 1.0 = perfect)"
    )

# Uncertainty techniques
uncertainty_options = {tech.value: tech for tech in UncertaintyTechniques}
selected_uncertainty_values = st.sidebar.multiselect(
    "Uncertainty Techniques",
    options=list(uncertainty_options.keys()),
    default=[UncertaintyTechniques.STD.value],
    help="Techniques to measure uncertainty in worker predictions"
)
selected_uncertainty = [uncertainty_options[val] for val in selected_uncertainty_values]

# Consistency techniques
consistency_options = {tech.value: tech for tech in ConsistencyTechniques}
selected_consistency_values = st.sidebar.multiselect(
    "Consistency Techniques",
    options=list(consistency_options.keys()),
    default=[ConsistencyTechniques.ONE_MINUS_UNCERTAINTY.value],
    help="Techniques to convert uncertainty to consistency scores"
)
selected_consistency = [consistency_options[val] for val in selected_consistency_values]

# Number of seeds
num_seeds = st.sidebar.slider(
    "Number of Random Seeds",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of random seeds to use for the simulation"
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    auto_download = st.checkbox(
        "Auto-download Datasets",
        value=True,
        help="Automatically download datasets if they are not found locally"
    )

    read_mode = st.selectbox(
        "Read Mode",
        options=[mode.value for mode in ReadMode],
        index=3,  # AUTO mode
        help="Method to read datasets"
    )

# Run button
run_simulation = st.sidebar.button("Run Simulation", type="primary")

# Main content
tab1, tab2, tab3 = st.tabs(["Simulation Results", "Worker Analysis", "About"])

with tab1:
    st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)

    if run_simulation:
        with st.spinner("Running simulation..."):
            # Create configuration
            config = Settings(
                dataset=dict(
                    dataset_name=selected_dataset,
                    datasetNames=[selected_dataset],
                    read_mode=ReadMode(read_mode),
                    path_all_datasets=Path("crowd_certain/datasets")
                ),
                simulation=dict(
                    n_workers_min_max=[n_workers_min, n_workers_max],
                    low_dis=low_quality,
                    high_dis=high_quality,
                    num_seeds=num_seeds,
                ),
                technique=dict(
                    uncertainty_techniques=selected_uncertainty,
                    consistency_techniques=selected_consistency,
                ),
                output=dict(
                    mode=OutputModes.CALCULATE,
                    save=False,
                )
            )

            # Check if dataset exists
            dataset_path = config.dataset.path_all_datasets / f"{selected_dataset_value}/{selected_dataset_value}.arff"
            if not dataset_path.exists() and auto_download:
                st.info(f"Dataset {selected_dataset_value} not found locally. Downloading...")

            # Run the simulation
            try:
                results = AIM1_3.calculate_one_dataset(config=config)
                st.session_state.results = results
                st.success(f"Simulation completed successfully for {selected_dataset_value} dataset!")
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
                st.exception(e)

                # Provide helpful message for common errors
                error_str = str(e)
                if "Could not find or download dataset" in error_str:
                    st.warning(
                        "The dataset could not be found or downloaded. Please check your internet connection "
                        "and try again. If the problem persists, you can manually download the dataset from "
                        "the UCI Machine Learning Repository and place it in the appropriate directory."
                    )

    if 'results' in st.session_state:
        results = st.session_state.results

        # Display metrics
        st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)

        # Get the first result to display metrics
        first_nl_key = list(results.outputs.keys())[0]
        first_result = results.outputs[first_nl_key][0]

        # Display metrics for proposed methods and benchmarks
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Proposed Methods")
            proposed_metrics = first_result.proposed.metrics
            st.dataframe(proposed_metrics)

        with col2:
            st.markdown("#### Benchmark Methods")
            benchmark_metrics = first_result.benchmark.metrics
            st.dataframe(benchmark_metrics)

        # Plot worker strength vs weight relationship
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

        # Display confidence scores
        st.markdown('<div class="sub-header">Confidence Scores</div>', unsafe_allow_html=True)

        # Get confidence scores for the first result
        confidence_scores = first_result.proposed.confidence_scores

        # Create tabs for each confidence score method
        if confidence_scores:
            confidence_tabs = st.tabs(list(confidence_scores.keys()))

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

with tab2:
    st.markdown('<div class="sub-header">Worker Analysis</div>', unsafe_allow_html=True)

    if 'results' in st.session_state:
        results = st.session_state.results

        # Get the first result to display worker information
        first_nl_key = list(results.outputs.keys())[0]
        first_result = results.outputs[first_nl_key][0]

        # Display worker strength information
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

        # Display worker weights
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

with tab3:
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

    st.markdown('<div class="sub-header">Contact</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>Artin Majdi - msm2024@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)
