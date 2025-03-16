#!/usr/bin/env python3
"""
Crowd-Certain Web Interface

A beginner-friendly Flask-based web interface for the Crowd-Certain framework.
This interface allows users to run simulations, visualize results, and explore
the performance of different crowd-sourced label aggregation techniques.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, abort
import pandas as pd
import numpy as np

# Import directly from the Crowd-Certain framework
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities import params
from crowd_certain.utilities.settings import Settings, get_settings, find_config_file
from crowd_certain.utilities.dataset_loader import find_dataset_path

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Ensure the instance path exists
os.makedirs(app.instance_path, exist_ok=True)

# Helper function to convert DataFrame to JSON
def dataframe_to_json(df):
    """Convert a pandas DataFrame to a JSON-serializable format."""
    if df is None:
        return None

    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        # Convert multi-index to strings
        df.columns = [' - '.join(map(str, col)).strip() for col in df.columns.values]

    # Convert DataFrame to dictionary
    # result = df.reset_index().fillna(value=None).to_dict(orient='records')
    result = df.reset_index().replace([np.nan, np.inf, -np.inf], None).to_dict(orient='records')

    return result

# Helper function to get available datasets
def get_available_datasets():
    """Get a list of all available datasets."""
    return [name.value for name in params.DatasetNames]

# Helper function to get uncertainty techniques
def get_uncertainty_techniques():
    """Get a list of available uncertainty techniques."""
    return [{'value': tech.value, 'name': tech.value.replace('_', ' ').title()}
            for tech in params.UncertaintyTechniques]

# Helper function to get consistency techniques
def get_consistency_techniques():
    """Get a list of available consistency techniques."""
    return [{'value': tech.value, 'name': tech.value.replace('_', ' ').title()}
            for tech in params.ConsistencyTechniques]

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html',
                          datasets=get_available_datasets(),
                          uncertainty_techniques=get_uncertainty_techniques(),
                          consistency_techniques=get_consistency_techniques())

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run a simulation with the provided configuration."""
    try:
        # Get configuration from request
        data = request.get_json()

        # Validate required fields
        if not data.get('datasets'):
            return jsonify({'error': 'No datasets selected'}), 400

        # Convert dataset strings to DatasetNames enums
        dataset_enums = [next(name for name in params.DatasetNames if name.value == ds)
                        for ds in data.get('datasets', [])]

        if not dataset_enums:
            return jsonify({'error': 'Invalid dataset selection'}), 400

        # Get uncertainty techniques
        uncertainty_enums = [next(tech for tech in params.UncertaintyTechniques if tech.value == tech_value)
                            for tech_value in data.get('uncertainty_techniques', ['standard_deviation'])]

        # Get consistency techniques
        consistency_enums = [next(tech for tech in params.ConsistencyTechniques if tech.value == tech_value)
                            for tech_value in data.get('consistency_techniques', ['one_minus_uncertainty'])]

        # Create settings
        settings = Settings(
            dataset=dict(
                dataset_name=dataset_enums[0],  # Use first dataset as primary
                datasetNames=dataset_enums,     # All selected datasets
                path_all_datasets=find_dataset_path()
            ),
            simulation=dict(
                n_workers_min_max=[
                    data.get('workers_min', 3),
                    data.get('workers_max', 8)
                ],
                low_dis=data.get('quality_min', 0.4),
                high_dis=data.get('quality_max', 1.0),
                num_seeds=data.get('seeds', 1),
                use_parallelization=data.get('use_parallelization', False),
                max_parallel_workers=data.get('max_parallel_workers', 1)
            ),
            technique=dict(
                uncertainty_techniques=uncertainty_enums,
                consistency_techniques=consistency_enums,
            ),
            output=dict(
                mode=params.OutputModes.CALCULATE,
                save=False,
            )
        )

        # Prepare response object
        response = {'results': {}}

        # Run simulation for each selected dataset
        for dataset_name in data.get('datasets', []):
            # Update settings for this specific dataset
            dataset_enum = next(name for name in params.DatasetNames if name.value == dataset_name)
            settings.dataset.dataset_name = dataset_enum

            # Run the simulation for this dataset
            results = AIM1_3.calculate_one_dataset(config=settings, dataset_name=dataset_enum)

            # Process results into JSON-compatible format
            response['results'][dataset_name] = {
                'metrics': {
                    'proposed': dataframe_to_json(
                        next(iter(results.outputs.values()))[0].proposed.metrics
                    ),
                    'benchmark': dataframe_to_json(
                        next(iter(results.outputs.values()))[0].benchmark.metrics
                    )
                },
                'weight_strength_relation': dataframe_to_json(results.weight_strength_relation)
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/datasets')
def datasets():
    """Return a list of available datasets."""
    return jsonify(get_available_datasets())

@app.route('/uncertainty_techniques')
def uncertainty_techniques():
    """Return a list of available uncertainty techniques."""
    return jsonify(get_uncertainty_techniques())

@app.route('/consistency_techniques')
def consistency_techniques():
    """Return a list of available consistency techniques."""
    return jsonify(get_consistency_techniques())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
