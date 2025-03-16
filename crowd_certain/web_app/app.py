"""
Crowd-Certain Web Application

A Flask-based web interface for the Crowd-Certain framework.
This web app allows users to run simulations, analyze results, and explore the performance
of different crowd-sourced label aggregation techniques.
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add the parent directory to the path so we can import from crowd_certain
sys.path.append(str(Path(__file__).parent.parent))

# Import Crowd-Certain modules
from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities import params
from crowd_certain.utilities.settings import Settings
from crowd_certain.utilities.params import OutputModes
from crowd_certain.utilities.dataset_loader import find_dataset_path

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Default configuration
default_config = Settings(
    dataset=dict(
        dataset_name = params.DatasetNames.IONOSPHERE,
        datasetNames = [params.DatasetNames.IONOSPHERE],
        path_all_datasets = Path(__file__).parent.parent / 'datasets'
    ),
    simulation=dict(
        low_dis = 0.4,
        high_dis = 1.0,
        num_seeds = 3,
        n_workers_min_max = [3, 8],
        use_parallelization = True,
        max_parallel_workers = min(4, os.cpu_count() or 4)
    ),
    technique=dict(
        uncertainty_techniques = [params.UncertaintyTechniques.STD],
        consistency_techniques = [params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
    ),
    output=dict(
        mode=OutputModes.CALCULATE,
        save=False,
    )
)

@app.route('/')
def index():
    """Render the main page of the web application."""
    # Get all available datasets
    dataset_options = {name.value: name.value for name in params.DatasetNames}

    # Get all available uncertainty techniques
    uncertainty_options = {tech.value: tech.value for tech in params.UncertaintyTechniques}

    # Get all available consistency techniques
    consistency_options = {tech.value: tech.value for tech in params.ConsistencyTechniques}

    return render_template('index.html',
                            dataset_options=dataset_options,
                            uncertainty_options=uncertainty_options,
                            consistency_options=consistency_options)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run a simulation with the provided parameters."""
    try:
        # Get parameters from the form
        data = request.json

        # Get selected datasets
        selected_datasets = data.get('datasets', ['IONOSPHERE'])
        dataset_names = [next((name for name in params.DatasetNames if name.value == ds), None)
                        for ds in selected_datasets]
        dataset_names = [ds for ds in dataset_names if ds is not None]

        if not dataset_names:
            return jsonify({'error': 'No valid datasets selected'})

        # Get worker parameters
        n_workers_min = int(data.get('workers_min', 3))
        n_workers_max = int(data.get('workers_max', 8))
        low_quality = float(data.get('quality_min', 0.4))
        high_quality = float(data.get('quality_max', 1.0))
        num_seeds = int(data.get('seeds', 3))

        # Get selected techniques
        uncertainty_techniques = data.get('uncertainty_techniques', ['STD'])
        uncertainty_techs = [next((tech for tech in params.UncertaintyTechniques if tech.value == t), None)
                            for t in uncertainty_techniques]
        uncertainty_techs = [t for t in uncertainty_techs if t is not None]

        consistency_techniques = data.get('consistency_techniques', ['ONE_MINUS_UNCERTAINTY'])
        consistency_techs = [next((tech for tech in params.ConsistencyTechniques if tech.value == t), None)
                            for t in consistency_techniques]
        consistency_techs = [t for t in consistency_techs if t is not None]

        # Create configuration
        config = Settings(
            dataset=dict(
                datasetNames = dataset_names,
                path_all_datasets = Path(__file__).parent.parent / 'datasets'
            ),
            simulation=dict(
                n_workers_min_max = [n_workers_min, n_workers_max],
                low_dis = low_quality,
                high_dis = high_quality,
                num_seeds = num_seeds,
                use_parallelization = True,
                max_parallel_workers = min(4, os.cpu_count() or 4)
            ),
            technique=dict(
                uncertainty_techniques = uncertainty_techs,
                consistency_techniques = consistency_techs,
            ),
            output=dict(
                mode = OutputModes.CALCULATE,
                save = False,
            )
        )

        # Run simulation
        if len(dataset_names) == 1:
            config.dataset.dataset_name = dataset_names[0]
            results = AIM1_3.calculate_one_dataset(config=config)

            # Process results for the single dataset
            processed_results = process_results(results)
            return jsonify(processed_results)
        else:
            results = AIM1_3.calculate_all_datasets(config=config)

            # Process results for multiple datasets
            all_processed_results = {}
            for dataset_name, dataset_results in results.items():
                all_processed_results[dataset_name] = process_results(dataset_results)

            # TODO: Need to fix the issue with this
            return jsonify(all_processed_results)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})

def process_results(results):
    """Process the results to return JSON-serializable data."""
    processed_data = {}

    # Get the first result
    first_nl_key = list(results.outputs.keys())[0]
    first_result = results.outputs[first_nl_key][0]

    # Get proposed metrics
    proposed_metrics = first_result.proposed.metrics.to_dict()
    processed_data['proposed_metrics'] = proposed_metrics

    # Get benchmark metrics
    benchmark_metrics = first_result.benchmark.metrics.to_dict()
    processed_data['benchmark_metrics'] = benchmark_metrics

    # Get worker strengths
    if hasattr(first_result, 'workers_strength'):
        processed_data['worker_strengths'] = first_result.workers_strength.to_dict()

    # Get worker weights
    if hasattr(first_result, 'weight'):
        processed_data['worker_weights'] = {
            'proposed': first_result.weight.PROPOSED.to_dict() if hasattr(first_result.weight, 'PROPOSED') else {},
            'tao': first_result.weight.TAO.to_dict() if hasattr(first_result.weight, 'TAO') else {},
            'sheng': first_result.weight.SHENG.to_dict() if hasattr(first_result.weight, 'SHENG') else {}
        }

    # Get confidence scores
    if hasattr(first_result.proposed, 'confidence_scores'):
        processed_data['confidence_scores'] = {
            k: v.to_dict() for k, v in first_result.proposed.confidence_scores.items()
        }

    return processed_data

@app.route('/datasets')
def list_datasets():
    """Return a list of available datasets."""
    datasets = [name.value for name in params.DatasetNames]
    return jsonify(datasets)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
