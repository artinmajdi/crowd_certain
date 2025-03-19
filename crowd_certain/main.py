"""
Crowd-Certain Main Entry Point

This module provides a command-line interface for running different aspects of the Crowd-Certain framework.
"""

import argparse
import sys
from pathlib import Path

from crowd_certain.utilities.utils import AIM1_3
from crowd_certain.utilities.config import params
from crowd_certain.utilities.config.settings import Settings, OutputModes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crowd-Certain: A framework for crowd-sourced label aggregation")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run simulation on a single dataset
    single_parser = subparsers.add_parser("single", help="Run simulation on a single dataset")
    single_parser.add_argument("--dataset", "-d", type=str, required=True, choices=[name.value for name in params.DatasetNames], help="Dataset name to use for simulation")
    single_parser.add_argument("--workers-min" , type=int   , default=3    , help="Minimum number of workers (default: 3)")
    single_parser.add_argument("--workers-max" , type=int   , default=8    , help="Maximum number of workers (default: 8)")
    single_parser.add_argument("--quality-min" , type=float , default=0.4  , help="Minimum worker quality (default: 0.4)")
    single_parser.add_argument("--quality-max" , type=float , default=1.0  , help="Maximum worker quality (default: 1.0)")
    single_parser.add_argument("--seeds"       , type=int   , default=3    , help="Number of random seeds (default: 3)")
    single_parser.add_argument("--output-dir"  , type=str   , default=None , help="Directory to save results (default: None , results not saved)")

    # Run simulation on all datasets
    all_parser = subparsers.add_parser("all", help="Run simulation on all datasets")
    all_parser.add_argument("--workers-min" , type=int   , default=3    , help="Minimum number of workers (default: 3)")
    all_parser.add_argument("--workers-max" , type=int   , default=8    , help="Maximum number of workers (default: 8)")
    all_parser.add_argument("--quality-min" , type=float , default=0.4  , help="Minimum worker quality (default: 0.4)")
    all_parser.add_argument("--quality-max" , type=float , default=1.0  , help="Maximum worker quality (default: 1.0)")
    all_parser.add_argument("--seeds"       , type=int   , default=3    , help="Number of random seeds (default: 3)")
    all_parser.add_argument("--output-dir"  , type=str   , default=None , help="Directory to save results (default: None , results not saved)")

    # Run dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the Streamlit dashboard")

    # List available datasets
    subparsers.add_parser("list-datasets", help="List available datasets")

    return parser.parse_args()


def run_single_dataset(args):
    """Run simulation on a single dataset."""
    print(f"Running simulation on dataset: {args.dataset}")

    # Convert dataset name string to enum
    dataset_name = next((name for name in params.DatasetNames if name.value == args.dataset), None)
    if dataset_name is None:
        print(f"Error: Dataset '{args.dataset}' not found")
        return

    # Create configuration
    config = Settings(
        dataset=dict(
            dataset_name      = dataset_name,
            datasetNames      = [dataset_name],
            path_all_datasets = Path("crowd_certain/datasets")
        ),
        simulation=dict(
            n_workers_min_max = [args.workers_min, args.workers_max],
            low_dis           = args.quality_min,
            high_dis          = args.quality_max,
            num_seeds         = args.seeds,
        ),
        technique=dict(
            uncertainty_techniques = [params.UncertaintyTechniques.STD],
            consistency_techniques = [params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
        ),
        output=dict(
            mode = OutputModes.CALCULATE if args.output_dir is None else OutputModes.SAVE,
            save = args.output_dir is not None,
            path = args.output_dir if args.output_dir is not None else None,
        )
    )

    # Run simulation
    try:
        results = AIM1_3.calculate_one_dataset(config=config)
        print(f"Simulation completed successfully for {args.dataset} dataset!")

        # Display some basic results
        first_nl_key = list(results.outputs.keys())[0]
        first_result = results.outputs[first_nl_key][0]

        print("\nProposed Methods Metrics:")
        print(first_result.proposed.metrics)

        print("\nBenchmark Methods Metrics:")
        print(first_result.benchmark.metrics)

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()


def run_all_datasets(args):
    """Run simulation on all datasets."""
    print("Running simulation on all datasets")

    # Create configuration
    config = Settings(
        dataset=dict(
            datasetNames      = list(params.DatasetNames),
            path_all_datasets = Path("crowd_certain/datasets")
        ),
        simulation=dict(
            n_workers_min_max = [args.workers_min, args.workers_max],
            low_dis           = args.quality_min,
            high_dis          = args.quality_max,
            num_seeds         = args.seeds,
        ),
        technique=dict(
            uncertainty_techniques = [params.UncertaintyTechniques.STD],
            consistency_techniques = [params.ConsistencyTechniques.ONE_MINUS_UNCERTAINTY],
        ),
        output=dict(
            mode = OutputModes.CALCULATE if args.output_dir is None else OutputModes.SAVE,
            save = args.output_dir is not None,
            path = args.output_dir if args.output_dir is not None else None,
        )
    )

    # Run simulation
    try:
        results = AIM1_3.calculate_all_datasets(config=config)
        print("Simulation completed successfully for all datasets!")

        # Display summary of results
        for dataset_name, dataset_results in results.items():
            print(f"\nResults for dataset: {dataset_name}")
            first_nl_key = list(dataset_results.outputs.keys())[0]
            first_result = dataset_results.outputs[first_nl_key][0]

            print("Proposed Methods Accuracy:")
            print(first_result.proposed.metrics.loc["accuracy"])

    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()


def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        import streamlit.web.cli as stcli
        import os

        # Get the path to the dashboard.py file
        dashboard_path = Path(__file__).parent / "utilities" / "dashboard.py"

        # Check if the file exists
        if not dashboard_path.exists():
            print(f"Error: Dashboard file not found at {dashboard_path}")
            return

        print(f"Starting Crowd-Certain Dashboard from {dashboard_path}")

        # Run the dashboard using Streamlit CLI
        sys.argv = ["streamlit", "run", str(dashboard_path)]
        stcli.main()

    except ImportError:
        print("Error: Streamlit is not installed. Please install it with 'pip install streamlit'")
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        import traceback
        traceback.print_exc()


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    for name in params.DatasetNames:
        print(f"  - {name.value}")


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "single":
        run_single_dataset(args)
    elif args.command == "all":
        run_all_datasets(args)
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "list-datasets":
        list_datasets()
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == "__main__":
    main()
