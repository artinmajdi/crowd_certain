"""
Example: Running the Streamlit dashboard programmatically

This example demonstrates how to run the Streamlit dashboard programmatically.
"""

import sys
import os
from pathlib import Path
import subprocess
import webbrowser
import time


def run_dashboard_with_subprocess():
    """Run the dashboard using subprocess."""
    print("Starting Crowd-Certain Dashboard using subprocess...")

    # Get the path to the dashboard.py file - updated path
    dashboard_path = Path(__file__).parent.parent / "utilities" / "visualization" / "dashboard.py"

    # Check if the file exists
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return False

    print(f"Dashboard file found at: {dashboard_path}")

    # Run the dashboard using subprocess
    try:
        # Start the dashboard in a subprocess
        process = subprocess.Popen(
            ["streamlit", "run", str(dashboard_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a moment for the dashboard to start
        time.sleep(3)

        # Open the dashboard in a web browser
        webbrowser.open("http://localhost:8501")

        print("Dashboard started successfully!")
        print("Press Ctrl+C to stop the dashboard...")

        # Wait for the user to press Ctrl+C
        try:
            while True:
                # Print output from the subprocess
                output = process.stdout.readline()
                if output:
                    print(output.strip())

                # Check if the process is still running
                if process.poll() is not None:
                    break

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping dashboard...")
        finally:
            # Terminate the subprocess
            process.terminate()
            process.wait()

        return True

    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        return False


def run_dashboard_with_streamlit_cli():
    """Run the dashboard using Streamlit CLI."""
    print("Starting Crowd-Certain Dashboard using Streamlit CLI...")

    # Get the path to the dashboard.py file
    dashboard_path = Path(__file__).parent.parent / "utilities" / "dashboard.py"

    # Check if the file exists
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return False

    print(f"Dashboard file found at: {dashboard_path}")

    # Run the dashboard using Streamlit CLI
    try:
        import streamlit.web.cli as stcli

        # Save the original sys.argv
        original_argv = sys.argv.copy()

        # Set sys.argv for Streamlit
        sys.argv = ["streamlit", "run", str(dashboard_path)]

        # Run the dashboard
        print("Starting dashboard...")
        stcli.main()

        # Restore the original sys.argv
        sys.argv = original_argv

        return True

    except ImportError:
        print("Error: Streamlit is not installed. Please install it with 'pip install streamlit'")
        return False
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        return False


if __name__ == "__main__":
    # Choose one of the following methods to run the dashboard
    run_dashboard_with_subprocess()
    # run_dashboard_with_streamlit_cli()
