# Crowd-Certain Web Application

A beginner-friendly web interface for the Crowd-Certain framework, allowing users to run simulations, analyze results, and explore the performance of different crowd-sourced label aggregation techniques.

## Features

- **Simple Configuration**: Easy-to-use interface for setting up simulation parameters
- **Interactive Visualizations**: Visualize results using Plotly.js charts
- **Worker Analysis**: Analyze worker strengths and weights
- **Benchmark Comparisons**: Compare proposed methods against established benchmarks
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.7+
- Flask
- Plotly.js (included via CDN)
- Bootstrap 5 (included via CDN)

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install flask
```

## Usage

1. Navigate to the web_app directory:

```bash
cd /path/to/crowd_certain/web_app
```

2. Run the Flask application:

```bash
python app.py
```

3. Open your web browser and go to:

```
http://localhost:5000
```

## Configuration Options

The web application allows you to configure the following parameters:

- **Datasets**: Select one or more datasets to run simulations on
- **Number of Workers**: Set the minimum and maximum number of workers
- **Worker Quality Range**: Set the minimum and maximum worker quality
- **Number of Seeds**: Set the number of random seeds for the simulation
- **Uncertainty Techniques**: Select one or more uncertainty quantification techniques
- **Consistency Techniques**: Select one or more consistency measurement techniques

## Sections

### Simulation

Configure and run simulations with different parameters. The application will display the status of the simulation and any errors that occur.

### Results

View the results of the simulation, including:

- **Metrics**: Accuracy, F1 score, precision, and recall for both proposed and benchmark methods
- **Confidence Scores**: Distribution of confidence scores for the aggregated labels

### Worker Analysis

Analyze the performance of workers, including:

- **Worker Strength**: The estimated quality of each worker
- **Worker Weights**: The weights assigned to each worker by different methods
- **Strength vs Weight**: The relationship between worker strength and weight

### About

Learn about the key features of the Crowd-Certain framework, benchmark methods, and dataset structure.

## Development

The web application is built using:

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Styling**: Bootstrap 5
- **Visualizations**: Plotly.js

### Project Structure

```
web_app/
├── app.py                 # Flask application
├── static/
│   ├── css/
│   │   └── styles.css     # Custom CSS styles
│   ├── js/
│   │   └── main.js        # JavaScript for interactive features
│   └── images/            # Image assets
└── templates/
    └── index.html         # Main HTML template
```

## License

This project is part of the Crowd-Certain framework. See the main repository for license information.
