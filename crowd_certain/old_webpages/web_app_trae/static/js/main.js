/**
 * Crowd-Certain Web Interface
 *
 * Main JavaScript file for handling user interactions, API calls,
 * and result visualization in the Crowd-Certain web interface.
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Form submission handler
    const simulationForm = document.getElementById('simulationForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const noResultsMessage = document.getElementById('noResultsMessage');
    const errorMessage = document.getElementById('errorMessage');

    // Parallelization options toggle
    const useParallelization = document.getElementById('useParallelization');
    const parallelizationOptions = document.getElementById('parallelizationOptions');

    useParallelization.addEventListener('change', function() {
        parallelizationOptions.style.display = this.checked ? 'block' : 'none';
    });

    // Form submission
    simulationForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultsContainer.style.display = 'none';
        noResultsMessage.style.display = 'none';
        errorMessage.style.display = 'none';

        // Get form values
        const datasets = Array.from(document.getElementById('datasets').selectedOptions).map(option => option.value);
        const workersMin = parseInt(document.getElementById('workersMin').value);
        const workersMax = parseInt(document.getElementById('workersMax').value);
        const qualityMin = parseFloat(document.getElementById('qualityMin').value);
        const qualityMax = parseFloat(document.getElementById('qualityMax').value);
        const seeds = parseInt(document.getElementById('seeds').value);
        const uncertaintyTechniques = Array.from(document.getElementById('uncertaintyTechniques').selectedOptions).map(option => option.value);
        const consistencyTechniques = Array.from(document.getElementById('consistencyTechniques').selectedOptions).map(option => option.value);
        const useParallelizationValue = document.getElementById('useParallelization').checked;
        const maxParallelWorkers = parseInt(document.getElementById('maxParallelWorkers').value);

        // Validate form
        if (datasets.length === 0) {
            showError('Please select at least one dataset');
            return;
        }

        if (workersMin > workersMax) {
            showError('Minimum number of workers cannot be greater than maximum');
            return;
        }

        if (qualityMin > qualityMax) {
            showError('Minimum worker quality cannot be greater than maximum');
            return;
        }

        if (uncertaintyTechniques.length === 0) {
            showError('Please select at least one uncertainty technique');
            return;
        }

        if (consistencyTechniques.length === 0) {
            showError('Please select at least one consistency technique');
            return;
        }

        // Prepare request data
        const requestData = {
            datasets: datasets,
            workers_min: workersMin,
            workers_max: workersMax,
            quality_min: qualityMin,
            quality_max: qualityMax,
            seeds: seeds,
            uncertainty_techniques: uncertaintyTechniques,
            consistency_techniques: consistencyTechniques,
            use_parallelization: useParallelizationValue,
            max_parallel_workers: maxParallelWorkers
        };

        // Send API request
        fetch('/run_simulation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'An error occurred while running the simulation');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';

            // Process and display results
            if (data.results && Object.keys(data.results).length > 0) {
                displayResults(data.results);
                resultsContainer.style.display = 'block';
            } else {
                noResultsMessage.style.display = 'block';
            }
        })
        .catch(error => {
            showError(error.message);
        });
    });

    // Function to display error messages
    function showError(message) {
        loadingIndicator.style.display = 'none';
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    // Function to display results
    function displayResults(results) {
        // Get the first dataset to display initially
        const datasets = Object.keys(results);
        const firstDataset = datasets[0];

        // Create dataset selector buttons
        createDatasetSelectors(datasets, results);

        // Display the first dataset results
        displayDatasetResults(firstDataset, results[firstDataset]);
    }

    // Function to create dataset selector buttons
    function createDatasetSelectors(datasets, results) {
        const datasetSelector = document.getElementById('datasetSelector');
        const wsDatasetSelector = document.getElementById('wsDatasetSelector');

        // Clear existing buttons
        datasetSelector.innerHTML = '';
        wsDatasetSelector.innerHTML = '';

        // Create buttons for each dataset
        datasets.forEach((dataset, index) => {
            // For metrics tab
            const button = document.createElement('button');
            button.type = 'button';
            button.className = index === 0 ? 'btn btn-primary btn-sm' : 'btn btn-outline-primary btn-sm';
            button.textContent = dataset;
            button.addEventListener('click', function() {
                // Update active button
                Array.from(datasetSelector.children).forEach(btn => {
                    btn.className = 'btn btn-outline-primary btn-sm';
                });
                this.className = 'btn btn-primary btn-sm';

                // Display selected dataset results
                displayDatasetResults(dataset, results[dataset]);
            });
            datasetSelector.appendChild(button);

            // For weight-strength tab
            const wsButton = document.createElement('button');
            wsButton.type = 'button';
            wsButton.className = index === 0 ? 'btn btn-primary btn-sm' : 'btn btn-outline-primary btn-sm';
            wsButton.textContent = dataset;
            wsButton.addEventListener('click', function() {
                // Update active button
                Array.from(wsDatasetSelector.children).forEach(btn => {
                    btn.className = 'btn btn-outline-primary btn-sm';
                });
                this.className = 'btn btn-primary btn-sm';

                // Display selected dataset weight-strength relation
                displayWeightStrengthRelation(dataset, results[dataset]);
            });
            wsDatasetSelector.appendChild(wsButton);
        });
    }

    // Function to display dataset results
    function displayDatasetResults(datasetName, datasetResults) {
        // Update dataset name
        document.getElementById('currentDataset').textContent = datasetName;
        document.getElementById('wsCurrentDataset').textContent = datasetName;

        // Display metrics tables
        displayMetricsTable(datasetResults.metrics.proposed, 'proposedMetricsTable');
        displayMetricsTable(datasetResults.metrics.benchmark, 'benchmarkMetricsTable');

        // Display weight-strength relation
        displayWeightStrengthRelation(datasetName, datasetResults);
    }

    // Function to display metrics table
    function displayMetricsTable(metrics, containerId) {
        const container = document.getElementById(containerId);

        if (!metrics || metrics.length === 0) {
            container.innerHTML = '<p class="text-muted">No metrics data available</p>';
            return;
        }

        // Create table
        const table = document.createElement('table');
        table.className = 'table table-sm table-hover';

        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');

        // Get all unique column names
        const columns = new Set();
        metrics.forEach(row => {
            Object.keys(row).forEach(key => columns.add(key));
        });

        // Add header cells
        columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = column.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        const tbody = document.createElement('tbody');

        metrics.forEach(row => {
            const tr = document.createElement('tr');

            columns.forEach(column => {
                const td = document.createElement('td');

                // Format numeric values
                if (typeof row[column] === 'number') {
                    td.textContent = row[column].toFixed(4);
                } else {
                    td.textContent = row[column] || '';
                }

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        container.innerHTML = '';
        container.appendChild(table);
    }

    // Function to display weight-strength relation
    function displayWeightStrengthRelation(datasetName, datasetResults) {
        const container = document.getElementById('weightStrengthPlot');

        if (!datasetResults.weight_strength_relation || datasetResults.weight_strength_relation.length === 0) {
            container.innerHTML = '<p class="text-muted">No weight-strength relation data available</p>';
            return;
        }

        // Prepare data for plotting
        const data = [];
        const wsData = datasetResults.weight_strength_relation;

        // Group data by method
        const methodGroups = {};

        wsData.forEach(row => {
            const method = row.method || 'Unknown';

            if (!methodGroups[method]) {
                methodGroups[method] = {
                    x: [],
                    y: [],
                    name: method,
                    mode: 'markers',
                    type: 'scatter',
                    marker: { size: 8 }
                };
            }

            methodGroups[method].x.push(row.strength || 0);
            methodGroups[method].y.push(row.weight || 0);
        });

        // Convert to array for Plotly
        for (const method in methodGroups) {
            data.push(methodGroups[method]);
        }

        // Plot layout
        const layout = {
            title: `Weight-Strength Relation for ${datasetName}`,
            xaxis: {
                title: 'Worker Strength',
                zeroline: true
            },
            yaxis: {
                title: 'Assigned Weight',
                zeroline: true
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            hovermode: 'closest',
            showlegend: true
        };

        // Create plot
        Plotly.newPlot(container, data, layout);
    }
});
