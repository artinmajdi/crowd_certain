// DOM Elements
const simulationForm = document.getElementById('simulation-form');
const datasetSelect = document.getElementById('dataset-select');
const workersMin = document.getElementById('workers-min');
const workersMax = document.getElementById('workers-max');
const qualityMin = document.getElementById('quality-min');
const qualityMax = document.getElementById('quality-max');
const seeds = document.getElementById('seeds');
const useParallelization = document.getElementById('use-parallelization');
const parallelizationSettings = document.getElementById('parallelization-settings');
const maxParallelWorkers = document.getElementById('max-parallel-workers');
const runSimulationBtn = document.getElementById('run-simulation');
const loading = document.getElementById('loading');
const resultsContainer = document.getElementById('results-container');
const infoSection = document.getElementById('info-section');
const resultsTabs = document.getElementById('results-tabs');
const resultsContent = document.getElementById('results-content');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Toggle parallelization settings
    useParallelization.addEventListener('change', (e) => {
        parallelizationSettings.style.display = e.target.checked ? 'block' : 'none';
        if (e.target.checked) {
            // Set default to number of CPU cores / 2 (approximation)
            const defaultCores = navigator.hardwareConcurrency || 4;
            maxParallelWorkers.value = Math.max(1, Math.floor(defaultCores / 2));
            maxParallelWorkers.max = defaultCores;
        }
    });

    // Form validation
    workersMin.addEventListener('change', () => {
        const min = parseInt(workersMin.value);
        const max = parseInt(workersMax.value);
        if (min > max) {
            workersMax.value = min;
        }
    });

    workersMax.addEventListener('change', () => {
        const min = parseInt(workersMin.value);
        const max = parseInt(workersMax.value);
        if (max < min) {
            workersMin.value = max;
        }
    });

    qualityMin.addEventListener('change', () => {
        const min = parseFloat(qualityMin.value);
        const max = parseFloat(qualityMax.value);
        if (min > max) {
            qualityMax.value = min;
        }
    });

    qualityMax.addEventListener('change', () => {
        const min = parseFloat(qualityMin.value);
        const max = parseFloat(qualityMax.value);
        if (max < min) {
            qualityMin.value = max;
        }
    });

    // Form submission
    simulationForm.addEventListener('submit', handleFormSubmission);
});

// Form handling
async function handleFormSubmission(e) {
    e.preventDefault();

    // Get selected datasets
    const selectedDatasets = Array.from(datasetSelect.selectedOptions).map(option => option.value);
    if (selectedDatasets.length === 0) {
        alert('Please select at least one dataset.');
        return;
    }

    // Get selected uncertainty techniques
    const uncertaintyTechniques = Array.from(document.querySelectorAll('.uncertainty-technique:checked')).map(checkbox => checkbox.value);
    if (uncertaintyTechniques.length === 0) {
        alert('Please select at least one uncertainty technique.');
        return;
    }

    // Get selected consistency techniques
    const consistencyTechniques = Array.from(document.querySelectorAll('.consistency-technique:checked')).map(checkbox => checkbox.value);
    if (consistencyTechniques.length === 0) {
        alert('Please select at least one consistency technique.');
        return;
    }

    // Show loading state
    loading.style.display = 'block';
    infoSection.style.display = 'none';
    resultsContainer.style.display = 'none';
    runSimulationBtn.disabled = true;

    try {
        // Prepare request data
        const requestData = {
            datasets: selectedDatasets,
            workers_min: parseInt(workersMin.value),
            workers_max: parseInt(workersMax.value),
            quality_min: parseFloat(qualityMin.value),
            quality_max: parseFloat(qualityMax.value),
            seeds: parseInt(seeds.value),
            uncertainty_techniques: uncertaintyTechniques,
            consistency_techniques: consistencyTechniques,
            use_parallelization: useParallelization.checked,
            max_parallel_workers: parseInt(maxParallelWorkers.value)
        };

        // Send request to server
        const response = await fetch('/run_simulation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        // Parse response
        const data = await response.json();

        if (response.ok) {
            // Display results
            displayResults(data.results);
        } else {
            // Display error
            throw new Error(data.error || 'An unknown error occurred');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error('Error running simulation:', error);
        infoSection.style.display = 'block';
    } finally {
        // Hide loading state
        loading.style.display = 'none';
        runSimulationBtn.disabled = false;
    }
}

// Display results
function displayResults(results) {
    // Clear previous results
    resultsTabs.innerHTML = '';
    resultsContent.innerHTML = '';

    // Show results container
    resultsContainer.style.display = 'block';

    // Get dataset names
    const datasetNames = Object.keys(results);

    // Create tabs for each dataset
    datasetNames.forEach((datasetName, index) => {
        // Create tab
        const tab = document.createElement('li');
        tab.className = 'nav-item';
        tab.innerHTML = `
            <a class="nav-link ${index === 0 ? 'active' : ''}"
               id="tab-${datasetName}"
               data-bs-toggle="tab"
               href="#content-${datasetName}"
               role="tab"
               aria-controls="content-${datasetName}"
               aria-selected="${index === 0 ? 'true' : 'false'}">
                ${datasetName}
            </a>
        `;
        resultsTabs.appendChild(tab);

        // Create tab content
        const tabContent = document.createElement('div');
        tabContent.className = `tab-pane fade ${index === 0 ? 'show active' : ''}`;
        tabContent.id = `content-${datasetName}`;
        tabContent.setAttribute('role', 'tabpanel');
        tabContent.setAttribute('aria-labelledby', `tab-${datasetName}`);

        // Add dataset results
        const datasetResult = results[datasetName];

        // Add metrics tables
        tabContent.innerHTML += `
            <div class="metrics-section">
                <h4>Performance Metrics</h4>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Proposed Methods</h5>
                            </div>
                            <div class="card-body">
                                <div id="proposed-metrics-${datasetName}"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Benchmark Methods</h5>
                            </div>
                            <div class="card-body">
                                <div id="benchmark-metrics-${datasetName}"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Worker Strength vs. Weight Relationship</h5>
                        </div>
                        <div class="card-body">
                            <div id="weight-strength-plot-${datasetName}" class="plot-container"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        resultsContent.appendChild(tabContent);

        // Create metrics tables after DOM is updated
        setTimeout(() => {
            // Create proposed metrics table
            createMetricsTable(
                datasetResult.metrics.proposed,
                document.getElementById(`proposed-metrics-${datasetName}`)
            );

            // Create benchmark metrics table
            createMetricsTable(
                datasetResult.metrics.benchmark,
                document.getElementById(`benchmark-metrics-${datasetName}`)
            );

            // Create weight strength plot
            createWeightStrengthPlot(
                datasetResult.weight_strength_relation,
                document.getElementById(`weight-strength-plot-${datasetName}`)
            );
        }, 100);
    });
}

// Create metrics table
function createMetricsTable(metrics, container) {
    if (!metrics || metrics.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No metrics data available</div>';
        return;
    }

    // Get column names
    const columns = Object.keys(metrics[0]).filter(k => k !== 'index');

    // Create table
    const table = document.createElement('table');
    table.className = 'table table-sm table-striped';

    // Create header
    let thead = '<thead><tr>';
    thead += '<th>Metric</th>';
    columns.forEach(col => {
        thead += `<th>${col}</th>`;
    });
    thead += '</tr></thead>';

    // Create body
    let tbody = '<tbody>';
    metrics.forEach(row => {
        tbody += '<tr>';
        tbody += `<td><strong>${row.index || ''}</strong></td>`;
        columns.forEach(col => {
            // Format numbers to 3 decimal places
            const value = isNaN(row[col]) ? row[col] : Number(row[col]).toFixed(3);
            tbody += `<td>${value}</td>`;
        });
        tbody += '</tr>';
    });
    tbody += '</tbody>';

    // Set table HTML
    table.innerHTML = thead + tbody;
    container.appendChild(table);
}

// Create weight strength plot
function createWeightStrengthPlot(data, container) {
    if (!data || data.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No weight strength data available</div>';
        return;
    }

    // Get unique series from data
    const seriesNames = Object.keys(data[0]).filter(k => k !== 'index');

    // Prepare traces for each series
    const traces = seriesNames.map(series => {
        return {
            x: data.map(d => d.index),
            y: data.map(d => d[series]),
            mode: 'markers',
            type: 'scatter',
            name: series
        };
    });

    // Plot layout
    const layout = {
        title: 'Worker Strength vs. Weight Relationship',
        xaxis: {
            title: 'Worker Strength'
        },
        yaxis: {
            title: 'Weight'
        },
        hovermode: 'closest',
        legend: {
            x: 0,
            y: 1
        }
    };

    // Create plot
    Plotly.newPlot(container, traces, layout);
}
