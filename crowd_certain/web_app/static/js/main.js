/**
 * Crowd-Certain Web Application
 * Main JavaScript file for handling user interactions and visualizations
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize form with default values
    initializeForm();
    
    // Add event listener for form submission
    document.getElementById('simulation-form').addEventListener('submit', function(e) {
        e.preventDefault();
        runSimulation();
    });
    
    // Add event listeners for tab navigation
    setupTabNavigation();
});

/**
 * Initialize the form with default values
 */
function initializeForm() {
    // Select first dataset by default
    const datasetsSelect = document.getElementById('datasets');
    if (datasetsSelect.options.length > 0) {
        datasetsSelect.options[0].selected = true;
    }
    
    // Select default uncertainty technique
    const uncertaintySelect = document.getElementById('uncertainty-techniques');
    if (uncertaintySelect.options.length > 0) {
        for (let i = 0; i < uncertaintySelect.options.length; i++) {
            if (uncertaintySelect.options[i].value === 'STD') {
                uncertaintySelect.options[i].selected = true;
                break;
            }
        }
    }
    
    // Select default consistency technique
    const consistencySelect = document.getElementById('consistency-techniques');
    if (consistencySelect.options.length > 0) {
        for (let i = 0; i < consistencySelect.options.length; i++) {
            if (consistencySelect.options[i].value === 'ONE_MINUS_UNCERTAINTY') {
                consistencySelect.options[i].selected = true;
                break;
            }
        }
    }
}

/**
 * Setup tab navigation
 */
function setupTabNavigation() {
    // Get all tab links
    const tabLinks = document.querySelectorAll('.nav-link');
    
    // Add click event listeners
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target tab
            const targetId = this.getAttribute('href');
            
            // Find the parent tab container
            const tabContainer = this.closest('.nav-tabs');
            const tabContentContainer = tabContainer.nextElementSibling;
            
            // Remove active class from all tabs in this container
            tabContainer.querySelectorAll('.nav-link').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Hide all tab panes in this container
            tabContentContainer.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('show', 'active');
            });
            
            // Show the target tab pane
            const targetPane = document.querySelector(targetId);
            if (targetPane) {
                targetPane.classList.add('show', 'active');
            }
        });
    });
}

/**
 * Run the simulation with the current form values
 */
function runSimulation() {
    // Show progress indicator
    document.getElementById('simulation-status').textContent = 'Running simulation...';
    document.getElementById('simulation-progress').classList.remove('d-none');
    
    // Get form values
    const formData = {
        datasets: getSelectedValues('datasets'),
        workers_min: document.getElementById('workers-min').value,
        workers_max: document.getElementById('workers-max').value,
        quality_min: document.getElementById('quality-min').value,
        quality_max: document.getElementById('quality-max').value,
        seeds: document.getElementById('seeds').value,
        uncertainty_techniques: getSelectedValues('uncertainty-techniques'),
        consistency_techniques: getSelectedValues('consistency-techniques')
    };
    
    // Send request to server
    fetch('/run_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        // Hide progress indicator
        document.getElementById('simulation-progress').classList.add('d-none');
        
        if (data.error) {
            // Show error message
            document.getElementById('simulation-status').textContent = 'Error: ' + data.error;
            document.getElementById('simulation-status').classList.remove('alert-info');
            document.getElementById('simulation-status').classList.add('alert-danger');
        } else {
            // Show success message
            document.getElementById('simulation-status').textContent = 'Simulation completed successfully!';
            document.getElementById('simulation-status').classList.remove('alert-info', 'alert-danger');
            document.getElementById('simulation-status').classList.add('alert-success');
            
            // Display results
            displayResults(data);
        }
    })
    .catch(error => {
        // Hide progress indicator
        document.getElementById('simulation-progress').classList.add('d-none');
        
        // Show error message
        document.getElementById('simulation-status').textContent = 'Error: ' + error.message;
        document.getElementById('simulation-status').classList.remove('alert-info');
        document.getElementById('simulation-status').classList.add('alert-danger');
    });
}

/**
 * Get selected values from a multi-select element
 */
function getSelectedValues(elementId) {
    const select = document.getElementById(elementId);
    const result = [];
    
    for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].selected) {
            result.push(select.options[i].value);
        }
    }
    
    return result;
}

/**
 * Display simulation results
 */
function displayResults(data) {
    // Show results sections
    document.getElementById('results').classList.remove('d-none');
    document.getElementById('worker-analysis').classList.remove('d-none');
    
    // Handle single dataset results
    if (data.proposed_metrics) {
        displaySingleDatasetResults(data);
    } 
    // Handle multiple dataset results
    else {
        const datasetName = Object.keys(data)[0];
        displaySingleDatasetResults(data[datasetName]);
    }
}

/**
 * Display results for a single dataset
 */
function displaySingleDatasetResults(data) {
    // Display metrics
    displayMetricsTable('proposed-metrics', data.proposed_metrics);
    displayMetricsTable('benchmark-metrics', data.benchmark_metrics);
    
    // Display confidence scores if available
    if (data.confidence_scores) {
        displayConfidenceScores(data.confidence_scores);
    }
    
    // Display worker strengths if available
    if (data.worker_strengths) {
        displayWorkerStrengths(data.worker_strengths);
    }
    
    // Display worker weights if available
    if (data.worker_weights) {
        displayWorkerWeights(data.worker_weights);
    }
    
    // Display strength vs weight relationship if both are available
    if (data.worker_strengths && data.worker_weights && data.worker_weights.proposed) {
        displayStrengthWeightRelation(data.worker_strengths, data.worker_weights.proposed);
    }
}

/**
 * Display metrics in a table format
 */
function displayMetricsTable(elementId, metrics) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';
    
    // Create table
    const table = document.createElement('table');
    table.className = 'table table-striped table-sm';
    
    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    const headerCell1 = document.createElement('th');
    headerCell1.textContent = 'Metric';
    headerRow.appendChild(headerCell1);
    
    const headerCell2 = document.createElement('th');
    headerCell2.textContent = 'Value';
    headerRow.appendChild(headerCell2);
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    
    // Add rows for each metric
    for (const metricName in metrics) {
        const row = document.createElement('tr');
        
        const nameCell = document.createElement('td');
        nameCell.textContent = metricName;
        row.appendChild(nameCell);
        
        const valueCell = document.createElement('td');
        
        // Handle different data types
        if (typeof metrics[metricName] === 'object') {
            // For nested objects (like accuracy for different methods)
            const subTable = document.createElement('table');
            subTable.className = 'table table-sm';
            
            for (const subKey in metrics[metricName]) {
                const subRow = document.createElement('tr');
                
                const subNameCell = document.createElement('td');
                subNameCell.textContent = subKey;
                subRow.appendChild(subNameCell);
                
                const subValueCell = document.createElement('td');
                subValueCell.textContent = metrics[metricName][subKey].toFixed(4);
                subRow.appendChild(subValueCell);
                
                subTable.appendChild(subRow);
            }
            
            valueCell.appendChild(subTable);
        } else {
            // For simple numeric values
            valueCell.textContent = typeof metrics[metricName] === 'number' 
                ? metrics[metricName].toFixed(4) 
                : metrics[metricName];
        }
        
        row.appendChild(valueCell);
        tbody.appendChild(row);
    }
    
    table.appendChild(tbody);
    container.appendChild(table);
}

/**
 * Display confidence scores visualization
 */
function displayConfidenceScores(confidenceScores) {
    // Get the first confidence score method
    const method = Object.keys(confidenceScores)[0];
    const scores = confidenceScores[method];
    
    // Extract data for plotting
    const data = [];
    
    // Convert scores to array format for plotting
    for (const key in scores) {
        if (scores[key] && typeof scores[key] === 'object') {
            const values = Object.values(scores[key]);
            
            // Create histogram data
            const trace = {
                x: values,
                type: 'histogram',
                name: key,
                opacity: 0.7,
                marker: {
                    line: {
                        color: 'white',
                        width: 1
                    }
                }
            };
            
            data.push(trace);
        }
    }
    
    // Plot layout
    const layout = {
        title: 'Confidence Score Distribution',
        xaxis: {title: 'Confidence Score'},
        yaxis: {title: 'Frequency'},
        barmode: 'overlay',
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1
        }
    };
    
    // Create plot
    Plotly.newPlot('confidence-scores-plot', data, layout);
}

/**
 * Display worker strengths visualization
 */
function displayWorkerStrengths(workerStrengths) {
    // Extract data for plotting
    const workers = Object.keys(workerStrengths);
    const strengths = Object.values(workerStrengths);
    
    // Sort data by strength
    const combined = workers.map((worker, index) => ({
        worker: worker,
        strength: strengths[index]
    }));
    
    combined.sort((a, b) => b.strength - a.strength);
    
    const sortedWorkers = combined.map(item => item.worker);
    const sortedStrengths = combined.map(item => item.strength);
    
    // Create bar chart
    const data = [{
        x: sortedWorkers,
        y: sortedStrengths,
        type: 'bar',
        marker: {
            color: 'rgba(75, 139, 190, 0.7)',
            line: {
                color: 'rgba(75, 139, 190, 1.0)',
                width: 1
            }
        }
    }];
    
    // Plot layout
    const layout = {
        title: 'Worker Strength',
        xaxis: {
            title: 'Worker ID',
            tickangle: -45
        },
        yaxis: {
            title: 'Strength'
        }
    };
    
    // Create plot
    Plotly.newPlot('worker-strength-plot', data, layout);
}

/**
 * Display worker weights visualization
 */
function displayWorkerWeights(workerWeights) {
    // Extract data for plotting
    const data = [];
    
    // Add trace for each weight type
    for (const weightType in workerWeights) {
        if (workerWeights[weightType] && Object.keys(workerWeights[weightType]).length > 0) {
            const workers = Object.keys(workerWeights[weightType]);
            const weights = Object.values(workerWeights[weightType]);
            
            // Sort data by worker ID
            const combined = workers.map((worker, index) => ({
                worker: worker,
                weight: weights[index]
            }));
            
            combined.sort((a, b) => a.worker.localeCompare(b.worker));
            
            const sortedWorkers = combined.map(item => item.worker);
            const sortedWeights = combined.map(item => item.weight);
            
            // Create trace
            const trace = {
                x: sortedWorkers,
                y: sortedWeights,
                type: 'bar',
                name: weightType.toUpperCase(),
                opacity: 0.7
            };
            
            data.push(trace);
        }
    }
    
    // Plot layout
    const layout = {
        title: 'Worker Weights by Method',
        xaxis: {
            title: 'Worker ID',
            tickangle: -45
        },
        yaxis: {
            title: 'Weight'
        },
        barmode: 'group',
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1
        }
    };
    
    // Create plot
    Plotly.newPlot('worker-weights-plot', data, layout);
}

/**
 * Display strength vs weight relationship
 */
function displayStrengthWeightRelation(workerStrengths, workerWeights) {
    // Extract data for plotting
    const data = [];
    
    // Combine strength and weight data
    const combined = [];
    
    for (const worker in workerStrengths) {
        if (workerWeights[worker] !== undefined) {
            combined.push({
                worker: worker,
                strength: workerStrengths[worker],
                weight: workerWeights[worker]
            });
        }
    }
    
    // Extract arrays for scatter plot
    const strengths = combined.map(item => item.strength);
    const weights = combined.map(item => item.weight);
    const workers = combined.map(item => item.worker);
    
    // Create scatter plot
    const trace = {
        x: strengths,
        y: weights,
        mode: 'markers+text',
        type: 'scatter',
        text: workers,
        textposition: 'top center',
        marker: {
            size: 10,
            color: 'rgba(75, 139, 190, 0.7)',
            line: {
                color: 'rgba(75, 139, 190, 1.0)',
                width: 1
            }
        }
    };
    
    data.push(trace);
    
    // Add trendline if there are enough points
    if (combined.length > 2) {
        // Calculate linear regression
        const result = linearRegression(strengths, weights);
        
        // Create trendline
        const xRange = [Math.min(...strengths), Math.max(...strengths)];
        const yRange = [
            result.slope * xRange[0] + result.intercept,
            result.slope * xRange[1] + result.intercept
        ];
        
        const trendline = {
            x: xRange,
            y: yRange,
            mode: 'lines',
            type: 'scatter',
            name: `Trendline (y = ${result.slope.toFixed(2)}x + ${result.intercept.toFixed(2)})`,
            line: {
                color: 'red',
                width: 2
            }
        };
        
        data.push(trendline);
    }
    
    // Plot layout
    const layout = {
        title: 'Worker Strength vs Weight Relationship',
        xaxis: {
            title: 'Worker Strength'
        },
        yaxis: {
            title: 'Worker Weight'
        },
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1
        }
    };
    
    // Create plot
    Plotly.newPlot('strength-weight-plot', data, layout);
}

/**
 * Calculate linear regression
 */
function linearRegression(x, y) {
    const n = x.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;
    
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumXX += x[i] * x[i];
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return {
        slope: slope,
        intercept: intercept
    };
}
