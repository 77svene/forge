/**
 * SOVEREIGN Interactive Documentation System
 * Transforms static LlamaFactory docs into a Jupyter-like environment
 * with runnable code, real-time visualization, and collaborative features
 */

(function() {
    'use strict';
    
    // ============================================
    // CONFIGURATION & CONSTANTS
    // ============================================
    const CONFIG = {
        jupyterLiteUrl: 'https://jupyterlite.github.io/demo/lab/index.html',
        gpuMemoryApi: '/api/gpu-memory',
        trainingMetricsApi: '/api/training-metrics',
        annotationEndpoint: '/api/annotations',
        modelPlaygroundEndpoint: '/api/playground',
        maxCodeExecutionTime: 30000,
        chartUpdateInterval: 2000,
        storageKeys: {
            annotations: 'llamafactory_annotations',
            userPreferences: 'llamafactory_user_prefs',
            executionHistory: 'llamafactory_exec_history'
        }
    };

    // ============================================
    // UTILITY FUNCTIONS
    // ============================================
    const Utils = {
        debounce: (func, wait) => {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        throttle: (func, limit) => {
            let inThrottle;
            return function(...args) {
                if (!inThrottle) {
                    func.apply(this, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        generateId: () => {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },

        formatBytes: (bytes, decimals = 2) => {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
        },

        formatTime: (seconds) => {
            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        },

        escapeHtml: (unsafe) => {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        },

        loadScript: (src, async = true) => {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.async = async;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        },

        loadStylesheet: (href) => {
            return new Promise((resolve, reject) => {
                const link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = href;
                link.onload = resolve;
                link.onerror = reject;
                document.head.appendChild(link);
            });
        }
    };

    // ============================================
    // JUPYTERLITE INTEGRATION
    // ============================================
    class JupyterLiteIntegration {
        constructor() {
            this.kernel = null;
            this.isReady = false;
            this.executionQueue = [];
            this.initialize();
        }

        async initialize() {
            try {
                // Load Pyodide for browser-based Python execution
                await Utils.loadScript('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');
                
                // Initialize Pyodide
                this.pyodide = await loadPyodide();
                
                // Install common packages for LlamaFactory
                await this.pyodide.loadPackage(['micropip']);
                await this.pyodide.runPythonAsync(`
                    import micropip
                    await micropip.install(['numpy', 'pandas', 'matplotlib'])
                `);
                
                this.isReady = true;
                this.processQueue();
                
                console.log('JupyterLite integration ready');
            } catch (error) {
                console.error('Failed to initialize JupyterLite:', error);
                this.showFallbackMessage();
            }
        }

        processQueue() {
            while (this.executionQueue.length > 0) {
                const task = this.executionQueue.shift();
                this.executeCode(task.code, task.element, task.options);
            }
        }

        async executeCode(code, element, options = {}) {
            if (!this.isReady) {
                this.executionQueue.push({ code, element, options });
                return;
            }

            const outputElement = element.querySelector('.code-output') || 
                                 this.createOutputElement(element);
            
            try {
                outputElement.innerHTML = '<div class="execution-status running">Executing...</div>';
                outputElement.classList.add('active');
                
                // Capture stdout/stderr
                const result = await this.pyodide.runPythonAsync(`
                    import sys
                    from io import StringIO
                    
                    # Redirect stdout/stderr
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = mystdout = StringIO()
                    sys.stderr = mystderr = StringIO()
                    
                    try:
                        ${code}
                        output = mystdout.getvalue()
                        error = mystderr.getvalue()
                        result = {'output': output, 'error': error}
                    except Exception as e:
                        result = {'output': '', 'error': str(e)}
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    result
                `);
                
                this.displayResult(result, outputElement, options);
                this.saveToHistory(code, result, options);
                
            } catch (error) {
                outputElement.innerHTML = `
                    <div class="execution-error">
                        <strong>Execution Error:</strong>
                        <pre>${Utils.escapeHtml(error.message)}</pre>
                    </div>
                `;
            }
        }

        displayResult(result, outputElement, options) {
            let html = '';
            
            if (result.output) {
                html += `<div class="output-text"><pre>${Utils.escapeHtml(result.output)}</pre></div>`;
            }
            
            if (result.error) {
                html += `<div class="error-text"><pre>${Utils.escapeHtml(result.error)}</pre></div>`;
            }
            
            if (options.visualize && result.output) {
                html += this.createVisualization(result.output, options);
            }
            
            if (!html) {
                html = '<div class="no-output">No output</div>';
            }
            
            outputElement.innerHTML = html;
            outputElement.classList.remove('running');
        }

        createVisualization(output, options) {
            // Parse output for visualization data
            const chartId = `chart-${Utils.generateId()}`;
            
            setTimeout(() => {
                if (window.Chart) {
                    this.renderChart(chartId, output, options);
                }
            }, 100);
            
            return `
                <div class="visualization-container">
                    <canvas id="${chartId}" width="400" height="200"></canvas>
                </div>
            `;
        }

        renderChart(chartId, output, options) {
            const ctx = document.getElementById(chartId);
            if (!ctx) return;
            
            // Simple parsing for demonstration
            const data = this.parseOutputForChart(output);
            
            new Chart(ctx, {
                type: options.chartType || 'line',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Output Visualization',
                        data: data.values,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Code Output Visualization'
                        }
                    }
                }
            });
        }

        parseOutputForChart(output) {
            // Simple parser for numeric outputs
            const lines = output.trim().split('\n');
            const values = [];
            const labels = [];
            
            lines.forEach((line, index) => {
                const num = parseFloat(line);
                if (!isNaN(num)) {
                    values.push(num);
                    labels.push(`Point ${index + 1}`);
                }
            });
            
            return { values, labels };
        }

        createOutputElement(element) {
            const outputDiv = document.createElement('div');
            outputDiv.className = 'code-output';
            element.appendChild(outputDiv);
            return outputDiv;
        }

        showFallbackMessage() {
            document.querySelectorAll('.interactive-code').forEach(element => {
                const fallback = document.createElement('div');
                fallback.className = 'jupyterlite-fallback';
                fallback.innerHTML = `
                    <p>Interactive code execution is temporarily unavailable.</p>
                    <p>Please check your internet connection or try again later.</p>
                `;
                element.appendChild(fallback);
            });
        }

        saveToHistory(code, result, options) {
            const history = JSON.parse(localStorage.getItem(CONFIG.storageKeys.executionHistory) || '[]');
            history.unshift({
                id: Utils.generateId(),
                timestamp: new Date().toISOString(),
                code: code.substring(0, 200),
                success: !result.error,
                options
            });
            
            // Keep only last 50 executions
            if (history.length > 50) {
                history.pop();
            }
            
            localStorage.setItem(CONFIG.storageKeys.executionHistory, JSON.stringify(history));
        }
    }

    // ============================================
    // PERFORMANCE VISUALIZATION
    // ============================================
    class PerformanceVisualizer {
        constructor() {
            this.charts = {};
            this.metrics = {
                gpuMemory: [],
                trainingLoss: [],
                learningRate: [],
                throughput: []
            };
            this.isMonitoring = false;
            this.initialize();
        }

        initialize() {
            this.createMonitoringUI();
            this.setupEventListeners();
        }

        createMonitoringUI() {
            const monitoringPanel = document.createElement('div');
            monitoringPanel.id = 'performance-monitoring-panel';
            monitoringPanel.className = 'performance-panel hidden';
            monitoringPanel.innerHTML = `
                <div class="panel-header">
                    <h3>Real-time Performance Monitoring</h3>
                    <button class="panel-toggle" aria-label="Toggle panel">
                        <svg viewBox="0 0 24 24" width="24" height="24">
                            <path fill="currentColor" d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z"/>
                        </svg>
                    </button>
                </div>
                <div class="panel-content">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h4>GPU Memory</h4>
                            <div class="metric-value" id="gpu-memory-value">-- GB</div>
                            <canvas id="gpu-memory-chart" height="100"></canvas>
                        </div>
                        <div class="metric-card">
                            <h4>Training Loss</h4>
                            <div class="metric-value" id="training-loss-value">--</div>
                            <canvas id="training-loss-chart" height="100"></canvas>
                        </div>
                        <div class="metric-card">
                            <h4>Learning Rate</h4>
                            <div class="metric-value" id="learning-rate-value">--</div>
                            <canvas id="learning-rate-chart" height="100"></canvas>
                        </div>
                        <div class="metric-card">
                            <h4>Throughput</h4>
                            <div class="metric-value" id="throughput-value">-- samples/s</div>
                            <canvas id="throughput-chart" height="100"></canvas>
                        </div>
                    </div>
                    <div class="controls">
                        <button id="start-monitoring" class="btn btn-primary">Start Monitoring</button>
                        <button id="stop-monitoring" class="btn btn-secondary" disabled>Stop Monitoring</button>
                        <button id="export-metrics" class="btn btn-outline">Export Data</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(monitoringPanel);
            
            // Add toggle button to documentation
            const toggleButton = document.createElement('button');
            toggleButton.id = 'performance-toggle';
            toggleButton.className = 'performance-toggle-btn';
            toggleButton.innerHTML = '📊 Performance';
            toggleButton.setAttribute('aria-label', 'Toggle performance monitoring');
            document.body.appendChild(toggleButton);
        }

        setupEventListeners() {
            document.getElementById('performance-toggle').addEventListener('click', () => {
                this.togglePanel();
            });
            
            document.getElementById('start-monitoring').addEventListener('click', () => {
                this.startMonitoring();
            });
            
            document.getElementById('stop-monitoring').addEventListener('click', () => {
                this.stopMonitoring();
            });
            
            document.getElementById('export-metrics').addEventListener('click', () => {
                this.exportMetrics();
            });
            
            document.querySelector('.panel-toggle').addEventListener('click', () => {
                this.togglePanel();
            });
        }

        togglePanel() {
            const panel = document.getElementById('performance-monitoring-panel');
            panel.classList.toggle('hidden');
        }

        async startMonitoring() {
            if (this.isMonitoring) return;
            
            this.isMonitoring = true;
            document.getElementById('start-monitoring').disabled = true;
            document.getElementById('stop-monitoring').disabled = false;
            
            // Initialize charts
            this.initializeCharts();
            
            // Start monitoring loop
            this.monitoringInterval = setInterval(() => {
                this.updateMetrics();
            }, CONFIG.chartUpdateInterval);
        }

        stopMonitoring() {
            this.isMonitoring = false;
            clearInterval(this.monitoringInterval);
            
            document.getElementById('start-monitoring').disabled = false;
            document.getElementById('stop-monitoring').disabled = true;
        }

        initializeCharts() {
            const chartConfig = {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            };
            
            // GPU Memory Chart
            this.charts.gpuMemory = new Chart(
                document.getElementById('gpu-memory-chart'),
                {
                    ...chartConfig,
                    data: {
                        labels: [],
                        datasets: [{
                            data: [],
                            borderColor: '#4e79a7',
                            backgroundColor: 'rgba(78, 121, 167, 0.1)',
                            fill: true
                        }]
                    }
                }
            );
            
            // Training Loss Chart
            this.charts.trainingLoss = new Chart(
                document.getElementById('training-loss-chart'),
                {
                    ...chartConfig,
                    data: {
                        labels: [],
                        datasets: [{
                            data: [],
                            borderColor: '#e15759',
                            backgroundColor: 'rgba(225, 87, 89, 0.1)',
                            fill: true
                        }]
                    }
                }
            );
            
            // Learning Rate Chart
            this.charts.learningRate = new Chart(
                document.getElementById('learning-rate-chart'),
                {
                    ...chartConfig,
                    data: {
                        labels: [],
                        datasets: [{
                            data: [],
                            borderColor: '#76b7b2',
                            backgroundColor: 'rgba(118, 183, 178, 0.1)',
                            fill: true
                        }]
                    }
                }
            );
            
            // Throughput Chart
            this.charts.throughput = new Chart(
                document.getElementById('throughput-chart'),
                {
                    ...chartConfig,
                    data: {
                        labels: [],
                        datasets: [{
                            data: [],
                            borderColor: '#f28e2c',
                            backgroundColor: 'rgba(242, 142, 44, 0.1)',
                            fill: true
                        }]
                    }
                }
            );
        }

        async updateMetrics() {
            try {
                // Simulate API calls - in production, these would be real endpoints
                const [gpuMemory, trainingLoss, learningRate, throughput] = await Promise.all([
                    this.fetchGPUMemory(),
                    this.fetchTrainingLoss(),
                    this.fetchLearningRate(),
                    this.fetchThroughput()
                ]);
                
                this.updateChart('gpuMemory', gpuMemory);
                this.updateChart('trainingLoss', trainingLoss);
                this.updateChart('learningRate', learningRate);
                this.updateChart('throughput', throughput);
                
                this.updateDisplayValues(gpuMemory, trainingLoss, learningRate, throughput);
                
            } catch (error) {
                console.error('Failed to update metrics:', error);
            }
        }

        async fetchGPUMemory() {
            // Simulated GPU memory usage (in GB)
            return Math.random() * 24 + 8; // 8-32 GB range
        }

        async fetchTrainingLoss() {
            // Simulated training loss
            return Math.random() * 2 + 0.5; // 0.5-2.5 range
        }

        async fetchLearningRate() {
            // Simulated learning rate
            return Math.random() * 0.001 + 0.0001; // 0.0001-0.0011 range
        }

        async fetchThroughput() {
            // Simulated throughput (samples per second)
            return Math.random() * 1000 + 500; // 500-1500 samples/s
        }

        updateChart(chartName, value) {
            const chart = this.charts[chartName];
            if (!chart) return;
            
            const now = new Date();
            const timeLabel = `${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`;
            
            // Add new data point
            chart.data.labels.push(timeLabel);
            chart.data.datasets[0].data.push(value);
            
            // Keep only last 20 data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none'); // Update without animation for performance
        }

        updateDisplayValues(gpuMemory, trainingLoss, learningRate, throughput) {
            document.getElementById('gpu-memory-value').textContent = 
                `${gpuMemory.toFixed(2)} GB`;
            document.getElementById('training-loss-value').textContent = 
                trainingLoss.toFixed(4);
            document.getElementById('learning-rate-value').textContent = 
                learningRate.toExponential(2);
            document.getElementById('throughput-value').textContent = 
                `${throughput.toFixed(0)} samples/s`;
        }

        exportMetrics() {
            const data = {
                timestamp: new Date().toISOString(),
                metrics: this.metrics,
                charts: Object.keys(this.charts).reduce((acc, key) => {
                    acc[key] = {
                        labels: this.charts[key].data.labels,
                        data: this.charts[key].data.datasets[0].data
                    };
                    return acc;
                }, {})
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `llamafactory-metrics-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    // ============================================
    // COLLABORATIVE ANNOTATION SYSTEM
    // ============================================
    class CollaborativeAnnotations {
        constructor() {
            this.annotations = this.loadAnnotations();
            this.currentUser = this.getCurrentUser();
            this.selectedRange = null;
            this.initialize();
        }

        initialize() {
            this.createAnnotationUI();
            this.setupEventListeners();
            this.renderAnnotations();
        }

        getCurrentUser() {
            let user = localStorage.getItem('llamafactory_user');
            if (!user) {
                user = {
                    id: Utils.generateId(),
                    name: `User_${Math.random().toString(36).substr(2, 9)}`,
                    color: this.getRandomColor()
                };
                localStorage.setItem('llamafactory_user', JSON.stringify(user));
            }
            return JSON.parse(user);
        }

        getRandomColor() {
            const colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
            ];
            return colors[Math.floor(Math.random() * colors.length)];
        }

        createAnnotationUI() {
            // Create annotation toolbar
            const toolbar = document.createElement('div');
            toolbar.id = 'annotation-toolbar';
            toolbar.className = 'annotation-toolbar hidden';
            toolbar.innerHTML = `
                <button class="annotation-btn" data-action="highlight" title="Highlight text">
                    <svg viewBox="0 0 24 24" width="20" height="20">
                        <path fill="currentColor" d="M12.5,4.5L12.5,4.5C14.5,4.5 16,6 16,8V12C16,14 14.5,15.5 12.5,15.5H11.5V19H10V15.5H7.5C5.5,15.5 4,14 4,12V8C4,6 5.5,4.5 7.5,4.5H12.5M7.5,6C6.7,6 6,6.7 6,8V12C6,13.3 6.7,14 7.5,14H12.5C13.3,14 14,13.3 14,12V8C14,6.7 13.3,6 12.5,6H7.5Z"/>
                    </svg>
                </button>
                <button class="annotation-btn" data-action="comment" title="Add comment">
                    <svg viewBox="0 0 24 24" width="20" height="20">
                        <path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20"/>
                    </svg>
                </button>
                <button class="annotation-btn" data-action="suggest" title="Suggest edit">
                    <svg viewBox="0 0 24 24" width="20" height="20">
                        <path fill="currentColor" d="M20.71,7.04C21.1,6.65 21.1,6 20.71,5.63L18.37,3.29C18,2.9 17.35,2.9 16.96,3.29L15.12,5.12L18.87,8.87M3,17.25V21H6.75L17.81,9.93L14.06,6.18L3,17.25Z"/>
                    </svg>
                </button>
            `;
            
            // Create annotation sidebar
            const sidebar = document.createElement('div');
            sidebar.id = 'annotation-sidebar';
            sidebar.className = 'annotation-sidebar hidden';
            sidebar.innerHTML = `
                <div class="sidebar-header">
                    <h3>Annotations</h3>
                    <button class="close-sidebar">&times;</button>
                </div>
                <div class="sidebar-content">
                    <div class="annotation-list" id="annotation-list"></div>
                </div>
                <div class="sidebar-footer">
                    <button class="btn btn-primary" id="add-annotation-btn">Add Annotation</button>
                </div>
            `;
            
            // Create annotation modal
            const modal = document.createElement('div');
            modal.id = 'annotation-modal';
            modal.className = 'annotation-modal hidden';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h4>Add Annotation</h4>
                        <button class="close-modal">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="form-group">
                            <label>Type:</label>
                            <select id="annotation-type">
                                <option value="highlight">Highlight</option>
                                <option value="comment">Comment</option>
                                <option value="suggestion">Suggestion</option>
                                <option value="question">Question</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Content:</label>
                            <textarea id="annotation-content" rows="4" placeholder="Enter your annotation..."></textarea>
                        </div>
                        <div class="form-group">
                            <label>Tags:</label>
                            <input type="text" id="annotation-tags" placeholder="bug, improvement, question">
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" id="cancel-annotation">Cancel</button>
                        <button class="btn btn-primary" id="save-annotation">Save</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(toolbar);
            document.body.appendChild(sidebar);
            document.body.appendChild(modal);
        }

        setupEventListeners() {
            // Text selection for annotations
            document.addEventListener('mouseup', (e) => {
                const selection = window.getSelection();
                if (selection.toString().trim().length > 0) {
                    this.showAnnotationToolbar(e, selection);
                } else {
                    this.hideAnnotationToolbar();
                }
            });
            
            // Toolbar actions
            document.querySelectorAll('.annotation-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const action = e.currentTarget.dataset.action;
                    this.handleAnnotationAction(action);
                });
            });
            
            // Sidebar controls
            document.getElementById('add-annotation-btn').addEventListener('click', () => {
                this.showAnnotationModal();
            });
            
            document.querySelector('.close-sidebar').addEventListener('click', () => {
                this.hideAnnotationSidebar();
            });
            
            // Modal controls
            document.querySelector('.close-modal').addEventListener('click', () => {
                this.hideAnnotationModal();
            });
            
            document.getElementById('cancel-annotation').addEventListener('click', () => {
                this.hideAnnotationModal();
            });
            
            document.getElementById('save-annotation').addEventListener('click', () => {
                this.saveAnnotation();
            });
            
            // Toggle sidebar button
            const toggleBtn = document.createElement('button');
            toggleBtn.id = 'annotation-toggle';
            toggleBtn.className = 'annotation-toggle-btn';
            toggleBtn.innerHTML = '📝 Annotations';
            toggleBtn.addEventListener('click', () => {
                this.toggleAnnotationSidebar();
            });
            document.body.appendChild(toggleBtn);
        }

        showAnnotationToolbar(event, selection) {
            const toolbar = document.getElementById('annotation-toolbar');
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            
            toolbar.style.top = `${rect.top + window.scrollY - 40}px`;
            toolbar.style.left = `${rect.left + window.scrollX}px`;
            toolbar.classList.remove('hidden');
            
            this.selectedRange = {
                text: selection.toString(),
                range: range.cloneRange(),
                rect: rect
            };
        }

        hideAnnotationToolbar() {
            document.getElementById('annotation-toolbar').classList.add('hidden');
            this.selectedRange = null;
        }

        handleAnnotationAction(action) {
            if (!this.selectedRange) return;
            
            switch (action) {
                case 'highlight':
                    this.createHighlight();
                    break;
                case 'comment':
                    this.showAnnotationModal('comment');
                    break;
                case 'suggest':
                    this.showAnnotationModal('suggestion');
                    break;
            }
            
            this.hideAnnotationToolbar();
        }

        createHighlight() {
            const highlight = document.createElement('mark');
            highlight.className = 'user-highlight';
            highlight.style.backgroundColor = this.currentUser.color + '40';
            highlight.dataset.userId = this.currentUser.id;
            highlight.dataset.timestamp = new Date().toISOString();
            
            try {
                this.selectedRange.range.surroundContents(highlight);
                
                // Save annotation
                const annotation = {
                    id: Utils.generateId(),
                    type: 'highlight',
                    text: this.selectedRange.text,
                    userId: this.currentUser.id,
                    userName: this.currentUser.name,
                    color: this.currentUser.color,
                    timestamp: new Date().toISOString(),
                    page: window.location.pathname,
                    position: this.getElementPosition(highlight)
                };
                
                this.saveAnnotationToStorage(annotation);
                this.renderAnnotation(annotation);
                
            } catch (error) {
                console.error('Failed to create highlight:', error);
            }
        }

        showAnnotationModal(type = 'comment') {
            const modal = document.getElementById('annotation-modal');
            modal.classList.remove('hidden');
            document.getElementById('annotation-type').value = type;
            document.getElementById('annotation-content').focus();
        }

        hideAnnotationModal() {
            document.getElementById('annotation-modal').classList.add('hidden');
            document.getElementById('annotation-content').value = '';
            document.getElementById('annotation-tags').value = '';
        }

        saveAnnotation() {
            const type = document.getElementById('annotation-type').value;
            const content = document.getElementById('annotation-content').value.trim();
            const tags = document.getElementById('annotation-tags').value
                .split(',')
                .map(tag => tag.trim())
                .filter(tag => tag);
            
            if (!content) {
                alert('Please enter annotation content');
                return;
            }
            
            const annotation = {
                id: Utils.generateId(),
                type: type,
                content: content,
                tags: tags,
                selectedText: this.selectedRange ? this.selectedRange.text : '',
                userId: this.currentUser.id,
                userName: this.currentUser.name,
                color: this.currentUser.color,
                timestamp: new Date().toISOString(),
                page: window.location.pathname,
                resolved: false
            };
            
            this.saveAnnotationToStorage(annotation);
            this.renderAnnotation(annotation);
            this.hideAnnotationModal();
        }

        saveAnnotationToStorage(annotation) {
            this.annotations.push(annotation);
            localStorage.setItem(
                CONFIG.storageKeys.annotations,
                JSON.stringify(this.annotations)
            );
        }

        loadAnnotations() {
            const stored = localStorage.getItem(CONFIG.storageKeys.annotations);
            return stored ? JSON.parse(stored) : [];
        }

        renderAnnotations() {
            const list = document.getElementById('annotation-list');
            list.innerHTML = '';
            
            const pageAnnotations = this.annotations.filter(
                ann => ann.page === window.location.pathname
            );
            
            if (pageAnnotations.length === 0) {
                list.innerHTML = '<div class="no-annotations">No annotations yet</div>';
                return;
            }
            
            pageAnnotations.forEach(annotation => {
                this.renderAnnotationItem(annotation, list);
            });
        }

        renderAnnotationItem(annotation, container) {
            const item = document.createElement('div');
            item.className = `annotation-item ${annotation.type}`;
            item.dataset.id = annotation.id;
            
            item.innerHTML = `
                <div class="annotation-header">
                    <span class="annotation-user" style="color: ${annotation.color}">
                        ${Utils.escapeHtml(annotation.userName)}
                    </span>
                    <span class="annotation-time">
                        ${new Date(annotation.timestamp).toLocaleString()}
                    </span>
                    ${annotation.resolved ? '<span class="annotation-resolved">✓ Resolved</span>' : ''}
                </div>
                <div class="annotation-content">
                    ${annotation.selectedText ? 
                        `<div class="selected-text">"${Utils.escapeHtml(annotation.selectedText)}"</div>` : 
                        ''}
                    ${Utils.escapeHtml(annotation.content)}
                </div>
                ${annotation.tags && annotation.tags.length > 0 ? 
                    `<div class="annotation-tags">
                        ${annotation.tags.map(tag => `<span class="tag">${Utils.escapeHtml(tag)}</span>`).join('')}
                    </div>` : 
                    ''}
                <div class="annotation-actions">
                    <button class="btn-resolve" data-id="${annotation.id}">
                        ${annotation.resolved ? 'Reopen' : 'Resolve'}
                    </button>
                    <button class="btn-delete" data-id="${annotation.id}">Delete</button>
                </div>
            `;
            
            container.appendChild(item);
            
            // Add event listeners
            item.querySelector('.btn-resolve').addEventListener('click', (e) => {
                this.toggleResolve(e.target.dataset.id);
            });
            
            item.querySelector('.btn-delete').addEventListener('click', (e) => {
                this.deleteAnnotation(e.target.dataset.id);
            });
        }

        renderAnnotation(annotation) {
            const list = document.getElementById('annotation-list');
            this.renderAnnotationItem(annotation, list);
        }

        toggleAnnotationSidebar() {
            const sidebar = document.getElementById('annotation-sidebar');
            sidebar.classList.toggle('hidden');
            if (!sidebar.classList.contains('hidden')) {
                this.renderAnnotations();
            }
        }

        hideAnnotationSidebar() {
            document.getElementById('annotation-sidebar').classList.add('hidden');
        }

        toggleResolve(annotationId) {
            const annotation = this.annotations.find(a => a.id === annotationId);
            if (annotation) {
                annotation.resolved = !annotation.resolved;
                localStorage.setItem(
                    CONFIG.storageKeys.annotations,
                    JSON.stringify(this.annotations)
                );
                this.renderAnnotations();
            }
        }

        deleteAnnotation(annotationId) {
            if (!confirm('Are you sure you want to delete this annotation?')) return;
            
            this.annotations = this.annotations.filter(a => a.id !== annotationId);
            localStorage.setItem(
                CONFIG.storageKeys.annotations,
                JSON.stringify(this.annotations)
            );
            this.renderAnnotations();
        }

        getElementPosition(element) {
            const rect = element.getBoundingClientRect();
            return {
                top: rect.top + window.scrollY,
                left: rect.left + window.scrollX,
                width: rect.width,
                height: rect.height
            };
        }
    }

    // ============================================
    // INTERACTIVE MODEL PLAYGROUND
    // ============================================
    class ModelPlayground {
        constructor() {
            this.models = [];
            this.currentModel = null;
            this.conversation = [];
            this.initialize();
        }

        initialize() {
            this.createPlaygroundUI();
            this.loadModels();
            this.setupEventListeners();
        }

        createPlaygroundUI() {
            const playground = document.createElement('div');
            playground.id = 'model-playground';
            playground.className = 'model-playground hidden';
            playground.innerHTML = `
                <div class="playground-header">
                    <h3>Model Playground</h3>
                    <div class="playground-controls">
                        <select id="model-select">
                            <option value="">Select a model...</option>
                        </select>
                        <button class="playground-toggle" aria-label="Toggle playground">
                            <svg viewBox="0 0 24 24" width="24" height="24">
                                <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="playground-content">
                    <div class="chat-container" id="chat-container">
                        <div class="welcome-message">
                            <h4>Welcome to LlamaFactory Model Playground</h4>
                            <p>Select a model and start chatting to test fine-tuned models.</p>
                            <div class="model-info" id="model-info"></div>
                        </div>
                    </div>
                    <div class="input-container">
                        <textarea id="user-input" placeholder="Type your message..." rows="3"></textarea>
                        <div class="input-actions">
                            <button class="btn btn-secondary" id="clear-chat">Clear</button>
                            <button class="btn btn-primary" id="send-message">Send</button>
                        </div>
                    </div>
                    <div class="playground-settings">
                        <div class="setting-group">
                            <label>Temperature:</label>
                            <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.7">
                            <span id="temperature-value">0.7</span>
                        </div>
                        <div class="setting-group">
                            <label>Max Tokens:</label>
                            <input type="number" id="max-tokens" min="1" max="4096" value="512">
                        </div>
                        <div class="setting-group">
                            <label>Top P:</label>
                            <input type="range" id="top-p" min="0" max="1" step="0.05" value="0.9">
                            <span id="top-p-value">0.9</span>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(playground);
            
            // Add toggle button
            const toggleBtn = document.createElement('button');
            toggleBtn.id = 'playground-toggle';
            toggleBtn.className = 'playground-toggle-btn';
            toggleBtn.innerHTML = '🤖 Playground';
            toggleBtn.addEventListener('click', () => {
                this.togglePlayground();
            });
            document.body.appendChild(toggleBtn);
        }

        async loadModels() {
            try {
                // In production, this would fetch from an API
                this.models = [
                    {
                        id: 'llama-7b-chat',
                        name: 'LLaMA 7B Chat',
                        description: 'Fine-tuned for conversational AI',
                        parameters: '7B',
                        type: 'chat'
                    },
                    {
                        id: 'llama-13b-instruct',
                        name: 'LLaMA 13B Instruct',
                        description: 'Instruction-following model',
                        parameters: '13B',
                        type: 'instruct'
                    },
                    {
                        id: 'code-llama-7b',
                        name: 'Code Llama 7B',
                        description: 'Specialized for code generation',
                        parameters: '7B',
                        type: 'code'
                    }
                ];
                
                this.populateModelSelect();
                
            } catch (error) {
                console.error('Failed to load models:', error);
            }
        }

        populateModelSelect() {
            const select = document.getElementById('model-select');
            select.innerHTML = '<option value="">Select a model...</option>';
            
            this.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.name} (${model.parameters})`;
                option.dataset.description = model.description;
                select.appendChild(option);
            });
        }

        setupEventListeners() {
            // Model selection
            document.getElementById('model-select').addEventListener('change', (e) => {
                this.selectModel(e.target.value);
            });
            
            // Send message
            document.getElementById('send-message').addEventListener('click', () => {
                this.sendMessage();
            });
            
            // Enter key to send
            document.getElementById('user-input').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Clear chat
            document.getElementById('clear-chat').addEventListener('click', () => {
                this.clearChat();
            });
            
            // Settings updates
            document.getElementById('temperature').addEventListener('input', (e) => {
                document.getElementById('temperature-value').textContent = e.target.value;
            });
            
            document.getElementById('top-p').addEventListener('input', (e) => {
                document.getElementById('top-p-value').textContent = e.target.value;
            });
            
            // Close playground
            document.querySelector('.playground-toggle').addEventListener('click', () => {
                this.hidePlayground();
            });
        }

        selectModel(modelId) {
            if (!modelId) {
                this.currentModel = null;
                document.getElementById('model-info').innerHTML = '';
                return;
            }
            
            this.currentModel = this.models.find(m => m.id === modelId);
            
            if (this.currentModel) {
                document.getElementById('model-info').innerHTML = `
                    <div class="model-details">
                        <h5>${this.currentModel.name}</h5>
                        <p>${this.currentModel.description}</p>
                        <div class="model-specs">
                            <span class="spec">Parameters: ${this.currentModel.parameters}</span>
                            <span class="spec">Type: ${this.currentModel.type}</span>
                        </div>
                    </div>
                `;
            }
        }

        async sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message || !this.currentModel) {
                alert('Please select a model and enter a message');
                return;
            }
            
            // Add user message to chat
            this.addMessageToChat('user', message);
            input.value = '';
            
            // Show typing indicator
            this.showTypingIndicator();
            
            try {
                // Get settings
                const settings = {
                    temperature: parseFloat(document.getElementById('temperature').value),
                    max_tokens: parseInt(document.getElementById('max-tokens').value),
                    top_p: parseFloat(document.getElementById('top-p').value)
                };
                
                // Simulate API call
                const response = await this.callModelAPI(message, settings);
                
                // Remove typing indicator and add model response
                this.hideTypingIndicator();
                this.addMessageToChat('assistant', response);
                
                // Update conversation history
                this.conversation.push(
                    { role: 'user', content: message },
                    { role: 'assistant', content: response }
                );
                
            } catch (error) {
                this.hideTypingIndicator();
                this.addMessageToChat('error', `Error: ${error.message}`);
            }
        }

        async callModelAPI(message, settings) {
            // Simulate API call - in production, this would call your model endpoint
            return new Promise((resolve) => {
                setTimeout(() => {
                    const responses = {
                        'llama-7b-chat': [
                            "I understand you're asking about: " + message,
                            "That's an interesting question. Let me think about it.",
                            "Based on my training, I would say..."
                        ],
                        'llama-13b-instruct': [
                            "I'll help you with that instruction.",
                            "Following your instructions, here's what I found:",
                            "As per your request..."
                        ],
                        'code-llama-7b': [
                            "Here's the code you requested:",
                            "```python\n# Code solution\nprint('Hello World')\n```",
                            "I can help you write that function."
                        ]
                    };
                    
                    const modelResponses = responses[this.currentModel.id] || responses['llama-7b-chat'];
                    const randomResponse = modelResponses[Math.floor(Math.random() * modelResponses.length)];
                    
                    resolve(randomResponse);
                }, 1000 + Math.random() * 2000);
            });
        }

        addMessageToChat(role, content) {
            const container = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${role}`;
            
            const timestamp = new Date().toLocaleTimeString();
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <span class="message-role">${role === 'user' ? 'You' : this.currentModel?.name || 'Assistant'}</span>
                    <span class="message-time">${timestamp}</span>
                </div>
                <div class="message-content">
                    ${this.formatMessageContent(content)}
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        formatMessageContent(content) {
            // Basic markdown-like formatting
            return content
                .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="$1">$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                .replace(/\*([^*]+)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');
        }

        showTypingIndicator() {
            const container = document.getElementById('chat-container');
            const indicator = document.createElement('div');
            indicator.className = 'typing-indicator';
            indicator.id = 'typing-indicator';
            indicator.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;
            container.appendChild(indicator);
            container.scrollTop = container.scrollHeight;
        }

        hideTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) {
                indicator.remove();
            }
        }

        clearChat() {
            const container = document.getElementById('chat-container');
            container.innerHTML = `
                <div class="welcome-message">
                    <h4>Welcome to LlamaFactory Model Playground</h4>
                    <p>Select a model and start chatting to test fine-tuned models.</p>
                    <div class="model-info" id="model-info"></div>
                </div>
            `;
            this.conversation = [];
            if (this.currentModel) {
                document.getElementById('model-info').innerHTML = `
                    <div class="model-details">
                        <h5>${this.currentModel.name}</h5>
                        <p>${this.currentModel.description}</p>
                    </div>
                `;
            }
        }

        togglePlayground() {
            const playground = document.getElementById('model-playground');
            playground.classList.toggle('hidden');
        }

        hidePlayground() {
            document.getElementById('model-playground').classList.add('hidden');
        }
    }

    // ============================================
    // MAIN INITIALIZATION
    // ============================================
    class InteractiveDocumentation {
        constructor() {
            this.jupyterlite = null;
            this.performance = null;
            this.annotations = null;
            this.playground = null;
            this.initialized = false;
        }

        async initialize() {
            if (this.initialized) return;
            
            console.log('Initializing SOVEREIGN Interactive Documentation...');
            
            try {
                // Load required dependencies
                await this.loadDependencies();
                
                // Initialize components
                this.jupyterlite = new JupyterLiteIntegration();
                this.performance = new PerformanceVisualizer();
                this.annotations = new CollaborativeAnnotations();
                this.playground = new ModelPlayground();
                
                // Transform static code blocks
                this.transformCodeBlocks();
                
                // Add interactive buttons to documentation
                this.addInteractiveButtons();
                
                this.initialized = true;
                console.log('Interactive documentation initialized successfully');
                
            } catch (error) {
                console.error('Failed to initialize interactive documentation:', error);
            }
        }

        async loadDependencies() {
            // Load Chart.js for visualizations
            await Utils.loadScript('https://cdn.jsdelivr.net/npm/chart.js');
            
            // Load styles
            await Utils.loadStylesheet('/_static/css/interactive.css');
        }

        transformCodeBlocks() {
            // Find all code blocks in documentation
            document.querySelectorAll('pre code, .highlight pre').forEach((codeBlock, index) => {
                const container = codeBlock.closest('pre') || codeBlock;
                const wrapper = document.createElement('div');
                wrapper.className = 'interactive-code';
                
                // Wrap the code block
                container.parentNode.insertBefore(wrapper, container);
                wrapper.appendChild(container);
                
                // Add interactive controls
                const controls = document.createElement('div');
                controls.className = 'code-controls';
                controls.innerHTML = `
                    <button class="run-code-btn" data-code-index="${index}">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z"/>
                        </svg>
                        Run
                    </button>
                    <button class="copy-code-btn">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path fill="currentColor" d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/>
                        </svg>
                        Copy
                    </button>
                    <button class="visualize-code-btn" data-code-index="${index}">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                            <path fill="currentColor" d="M16,11.78L20.24,4.45L21.97,5.45L16.74,14.5L10.23,10.75L5.46,19H22V21H2V3H4V17.54L9.5,8L16,11.78Z"/>
                        </svg>
                        Visualize
                    </button>
                `;
                
                wrapper.insertBefore(controls, container);
                
                // Store original code
                wrapper.dataset.originalCode = codeBlock.textContent;
            });
            
            // Add event listeners to new buttons
            this.setupCodeControlEvents();
        }

        setupCodeControlEvents() {
            document.querySelectorAll('.run-code-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const codeIndex = e.target.closest('.run-code-btn').dataset.codeIndex;
                    const codeBlock = document.querySelectorAll('.interactive-code')[codeIndex];
                    const code = codeBlock.dataset.originalCode;
                    
                    if (this.jupyterlite && this.jupyterlite.isReady) {
                        this.jupyterlite.executeCode(code, codeBlock);
                    } else {
                        alert('Code execution environment is not ready yet. Please wait...');
                    }
                });
            });
            
            document.querySelectorAll('.copy-code-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const codeBlock = e.target.closest('.interactive-code');
                    const code = codeBlock.dataset.originalCode;
                    
                    navigator.clipboard.writeText(code).then(() => {
                        const originalText = btn.innerHTML;
                        btn.innerHTML = '✓ Copied!';
                        setTimeout(() => {
                            btn.innerHTML = originalText;
                        }, 2000);
                    });
                });
            });
            
            document.querySelectorAll('.visualize-code-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const codeIndex = e.target.closest('.visualize-code-btn').dataset.codeIndex;
                    const codeBlock = document.querySelectorAll('.interactive-code')[codeIndex];
                    const code = codeBlock.dataset.originalCode;
                    
                    if (this.jupyterlite && this.jupyterlite.isReady) {
                        this.jupyterlite.executeCode(code, codeBlock, { visualize: true });
                    }
                });
            });
        }

        addInteractiveButtons() {
            // Add floating action buttons
            const fabContainer = document.createElement('div');
            fabContainer.className = 'fab-container';
            fabContainer.innerHTML = `
                <button class="fab-button fab-main" aria-label="Interactive features">
                    <svg viewBox="0 0 24 24" width="24" height="24">
                        <path fill="currentColor" d="M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8M12,10A2,2 0 0,0 10,12A2,2 0 0,0 12,14A2,2 0 0,0 14,12A2,2 0 0,0 12,10M10,22C9.75,22 9.54,21.82 9.5,21.58L9.13,18.93C8.5,18.68 7.96,18.34 7.44,17.94L4.95,18.95C4.73,19.03 4.46,18.95 4.34,18.73L2.34,15.27C2.21,15.05 2.27,14.78 2.46,14.63L4.57,12.97L4.5,12L4.57,11L2.46,9.37C2.27,9.22 2.21,8.95 2.34,8.73L4.34,5.27C4.46,5.05 4.73,4.96 4.95,5.05L7.44,6.05C7.96,5.66 8.5,5.32 9.13,5.07L9.5,2.42C9.54,2.18 9.75,2 10,2H14C14.25,2 14.46,2.18 14.5,2.42L14.87,5.07C15.5,5.32 16.04,5.66 16.56,6.05L19.05,5.05C19.27,4.96 19.54,5.05 19.66,5.27L21.66,8.73C21.79,8.95 21.73,9.22 21.54,9.37L19.43,11L19.5,12L19.43,13L21.54,14.63C21.73,14.78 21.79,15.05 21.66,15.27L19.66,18.73C19.54,18.95 19.27,19.04 19.05,18.95L16.56,17.95C16.04,18.34 15.5,18.68 14.87,18.93L14.5,21.58C14.46,21.82 14.25,22 14,22H10Z"/>
                    </svg>
                </button>
                <div class="fab-menu">
                    <button class="fab-button" data-action="run-all" title="Run all code examples">
                        <svg viewBox="0 0 24 24" width="20" height="20">
                            <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z"/>
                        </svg>
                    </button>
                    <button class="fab-button" data-action="toggle-monitoring" title="Toggle performance monitoring">
                        <svg viewBox="0 0 24 24" width="20" height="20">
                            <path fill="currentColor" d="M22,21H2V3H4V19H6V10H10V19H12V6H16V19H18V14H22V21Z"/>
                        </svg>
                    </button>
                    <button class="fab-button" data-action="toggle-annotations" title="Toggle annotations">
                        <svg viewBox="0 0 24 24" width="20" height="20">
                            <path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H6L4,18V4H20"/>
                        </svg>
                    </button>
                    <button class="fab-button" data-action="toggle-playground" title="Toggle model playground">
                        <svg viewBox="0 0 24 24" width="20" height="20">
                            <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M12,6A6,6 0 0,0 6,12A6,6 0 0,0 12,18A6,6 0 0,0 18,12A6,6 0 0,0 12,6M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8Z"/>
                        </svg>
                    </button>
                </div>
            `;
            
            document.body.appendChild(fabContainer);
            
            // Add FAB event listeners
            document.querySelector('.fab-main').addEventListener('click', () => {
                fabContainer.classList.toggle('active');
            });
            
            document.querySelectorAll('.fab-menu .fab-button').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const action = e.currentTarget.dataset.action;
                    this.handleFabAction(action);
                    fabContainer.classList.remove('active');
                });
            });
        }

        handleFabAction(action) {
            switch (action) {
                case 'run-all':
                    this.runAllCodeExamples();
                    break;
                case 'toggle-monitoring':
                    if (this.performance) {
                        this.performance.togglePanel();
                    }
                    break;
                case 'toggle-annotations':
                    if (this.annotations) {
                        this.annotations.toggleAnnotationSidebar();
                    }
                    break;
                case 'toggle-playground':
                    if (this.playground) {
                        this.playground.togglePlayground();
                    }
                    break;
            }
        }

        runAllCodeExamples() {
            const codeBlocks = document.querySelectorAll('.interactive-code');
            codeBlocks.forEach((block, index) => {
                setTimeout(() => {
                    const code = block.dataset.originalCode;
                    if (this.jupyterlite && this.jupyterlite.isReady) {
                        this.jupyterlite.executeCode(code, block);
                    }
                }, index * 1000); // Stagger execution
            });
        }
    }

    // ============================================
    // CSS STYLES (Injected dynamically)
    // ============================================
    function injectStyles() {
        const styles = `
            /* Interactive Documentation Styles */
            .interactive-code {
                position: relative;
                margin: 1.5rem 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .code-controls {
                display: flex;
                gap: 0.5rem;
                padding: 0.5rem;
                background: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
            }
            
            .code-controls button {
                display: flex;
                align-items: center;
                gap: 0.3rem;
                padding: 0.3rem 0.6rem;
                border: 1px solid #dee2e6;
                background: white;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.85rem;
                transition: all 0.2s;
            }
            
            .code-controls button:hover {
                background: #e9ecef;
                border-color: #adb5bd;
            }
            
            .code-output {
                padding: 1rem;
                background: #f8f9fa;
                border-top: 1px solid #e9ecef;
                max-height: 300px;
                overflow-y: auto;
                display: none;
            }
            
            .code-output.active {
                display: block;
            }
            
            .execution-status {
                padding: 0.5rem;
                text-align: center;
                font-style: italic;
                color: #6c757d;
            }
            
            .execution-status.running {
                color: #007bff;
            }
            
            .output-text pre, .error-text pre {
                margin: 0;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .error-text {
                color: #dc3545;
                background: #f8d7da;
                padding: 0.5rem;
                border-radius: 4px;
            }
            
            .visualization-container {
                margin-top: 1rem;
                padding: 1rem;
                background: white;
                border-radius: 4px;
                border: 1px solid #e9ecef;
            }
            
            /* Performance Monitoring Panel */
            .performance-panel {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 400px;
                max-height: 500px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                z-index: 1000;
                overflow: hidden;
            }
            
            .performance-panel.hidden {
                display: none;
            }
            
            .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .panel-header h3 {
                margin: 0;
                font-size: 1.1rem;
            }
            
            .panel-toggle {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 0.2rem;
            }
            
            .panel-content {
                padding: 1rem;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
                margin-bottom: 1rem;
            }
            
            .metric-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
            
            .metric-card h4 {
                margin: 0 0 0.5rem 0;
                font-size: 0.9rem;
                color: #495057;
            }
            
            .metric-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #212529;
                margin-bottom: 0.5rem;
            }
            
            .controls {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            
            .performance-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 440px;
                padding: 0.8rem 1.2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                z-index: 999;
                font-weight: 500;
                transition: all 0.3s;
            }
            
            .performance-toggle-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
            }
            
            /* Annotation System Styles */
            .annotation-toolbar {
                position: absolute;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                padding: 0.5rem;
                display: flex;
                gap: 0.5rem;
                z-index: 1001;
            }
            
            .annotation-toolbar.hidden {
                display: none;
            }
            
            .annotation-btn {
                width: 36px;
                height: 36px;
                border: none;
                background: #f8f9fa;
                border-radius: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .annotation-btn:hover {
                background: #e9ecef;
                transform: scale(1.05);
            }
            
            .annotation-sidebar {
                position: fixed;
                top: 0;
                right: 0;
                width: 350px;
                height: 100vh;
                background: white;
                box-shadow: -4px 0 20px rgba(0,0,0,0.1);
                z-index: 1002;
                display: flex;
                flex-direction: column;
            }
            
            .annotation-sidebar.hidden {
                display: none;
            }
            
            .sidebar-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                background: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
            }
            
            .sidebar-header h3 {
                margin: 0;
            }
            
            .close-sidebar {
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: #6c757d;
            }
            
            .sidebar-content {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
            }
            
            .sidebar-footer {
                padding: 1rem;
                border-top: 1px solid #e9ecef;
            }
            
            .annotation-item {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border-left: 4px solid;
            }
            
            .annotation-item.highlight {
                border-left-color: #ffc107;
            }
            
            .annotation-item.comment {
                border-left-color: #17a2b8;
            }
            
            .annotation-item.suggestion {
                border-left-color: #28a745;
            }
            
            .annotation-item.question {
                border-left-color: #6f42c1;
            }
            
            .annotation-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
                font-size: 0.85rem;
            }
            
            .annotation-user {
                font-weight: 500;
            }
            
            .annotation-time {
                color: #6c757d;
            }
            
            .annotation-resolved {
                color: #28a745;
                font-weight: 500;
            }
            
            .selected-text {
                background: #fff3cd;
                padding: 0.5rem;
                border-radius: 4px;
                margin-bottom: 0.5rem;
                font-style: italic;
                font-size: 0.9rem;
            }
            
            .annotation-tags {
                margin-top: 0.5rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.3rem;
            }
            
            .tag {
                background: #e9ecef;
                padding: 0.2rem 0.5rem;
                border-radius: 12px;
                font-size: 0.75rem;
                color: #495057;
            }
            
            .annotation-actions {
                margin-top: 0.5rem;
                display: flex;
                gap: 0.5rem;
            }
            
            .annotation-actions button {
                padding: 0.3rem 0.6rem;
                font-size: 0.8rem;
                border: 1px solid #dee2e6;
                background: white;
                border-radius: 4px;
                cursor: pointer;
            }
            
            .annotation-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 600px;
                padding: 0.8rem 1.2rem;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
                z-index: 999;
                font-weight: 500;
                transition: all 0.3s;
            }
            
            .annotation-toggle-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4);
            }
            
            .annotation-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1003;
            }
            
            .annotation-modal.hidden {
                display: none;
            }
            
            .modal-content {
                background: white;
                border-radius: 12px;
                width: 500px;
                max-width: 90%;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                border-bottom: 1px solid #e9ecef;
            }
            
            .modal-header h4 {
                margin: 0;
            }
            
            .close-modal {
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: #6c757d;
            }
            
            .modal-body {
                padding: 1rem;
            }
            
            .form-group {
                margin-bottom: 1rem;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 500;
            }
            
            .form-group select,
            .form-group input,
            .form-group textarea {
                width: 100%;
                padding: 0.5rem;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 1rem;
            }
            
            .modal-footer {
                display: flex;
                justify-content: flex-end;
                gap: 0.5rem;
                padding: 1rem;
                border-top: 1px solid #e9ecef;
            }
            
            /* Model Playground Styles */
            .model-playground {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 800px;
                max-width: 95%;
                height: 600px;
                max-height: 90vh;
                background: white;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                z-index: 1004;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .model-playground.hidden {
                display: none;
            }
            
            .playground-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .playground-header h3 {
                margin: 0;
            }
            
            .playground-controls {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            
            .playground-controls select {
                padding: 0.5rem;
                border-radius: 6px;
                border: none;
                background: rgba(255,255,255,0.2);
                color: white;
                min-width: 200px;
            }
            
            .playground-controls select option {
                background: white;
                color: #333;
            }
            
            .playground-toggle {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 0.2rem;
            }
            
            .playground-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                background: #f8f9fa;
            }
            
            .welcome-message {
                text-align: center;
                padding: 2rem;
                color: #6c757d;
            }
            
            .welcome-message h4 {
                margin-bottom: 0.5rem;
                color: #495057;
            }
            
            .model-info {
                margin-top: 1rem;
                padding: 1rem;
                background: white;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
            
            .model-details h5 {
                margin: 0 0 0.5rem 0;
                color: #212529;
            }
            
            .model-specs {
                display: flex;
                gap: 1rem;
                margin-top: 0.5rem;
            }
            
            .spec {
                background: #e9ecef;
                padding: 0.2rem 0.5rem;
                border-radius: 12px;
                font-size: 0.8rem;
            }
            
            .chat-message {
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 12px;
                max-width: 80%;
            }
            
            .chat-message.user {
                background: #007bff;
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 4px;
            }
            
            .chat-message.assistant {
                background: white;
                border: 1px solid #e9ecef;
                margin-right: auto;
                border-bottom-left-radius: 4px;
            }
            
            .chat-message.error {
                background: #f8d7da;
                color: #721c24;
                margin: 0 auto;
            }
            
            .message-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
                font-size: 0.85rem;
                opacity: 0.8;
            }
            
            .message-content {
                line-height: 1.5;
            }
            
            .message-content pre {
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 0.5rem;
                border-radius: 4px;
                overflow-x: auto;
                margin: 0.5rem 0;
            }
            
            .message-content code {
                background: #f1f3f4;
                padding: 0.1rem 0.3rem;
                border-radius: 3px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            }
            
            .typing-indicator {
                display: flex;
                gap: 0.3rem;
                padding: 1rem;
                justify-content: center;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                background: #6c757d;
                border-radius: 50%;
                animation: typingAnimation 1.4s infinite ease-in-out;
            }
            
            .typing-dot:nth-child(1) {
                animation-delay: -0.32s;
            }
            
            .typing-dot:nth-child(2) {
                animation-delay: -0.16s;
            }
            
            @keyframes typingAnimation {
                0%, 80%, 100% {
                    transform: scale(0);
                }
                40% {
                    transform: scale(1);
                }
            }
            
            .input-container {
                padding: 1rem;
                border-top: 1px solid #e9ecef;
                background: white;
            }
            
            .input-container textarea {
                width: 100%;
                padding: 0.8rem;
                border: 1px solid #ced4da;
                border-radius: 8px;
                resize: none;
                font-family: inherit;
                font-size: 1rem;
                margin-bottom: 0.5rem;
            }
            
            .input-actions {
                display: flex;
                justify-content: flex-end;
                gap: 0.5rem;
            }
            
            .playground-settings {
                padding: 1rem;
                border-top: 1px solid #e9ecef;
                background: #f8f9fa;
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }
            
            .setting-group {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .setting-group label {
                font-size: 0.9rem;
                color: #495057;
                min-width: 80px;
            }
            
            .setting-group input[type="range"] {
                width: 100px;
            }
            
            .setting-group input[type="number"] {
                width: 80px;
                padding: 0.3rem;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
            
            .playground-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 780px;
                padding: 0.8rem 1.2rem;
                background: #fd7e14;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(253, 126, 20, 0.3);
                z-index: 999;
                font-weight: 500;
                transition: all 0.3s;
            }
            
            .playground-toggle-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(253, 126, 20, 0.4);
            }
            
            /* Floating Action Button */
            .fab-container {
                position: fixed;
                bottom: 100px;
                right: 20px;
                z-index: 998;
            }
            
            .fab-button {
                width: 56px;
                height: 56px;
                border-radius: 50%;
                border: none;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s;
            }
            
            .fab-button:hover {
                transform: scale(1.1);
            }
            
            .fab-main {
                position: relative;
                z-index: 2;
            }
            
            .fab-menu {
                position: absolute;
                bottom: 70px;
                right: 0;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s;
            }
            
            .fab-container.active .fab-menu {
                opacity: 1;
                visibility: visible;
            }
            
            .fab-menu .fab-button {
                width: 48px;
                height: 48px;
                background: white;
                color: #667eea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
            
            .fab-menu .fab-button:hover {
                background: #f8f9fa;
            }
            
            /* Button Styles */
            .btn {
                padding: 0.5rem 1rem;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.2s;
            }
            
            .btn-primary {
                background: #007bff;
                color: white;
            }
            
            .btn-primary:hover {
                background: #0056b3;
            }
            
            .btn-secondary {
                background: #6c757d;
                color: white;
            }
            
            .btn-secondary:hover {
                background: #545b62;
            }
            
            .btn-outline {
                background: transparent;
                border: 1px solid #007bff;
                color: #007bff;
            }
            
            .btn-outline:hover {
                background: #007bff;
                color: white;
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .performance-panel {
                    width: 90%;
                    right: 5%;
                    bottom: 10px;
                }
                
                .performance-toggle-btn {
                    right: 5%;
                    bottom: 520px;
                }
                
                .annotation-sidebar {
                    width: 100%;
                }
                
                .annotation-toggle-btn {
                    right: 5%;
                    bottom: 580px;
                }
                
                .model-playground {
                    width: 95%;
                    height: 80vh;
                }
                
                .playground-toggle-btn {
                    right: 5%;
                    bottom: 640px;
                }
                
                .fab-container {
                    right: 10px;
                    bottom: 80px;
                }
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    // ============================================
    // INITIALIZATION ON DOM READY
    // ============================================
    document.addEventListener('DOMContentLoaded', function() {
        // Inject styles
        injectStyles();
        
        // Initialize interactive documentation
        const interactiveDocs = new InteractiveDocumentation();
        
        // Make globally available for external access
        window.LlamaFactoryInteractive = interactiveDocs;
        
        // Auto-initialize after a short delay
        setTimeout(() => {
            interactiveDocs.initialize();
        }, 1000);
    });

    // ============================================
    // EXPORT FOR MODULE SYSTEMS
    // ============================================
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = {
            InteractiveDocumentation,
            JupyterLiteIntegration,
            PerformanceVisualizer,
            CollaborativeAnnotations,
            ModelPlayground,
            Utils
        };
    }

})();