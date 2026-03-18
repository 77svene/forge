/**
 * LlamaFactory Interactive Documentation & Live Playground
 * Transform static documentation into an interactive experience with live code execution
 * and model comparison tools.
 */

class LlamaFactoryPlayground {
    constructor() {
        this.pyodide = null;
        this.isLoading = false;
        this.activePlaygrounds = new Map();
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadPyodide();
        this.initializePlaygrounds();
        this.setupModelComparison();
    }

    async loadPyodide() {
        if (this.pyodide) return this.pyodide;
        
        this.isLoading = true;
        this.showLoadingIndicator();
        
        try {
            // Load Pyodide from CDN
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
            document.head.appendChild(script);
            
            await new Promise((resolve) => {
                script.onload = resolve;
            });
            
            this.pyodide = await loadPyodide();
            
            // Install essential packages
            await this.pyodide.loadPackage(['numpy', 'pandas', 'micropip']);
            
            // Install LlamaFactory dependencies
            await this.pyodide.runPythonAsync(`
                import micropip
                await micropip.install('transformers')
                await micropip.install('peft')
                await micropip.install('datasets')
                await micropip.install('accelerate')
                await micropip.install('bitsandbytes')
            `);
            
            // Load LlamaFactory utilities
            await this.loadLlamaFactoryModules();
            
            this.hideLoadingIndicator();
            this.isLoading = false;
            
            return this.pyodide;
        } catch (error) {
            console.error('Failed to load Pyodide:', error);
            this.showError('Failed to load Python environment. Please refresh the page.');
            throw error;
        }
    }

    async loadLlamaFactoryModules() {
        // Load essential LlamaFactory modules for in-browser execution
        const moduleCode = `
import sys
from io import StringIO
import json
import numpy as np

# Mock LlamaFactory core functionality for browser environment
class MockLlamaFactory:
    def __init__(self):
        self.config = {}
        
    def load_config(self, config_str):
        """Load configuration from string"""
        self.config = json.loads(config_str)
        return self.config
    
    def calculate_flops(self, model_size, batch_size, seq_length, hidden_size):
        """Calculate FLOPs for training"""
        # Simplified FLOPs calculation
        flops = 6 * model_size * batch_size * seq_length
        return flops / 1e9  # Convert to GFLOPs
    
    def calculate_mfu(self, flops, gpu_tflops, num_gpus=1):
        """Calculate Model FLOPs Utilization"""
        mfu = (flops / 1e3) / (gpu_tflops * num_gpus)
        return min(mfu, 1.0)
    
    def estimate_memory(self, model_size, batch_size, seq_length, precision='fp16'):
        """Estimate memory requirements"""
        bytes_per_param = {'fp32': 4, 'fp16': 2, 'int8': 1, 'int4': 0.5}
        param_memory = model_size * bytes_per_param[precision]
        activation_memory = batch_size * seq_length * 4096 * 2  # Rough estimate
        total_gb = (param_memory + activation_memory) / (1024**3)
        return total_gb
    
    def compare_models(self, model_configs):
        """Compare multiple model configurations"""
        results = []
        for config in model_configs:
            flops = self.calculate_flops(
                config['model_size'],
                config['batch_size'],
                config['seq_length'],
                config.get('hidden_size', 4096)
            )
            memory = self.estimate_memory(
                config['model_size'],
                config['batch_size'],
                config['seq_length'],
                config.get('precision', 'fp16')
            )
            results.append({
                'model': config['name'],
                'flops_gflops': round(flops, 2),
                'memory_gb': round(memory, 2),
                'throughput_est': round(flops / 100, 2)  # Simplified throughput estimate
            })
        return results

# Initialize global LlamaFactory instance
llamafactory = MockLlamaFactory()

# Utility functions for interactive examples
def run_fine_tuning_example(model_name, learning_rate, batch_size, epochs, lora_r=8, lora_alpha=32):
    """Simulate fine-tuning configuration"""
    config = {
        "model_name": model_name,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "num_epochs": int(epochs),
        "lora_config": {
            "r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05
        }
    }
    
    # Calculate training metrics
    model_sizes = {
        "llama-7b": 7e9,
        "llama-13b": 13e9,
        "llama-70b": 70e9,
        "qwen-7b": 7e9,
        "mistral-7b": 7e9
    }
    
    model_size = model_sizes.get(model_name.split('/')[-1].lower(), 7e9)
    flops = llamafactory.calculate_flops(model_size, int(batch_size), 2048, 4096)
    memory = llamafactory.estimate_memory(model_size, int(batch_size), 2048)
    
    result = {
        "configuration": config,
        "estimated_flops_gflops": round(flops, 2),
        "estimated_memory_gb": round(memory, 2),
        "training_time_est_hours": round(flops / 1000, 2),  # Rough estimate
        "recommendation": "Good configuration" if float(learning_rate) < 5e-4 else "Consider lower learning rate"
    }
    
    return json.dumps(result, indent=2)

def compare_models_example(models_json):
    """Compare multiple model configurations"""
    models = json.loads(models_json)
    comparison = llamafactory.compare_models(models)
    return json.dumps(comparison, indent=2)

def calculate_lr_schedule(total_steps, warmup_steps, min_lr=1e-6, max_lr=1e-4):
    """Calculate learning rate schedule"""
    import numpy as np
    
    steps = np.arange(total_steps)
    warmup = np.linspace(min_lr, max_lr, warmup_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * (steps[warmup_steps:] - warmup_steps) / (total_steps - warmup_steps)))
    decay = max_lr * cosine_decay
    
    lr_schedule = np.concatenate([warmup, decay])
    
    return {
        "steps": steps.tolist(),
        "learning_rates": lr_schedule.tolist(),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps
    }

print("LlamaFactory Playground loaded successfully!")
`;
        
        await this.pyodide.runPythonAsync(moduleCode);
    }

    setupEventListeners() {
        // Handle playground run buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('playground-run-btn')) {
                const playgroundId = e.target.dataset.playground;
                this.runPlayground(playgroundId);
            }
            
            if (e.target.classList.contains('playground-reset-btn')) {
                const playgroundId = e.target.dataset.playground;
                this.resetPlayground(playgroundId);
            }
            
            if (e.target.classList.contains('playground-copy-btn')) {
                const playgroundId = e.target.dataset.playground;
                this.copyPlaygroundCode(playgroundId);
            }
        });
        
        // Handle input changes
        document.addEventListener('input', (e) => {
            if (e.target.classList.contains('playground-input')) {
                const playgroundId = e.target.dataset.playground;
                this.updatePlaygroundPreview(playgroundId);
            }
        });
        
        // Handle model comparison tab switching
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('comparison-tab')) {
                this.switchComparisonTab(e.target.dataset.tab);
            }
        });
    }

    initializePlaygrounds() {
        const playgrounds = document.querySelectorAll('[data-playground-id]');
        
        playgrounds.forEach(playground => {
            const playgroundId = playground.dataset.playgroundId;
            const codeTemplate = playground.querySelector('.playground-code-template');
            const inputs = playground.querySelectorAll('.playground-input');
            
            if (codeTemplate && inputs.length > 0) {
                this.activePlaygrounds.set(playgroundId, {
                    element: playground,
                    codeTemplate: codeTemplate.textContent,
                    inputs: Array.from(inputs),
                    outputElement: playground.querySelector('.playground-output'),
                    statusElement: playground.querySelector('.playground-status')
                });
                
                this.updatePlaygroundPreview(playgroundId);
            }
        });
    }

    setupModelComparison() {
        const comparisonContainer = document.getElementById('model-comparison');
        if (!comparisonContainer) return;
        
        // Initialize comparison chart
        this.comparisonChart = null;
        this.initializeComparisonChart();
        
        // Setup model selection
        const modelCheckboxes = comparisonContainer.querySelectorAll('.model-checkbox');
        modelCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateComparison());
        });
        
        // Initial comparison
        this.updateComparison();
    }

    initializeComparisonChart() {
        const ctx = document.getElementById('comparison-chart');
        if (!ctx) return;
        
        // Simple bar chart implementation
        this.comparisonChart = {
            canvas: ctx,
            ctx: ctx.getContext('2d'),
            data: [],
            render: function() {
                const ctx = this.ctx;
                const width = this.canvas.width;
                const height = this.canvas.height;
                const padding = 40;
                
                ctx.clearRect(0, 0, width, height);
                
                if (this.data.length === 0) {
                    ctx.fillStyle = '#666';
                    ctx.textAlign = 'center';
                    ctx.fillText('Select models to compare', width/2, height/2);
                    return;
                }
                
                // Draw axes
                ctx.strokeStyle = '#ccc';
                ctx.beginPath();
                ctx.moveTo(padding, padding);
                ctx.lineTo(padding, height - padding);
                ctx.lineTo(width - padding, height - padding);
                ctx.stroke();
                
                // Draw bars
                const barWidth = (width - 2 * padding) / (this.data.length * 2);
                const maxFlops = Math.max(...this.data.map(d => d.flops_gflops));
                
                this.data.forEach((item, index) => {
                    const x = padding + index * 2 * barWidth + barWidth/2;
                    const barHeight = (item.flops_gflops / maxFlops) * (height - 2 * padding);
                    const y = height - padding - barHeight;
                    
                    // Draw bar
                    ctx.fillStyle = this.getColor(index);
                    ctx.fillRect(x, y, barWidth, barHeight);
                    
                    // Draw label
                    ctx.fillStyle = '#333';
                    ctx.textAlign = 'center';
                    ctx.fillText(item.model, x + barWidth/2, height - padding + 20);
                    ctx.fillText(`${item.flops_gflops} GFLOPs`, x + barWidth/2, y - 5);
                });
            },
            getColor: function(index) {
                const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'];
                return colors[index % colors.length];
            }
        };
    }

    async runPlayground(playgroundId) {
        const playground = this.activePlaygrounds.get(playgroundId);
        if (!playground || this.isLoading) return;
        
        const { element, codeTemplate, inputs, outputElement, statusElement } = playground;
        
        try {
            // Update status
            if (statusElement) {
                statusElement.textContent = 'Running...';
                statusElement.className = 'playground-status running';
            }
            
            // Gather input values
            const inputValues = {};
            inputs.forEach(input => {
                const paramName = input.dataset.param;
                let value = input.value;
                
                // Convert to appropriate type
                if (input.dataset.type === 'number') {
                    value = parseFloat(value);
                } else if (input.dataset.type === 'json') {
                    try {
                        value = JSON.parse(value);
                    } catch (e) {
                        throw new Error(`Invalid JSON for ${paramName}`);
                    }
                }
                
                inputValues[paramName] = value;
            });
            
            // Replace placeholders in code template
            let code = codeTemplate;
            for (const [key, value] of Object.entries(inputValues)) {
                const placeholder = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
                code = code.replace(placeholder, JSON.stringify(value));
            }
            
            // Execute code
            const result = await this.pyodide.runPythonAsync(code);
            
            // Display result
            if (outputElement) {
                outputElement.innerHTML = `
                    <div class="playground-result">
                        <pre><code>${this.escapeHtml(result)}</code></pre>
                    </div>
                `;
            }
            
            // Update status
            if (statusElement) {
                statusElement.textContent = 'Completed';
                statusElement.className = 'playground-status success';
            }
            
        } catch (error) {
            console.error('Playground execution error:', error);
            
            if (outputElement) {
                outputElement.innerHTML = `
                    <div class="playground-error">
                        <strong>Error:</strong> ${this.escapeHtml(error.message)}
                    </div>
                `;
            }
            
            if (statusElement) {
                statusElement.textContent = 'Error';
                statusElement.className = 'playground-status error';
            }
        }
    }

    updatePlaygroundPreview(playgroundId) {
        const playground = this.activePlaygrounds.get(playgroundId);
        if (!playground) return;
        
        const { codeTemplate, inputs, element } = playground;
        const previewElement = element.querySelector('.playground-preview');
        
        if (!previewElement) return;
        
        // Gather current input values
        const inputValues = {};
        inputs.forEach(input => {
            inputValues[input.dataset.param] = input.value;
        });
        
        // Replace placeholders for preview
        let previewCode = codeTemplate;
        for (const [key, value] of Object.entries(inputValues)) {
            const placeholder = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
            previewCode = previewCode.replace(placeholder, value);
        }
        
        previewElement.textContent = previewCode;
    }

    resetPlayground(playgroundId) {
        const playground = this.activePlaygrounds.get(playgroundId);
        if (!playground) return;
        
        const { inputs, outputElement, statusElement } = playground;
        
        // Reset inputs to default values
        inputs.forEach(input => {
            if (input.defaultValue) {
                input.value = input.defaultValue;
            }
        });
        
        // Clear output
        if (outputElement) {
            outputElement.innerHTML = '';
        }
        
        // Reset status
        if (statusElement) {
            statusElement.textContent = 'Ready';
            statusElement.className = 'playground-status';
        }
        
        this.updatePlaygroundPreview(playgroundId);
    }

    async copyPlaygroundCode(playgroundId) {
        const playground = this.activePlaygrounds.get(playgroundId);
        if (!playground) return;
        
        const { codeTemplate, inputs } = playground;
        
        // Gather input values
        const inputValues = {};
        inputs.forEach(input => {
            inputValues[input.dataset.param] = input.value;
        });
        
        // Replace placeholders
        let code = codeTemplate;
        for (const [key, value] of Object.entries(inputValues)) {
            const placeholder = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
            code = code.replace(placeholder, value);
        }
        
        try {
            await navigator.clipboard.writeText(code);
            
            // Show feedback
            const btn = document.querySelector(`[data-playground="${playgroundId}"].playground-copy-btn`);
            if (btn) {
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            }
        } catch (error) {
            console.error('Failed to copy code:', error);
        }
    }

    async updateComparison() {
        const comparisonContainer = document.getElementById('model-comparison');
        if (!comparisonContainer) return;
        
        const selectedModels = [];
        const checkboxes = comparisonContainer.querySelectorAll('.model-checkbox:checked');
        
        checkboxes.forEach(checkbox => {
            const modelCard = checkbox.closest('.model-card');
            if (modelCard) {
                const modelData = {
                    name: modelCard.dataset.modelName,
                    model_size: parseFloat(modelCard.dataset.modelSize),
                    batch_size: 4,
                    seq_length: 2048,
                    hidden_size: 4096,
                    precision: 'fp16'
                };
                selectedModels.push(modelData);
            }
        });
        
        if (selectedModels.length === 0) {
            if (this.comparisonChart) {
                this.comparisonChart.data = [];
                this.comparisonChart.render();
            }
            return;
        }
        
        try {
            // Run comparison in Pyodide
            const comparisonCode = `
import json
models = ${JSON.stringify(selectedModels)}
comparison = llamafactory.compare_models(models)
json.dumps(comparison)
`;
            
            const result = await this.pyodide.runPythonAsync(comparisonCode);
            const comparisonData = JSON.parse(result);
            
            // Update chart
            if (this.comparisonChart) {
                this.comparisonChart.data = comparisonData;
                this.comparisonChart.render();
            }
            
            // Update table
            this.updateComparisonTable(comparisonData);
            
        } catch (error) {
            console.error('Comparison error:', error);
        }
    }

    updateComparisonTable(data) {
        const tableBody = document.querySelector('#comparison-table tbody');
        if (!tableBody) return;
        
        tableBody.innerHTML = data.map(item => `
            <tr>
                <td>${item.model}</td>
                <td>${item.flops_gflops} GFLOPs</td>
                <td>${item.memory_gb} GB</td>
                <td>${item.throughput_est} samples/sec</td>
            </tr>
        `).join('');
    }

    switchComparisonTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.comparison-tab').forEach(tab => {
            tab.classList.remove('active');
            if (tab.dataset.tab === tabId) {
                tab.classList.add('active');
            }
        });
        
        // Update tab content
        document.querySelectorAll('.comparison-tab-content').forEach(content => {
            content.classList.remove('active');
            if (content.id === tabId) {
                content.classList.add('active');
            }
        });
    }

    showLoadingIndicator() {
        const loader = document.createElement('div');
        loader.id = 'pyodide-loader';
        loader.innerHTML = `
            <div class="loader-overlay">
                <div class="loader-content">
                    <div class="spinner"></div>
                    <p>Loading Python Environment...</p>
                    <small>This may take a moment on first load</small>
                </div>
            </div>
        `;
        document.body.appendChild(loader);
    }

    hideLoadingIndicator() {
        const loader = document.getElementById('pyodide-loader');
        if (loader) {
            loader.remove();
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'playground-global-error';
        errorDiv.innerHTML = `
            <div class="error-content">
                <strong>Error:</strong> ${message}
                <button onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 10000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Public API methods
    async executeCode(code, params = {}) {
        if (!this.pyodide) {
            await this.loadPyodide();
        }
        
        let processedCode = code;
        for (const [key, value] of Object.entries(params)) {
            const placeholder = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
            processedCode = processedCode.replace(placeholder, JSON.stringify(value));
        }
        
        return await this.pyodide.runPythonAsync(processedCode);
    }

    getAvailableModels() {
        return [
            { id: 'llama-7b', name: 'LLaMA-7B', size: '7B' },
            { id: 'llama-13b', name: 'LLaMA-13B', size: '13B' },
            { id: 'llama-70b', name: 'LLaMA-70B', size: '70B' },
            { id: 'qwen-7b', name: 'Qwen-7B', size: '7B' },
            { id: 'mistral-7b', name: 'Mistral-7B', size: '7B' },
            { id: 'baichuan2-7b', name: 'Baichuan2-7B', size: '7B' }
        ];
    }

    calculateOptimalConfig(modelSize, gpuMemory) {
        // Calculate optimal batch size and other parameters based on GPU memory
        const bytesPerParam = 2; // fp16
        const modelMemory = modelSize * bytesPerParam;
        const availableMemory = gpuMemory * 0.9; // 90% utilization
        
        const maxBatchSize = Math.floor((availableMemory - modelMemory) / (2048 * 4096 * 2));
        
        return {
            recommended_batch_size: Math.max(1, Math.min(32, maxBatchSize)),
            gradient_checkpointing: modelMemory > availableMemory * 0.7,
            use_lora: modelSize > 7e9,
            lora_r: modelSize > 13e9 ? 16 : 8
        };
    }
}

// Initialize playground when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.llamaFactoryPlayground = new LlamaFactoryPlayground();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LlamaFactoryPlayground;
}