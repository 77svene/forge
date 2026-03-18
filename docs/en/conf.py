import os
import sys
import json

# Add parent dir to path to allow importing conf.py
sys.path.insert(0, os.path.abspath(".."))

from conf import *  # noqa: F403

# Language settings
language = "en"
html_search_language = "en"

# Static files
# Point to the root _static directory
html_static_path = ["../_static"]

# Add custom JS for language switcher and interactive features
html_js_files = [
    "js/switcher.js",
    "https://cdn.jsdelivr.net/npm/jupyterlite@0.1.0/dist/piplite.js",
    "js/playground.js",
    "js/interactive-widgets.js",
]

# Add custom CSS for interactive components
html_css_files = [
    "https://cdn.jsdelivr.net/npm/jupyterlite@0.1.0/dist/piplite.css",
    "css/playground.css",
    "css/interactive-widgets.css",
]

# Interactive Documentation Configuration
interactive_config = {
    "enable_playground": True,
    "enable_model_comparison": True,
    "enable_hyperparameter_tuning": True,
    "default_models": ["llama2", "mistral", "falcon"],
    "default_datasets": ["alpaca", "dolly", "self-instruct"],
    "default_hyperparameters": {
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 100,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05
    }
}

# Setup function for Sphinx
def setup(app):
    """Setup Sphinx app with interactive documentation features."""
    # Add custom configuration values
    app.add_config_value('interactive_config', interactive_config, 'html')
    
    # Connect to builder-inited event to inject configuration
    app.connect('builder-inited', inject_interactive_config)
    
    # Add custom directives for interactive components
    app.add_directive('try-it', TryItDirective)
    app.add_directive('model-compare', ModelCompareDirective)
    app.add_directive('hyperparameter-tuner', HyperparameterTunerDirective)

def inject_interactive_config(app):
    """Inject interactive configuration into the HTML context."""
    if app.builder.name == 'html':
        # Add interactive config to template context
        app.builder.globalcontext['interactive_config'] = json.dumps(interactive_config)
        
        # Create playground directory if it doesn't exist
        playground_dir = os.path.join(app.outdir, '_static', 'playground')
        os.makedirs(playground_dir, exist_ok=True)

# Custom directive for "Try It" sections
from docutils.parsers.rst import Directive
from docutils import nodes

class TryItDirective(Directive):
    """Directive to create interactive 'Try It' sections."""
    required_arguments = 1  # Feature name
    optional_arguments = 0
    has_content = True
    
    def run(self):
        feature_name = self.arguments[0]
        code_content = '\n'.join(self.content)
        
        # Create container for interactive section
        container = nodes.container(classes=['try-it-section', f'try-it-{feature_name}'])
        
        # Add header
        header = nodes.rubric(text=f"Try It: {feature_name.replace('-', ' ').title()}")
        container += header
        
        # Add code block
        code_block = nodes.literal_block(code_content, code_content)
        code_block['language'] = 'python'
        container += code_block
        
        # Add interactive controls
        controls = nodes.container(classes=['try-it-controls'])
        run_button = nodes.inline(text="▶ Run", classes=['run-button'])
        reset_button = nodes.inline(text="↺ Reset", classes=['reset-button'])
        controls += run_button
        controls += reset_button
        container += controls
        
        # Add output area
        output = nodes.container(classes=['try-it-output'])
        container += output
        
        return [container]

class ModelCompareDirective(Directive):
    """Directive for model comparison tools."""
    has_content = True
    
    def run(self):
        container = nodes.container(classes=['model-comparison-widget'])
        container += nodes.rubric(text="Model Comparison Tool")
        
        # Add model selector
        model_selector = nodes.container(classes=['model-selector'])
        container += model_selector
        
        # Add comparison metrics
        metrics = nodes.container(classes=['comparison-metrics'])
        container += metrics
        
        return [container]

class HyperparameterTunerDirective(Directive):
    """Directive for hyperparameter tuning widgets."""
    has_content = True
    
    def run(self):
        container = nodes.container(classes=['hyperparameter-tuner'])
        container += nodes.rubric(text="Hyperparameter Tuning")
        
        # Add sliders for each hyperparameter
        for param, value in interactive_config['default_hyperparameters'].items():
            param_container = nodes.container(classes=['param-slider'])
            param_container += nodes.paragraph(text=f"{param.replace('_', ' ').title()}:")
            slider = nodes.inline(classes=['slider-widget'], attributes={'data-param': param, 'data-value': value})
            param_container += slider
            container += param_container
        
        return [container]