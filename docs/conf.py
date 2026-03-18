# Configuration file for the Sphinx documentation builder.


# Define common settings here
project = "forge"
copyright = "2024, forge Team"
author = "forge Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_jupyterlite",  # Add JupyterLite integration for interactive documentation
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_js_files = [
    "js/switcher.js",
    "js/playground.js",  # Add playground functionality
    "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js",  # Pyodide for in-browser Python
]

html_css_files = [
    "css/lang-switcher.css",
    "css/playground.css",  # Add playground styling
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# JupyterLite configuration for interactive documentation
jupyterlite_dir = "_notebooks"  # Directory for notebook content
jupyterlite_config = {
    "LiteBuildConfig": {
        "federated_extensions": [
            "https://conda.anaconda.org/conda-forge/noarch/ipycanvas-0.13.1-pyhd8ed1ab_0.tar.bz2",
            "https://conda.anaconda.org/conda-forge/noarch/ipycytoscape-1.3.3-pyhd8ed1ab_1.tar.bz2",
            "https://conda.anaconda.org/conda-forge/noarch/ipympl-0.9.3-pyhd8ed1ab_0.tar.bz2",
            "https://conda.anaconda.org/conda-forge/noarch/ipywidgets-8.1.0-pyhd8ed1ab_0.tar.bz2",
        ],
    },
    "PiPyConfig": {
        "piplite_urls": [
            "https://files.pythonhosted.org/packages/py3/i/ipywidgets/ipywidgets-8.1.0-py3-none-any.whl",
        ],
    },
}

# Playground configuration
playground_config = {
    "default_kernel": "python",
    "enable_widgets": True,
    "max_execution_time": 30,  # seconds
    "allowed_packages": [
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
    ],
}

# Add custom directives for interactive elements
rst_prolog = """
.. |try_it| raw:: html

   <button class="try-it-button" onclick="openPlayground(this)">Try It</button>

.. |playground| raw:: html

   <div class="playground-container">
     <div class="playground-header">
       <span class="playground-title">Interactive Playground</span>
       <button class="playground-close" onclick="closePlayground(this)">×</button>
     </div>
     <div class="playground-editor" data-language="python"></div>
     <div class="playground-controls">
       <button class="playground-run" onclick="runPlayground(this)">▶ Run</button>
       <button class="playground-reset" onclick="resetPlayground(this)">↺ Reset</button>
     </div>
     <div class="playground-output"></div>
   </div>
"""

# Setup for interactive model comparison
def setup(app):
    app.add_css_file("css/playground.css")
    app.add_js_file("js/playground.js")
    app.add_js_file("js/model-comparison.js")
    
    # Add custom configuration to the build environment
    app.connect("builder-inited", configure_playground)
    
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

def configure_playground(app):
    """Configure the interactive playground during build."""
    import os
    import json
    
    # Create playground configuration file
    playground_config_path = os.path.join(app.outdir, "_static", "playground-config.json")
    os.makedirs(os.path.dirname(playground_config_path), exist_ok=True)
    
    with open(playground_config_path, "w") as f:
        json.dump(playground_config, f, indent=2)