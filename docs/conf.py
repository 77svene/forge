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
    "jupyterlite_sphinx",
    "sphinxcontrib.httpdomain",
    "sphinx_tabs.tabs",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_js_files = [
    "js/switcher.js",
    "js/jupyterlite-helper.js",
    "js/interactive-docs.js",
    "js/annotation-system.js",
    "https://cdn.plot.ly/plotly-2.24.1.min.js",
]

html_css_files = [
    "css/lang-switcher.css",
    "css/interactive-docs.css",
    "css/annotation-system.css",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# JupyterLite configuration
jupyterlite_config = {
    "lite": {
        "settings": {
            "theme": "light",
        },
        "pipliteUrls": [
            "https://pypi.org/simple",
        ],
    },
}

# Interactive documentation settings
interactive_docs_config = {
    "enable_live_examples": True,
    "enable_gpu_monitoring": True,
    "enable_collaborative_annotations": True,
    "enable_model_playground": True,
    "default_kernel": "python",
    "max_execution_time": 300,
    "show_performance_charts": True,
    "enable_export": True,
}

# Real-time visualization settings
visualization_config = {
    "update_interval": 1000,  # ms
    "enable_gpu_metrics": True,
    "enable_training_metrics": True,
    "chart_height": 400,
    "enable_download": True,
}

# Collaborative annotation settings
annotation_config = {
    "enable_public_annotations": True,
    "enable_private_annotations": False,
    "annotation_api_url": "/api/annotations",
    "enable_upvoting": True,
    "enable_threaded_comments": True,
}

# Model playground settings
model_playground_config = {
    "supported_models": [
        "llama-2-7b",
        "llama-2-13b",
        "llama-2-70b",
        "mistral-7b",
        "falcon-7b",
    ],
    "max_input_length": 2048,
    "enable_streaming": True,
    "default_temperature": 0.7,
    "enable_parameter_tuning": True,
}

# Sphinx configuration for better interactive experience
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Enable MyST parser for Jupyter notebooks
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Additional configuration for live code execution
jupyterlite_dir = "_static/jupyterlite"
jupyterlite_contents_dir = "notebooks"

# Custom roles and directives
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: output(code)
   :language: text
   :class: output

.. |run-button| replace:: ▶️ Run
.. |reset-button| replace:: 🔄 Reset
.. |share-button| replace:: 📤 Share
.. |annotate-button| replace:: ✏️ Annotate
"""