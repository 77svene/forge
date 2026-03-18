import os
import sys

# Add parent dir to path to allow importing conf.py
sys.path.insert(0, os.path.abspath(".."))

from conf import *  # noqa: F403

# Language settings
language = "zh_CN"
html_search_language = "zh"

# Static files
# Point to the root _static directory
html_static_path = ["../_static"]

# Add custom JS for language switcher
html_js_files = [
    "js/switcher.js",
    "js/jupyterlite-core.js",
    "js/interactive-docs.js",
    "js/real-time-viz.js",
    "js/collaborative-annotations.js",
]

# Add custom CSS for interactive documentation
html_css_files = [
    "css/interactive-docs.css",
    "css/jupyterlite-custom.css",
]

# Extensions for interactive documentation
extensions = [
    *globals().get("extensions", []),
    "jupyterlite_sphinx",
    "sphinx_copybutton",
    "sphinx_tabs",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# JupyterLite configuration
jupyterlite_config = "jupyterlite_config.json"
jupyterlite_contents = ["../notebooks"]
jupyterlite_dir = "_static/jupyterlite"

# Enable real-time visualization
realtime_viz_enabled = True
realtime_viz_port = 8050

# Collaborative annotations configuration
annotations_enabled = True
annotations_api_endpoint = "https://annotations.forge.org/api"

# Interactive playground settings
playground_enabled = True
playground_models = [
    "llama-2-7b",
    "llama-2-13b",
    "chatglm3-6b",
    "qwen-7b",
]

# Custom configuration for interactive elements
interactive_config = {
    "enable_code_execution": True,
    "enable_gpu_monitoring": True,
    "enable_live_metrics": True,
    "enable_collaborative_editing": True,
    "default_kernel": "python3",
    "execution_timeout": 30,
    "max_output_lines": 1000,
}

# HTML theme options for interactive documentation
html_theme_options = {
    *globals().get("html_theme_options", {}),
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "announcement": "🚀 Interactive documentation with live examples now available!",
}

# Custom sidebar for interactive features
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/interactive-playground.html",
        "sidebar/real-time-metrics.html",
        "sidebar/collaborative-annotations.html",
        "sidebar/scroll-end.html",
    ]
}

# Template paths for custom templates
templates_path = ["_templates", "../_templates"]

# Additional context for templates
html_context = {
    *globals().get("html_context", {}),
    "interactive_mode": True,
    "jupyterlite_url": "./jupyterlite/lab/index.html",
    "playground_url": "./playground/",
    "viz_dashboard_url": "./dashboard/",
}