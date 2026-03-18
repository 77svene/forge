import os
import sys

# Add parent dir to path to allow importing conf.py
sys.path.insert(0, os.path.abspath(".."))

from conf import *  # noqa: F403

# Language settings
language = "en"
html_search_language = "en"

# Static files
# Point to the root _static directory
html_static_path = ["../_static"]

# Add custom JS for language switcher
html_js_files = [
    "js/switcher.js",
    "js/interactive.js",
    "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js",
    "https://cdn.plot.ly/plotly-2.24.1.min.js",
    "js/jupyterlite-core.js",
]

# Add custom CSS for interactive documentation
html_css_files = [
    "css/interactive.css",
    "css/jupyterlite.css",
]

# Extensions for interactive documentation
extensions = [
    *globals().get("extensions", []),
    "sphinxcontrib.video",
    "sphinxcontrib.mermaid",
    "jupyterlite_sphinx",
]

# JupyterLite configuration
jupyterlite_config = {
    "LiteBuildConfig": {
        "contents": ["../notebooks"],
        "output_dir": "_static/jupyterlite",
    },
    "SphinxAddonConfig": {
        "default_kernel": "python",
        "theme": "lab",
    },
}

# Interactive features configuration
interactive_features = {
    "enable_live_examples": True,
    "enable_gpu_monitoring": True,
    "enable_collaborative_annotations": True,
    "enable_model_playground": True,
    "gpu_monitoring_endpoint": "/api/gpu-stats",
    "annotation_api_endpoint": "/api/annotations",
    "playground_model_endpoint": "/api/model-inference",
}

# Template configuration for interactive elements
templates_path = ["_templates", "../_templates"]

# HTML theme options for interactive documentation
html_theme_options = {
    **globals().get("html_theme_options", {}),
    "navbar_center": ["navbar-nav", "interactive-nav"],
    "announcement": "Interactive Documentation: Run examples, visualize training, collaborate!",
}

# Custom sidebar for interactive features
html_sidebars = {
    "**": [
        "sidebar-logo",
        "search-field",
        "sidebar-nav",
        "interactive-sidebar",
        "collaborative-annotations",
    ]
}

# Notebook execution settings
nbsphinx_execute = "never"  # We handle execution via JupyterLite
nbsphinx_allow_errors = True

# Mermaid configuration for diagrams
mermaid_output_format = "svg"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'default'});"

# Video settings for tutorial videos
video_enforce_extra_source = False

# Custom roles for interactive elements
rst_prolog = """
.. role:: interactive-note
   :class: interactive-note

.. role:: runnable-example
   :class: runnable-example

.. role:: gpu-monitor
   :class: gpu-monitor
"""

# Add custom directives
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonition import BaseAdmonition

class InteractiveNote(BaseAdmonition):
    node_class = nodes.note
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "title": directives.unchanged,
        "kernel": directives.unchanged,
    }

def setup(app):
    app.add_directive("interactive-note", InteractiveNote)
    app.add_js_file("js/interactive.js")
    app.add_css_file("css/interactive.css")
    
    # Add custom events for interactive features
    app.connect("html-page-context", add_interactive_context)
    
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

def add_interactive_context(app, pagename, templatename, context, doctree):
    """Add interactive feature flags to template context"""
    context["interactive_features"] = interactive_features
    context["jupyterlite_base_url"] = "_static/jupyterlite"