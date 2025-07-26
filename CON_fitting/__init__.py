# Makes the CON_fitting directory a Python package for easier imports.

from importlib import import_module as _import_module

# Expose key submodules at package level for convenience
try:
    _import_module("CON_fitting.src")
except ImportError:
    # If for some reason src cannot be imported, fail silently to avoid breaking user code
    pass 