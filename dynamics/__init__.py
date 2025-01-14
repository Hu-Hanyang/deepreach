import os
import pkgutil
import importlib

__all__ = []  # Define __all__ to explicitly export module names

# Current directory of the `dynamics` package
current_dir = os.path.dirname(__file__)

# Iterate through all modules in the `dynamics` folder
for _, module_name, _ in pkgutil.iter_modules([current_dir]):
    # Import each module dynamically
    module = importlib.import_module(f".{module_name}", package=__name__)
    # Add module to the global namespace
    globals()[module_name] = module
    # Append the module name to __all__ for wildcard imports
    __all__.append(module_name)