import os
import importlib

# Dynamically import all modules in this folder
module_files = [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.py') and not f.startswith('__')]
__all__ = []

for module_file in module_files:
    module_name = module_file[:-3]  # Remove ".py" extension
    module = importlib.import_module(f'.{module_name}', package=__name__)  # Import relative to this package
    globals()[module_name] = module  # Add module to the global namespace
    __all__.append(module_name)  # Include the module in __all__