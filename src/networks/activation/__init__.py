import os
from importlib import import_module

ACTIVATION_FUNCTIONS = {}

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py" and file != "activation.py":
        module = f"src.networks.activation.{file[:-3]}"
        import_module(module)
