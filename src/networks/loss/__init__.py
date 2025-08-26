import os
from importlib import import_module

LOSS_FUNCTIONS = {}

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py" and file != "loss.py":
        module = f"src.networks.loss.{file[:-3]}"
        import_module(module)
