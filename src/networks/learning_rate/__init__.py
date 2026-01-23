import os
from importlib import import_module

LEARNING_RATES = {}

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py" and file != "learning_rate.py":
        module = f"networks.learning_rate.{file[:-3]}"
        import_module(module)
