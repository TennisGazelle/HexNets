"""Pytest configuration file for hexnets tests."""

import sys
import importlib
from pathlib import Path

# Get our src directory path (conftest.py is in tests/, so project root is parent of tests/)
project_root = Path(__file__).resolve().parent.parent
our_src = str(project_root / "src")

def pytest_configure(config):
    """Configure pytest - runs before test collection."""
    # Remove conflicting paths (from other projects)
    sys.path = [p for p in sys.path if "/arbor" not in p and "/solve-for-x" not in p]

    # Clear ALL cached modules that might conflict
    modules_to_clear = []
    for name in list(sys.modules.keys()):
        if (name.startswith('commands') or 
            name.startswith('services') or 
            name.startswith('networks') or
            name.startswith('data')):
            modules_to_clear.append(name)
    
    for name in modules_to_clear:
        del sys.modules[name]
    
    # Ensure our src is first
    if our_src in sys.path:
        sys.path.remove(our_src)
    sys.path.insert(0, our_src)

# Do it at import time too (runs when conftest is imported)
sys.path = [p for p in sys.path if '/arbor' not in p and '/solve-for-x' not in p]
if our_src not in sys.path:
    sys.path.insert(0, our_src)
