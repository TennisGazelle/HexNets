# """Pytest configuration file for hexnets tests.

# This file sets up the Python path so that imports work correctly.
# The source files use imports like 'from commands.command import' which
# expects the 'src' directory to be in the Python path.
# """

# import sys
# from pathlib import Path

# def pytest_configure(config):
#     """Configure pytest - runs before test collection."""
#     # Get the absolute path to this project's src directory
#     project_root = Path(__file__).parent.parent
#     src_path_abs = (project_root / "src").resolve()
    
#     # Remove any other 'src' directories from path to avoid conflicts
#     sys.path = [p for p in sys.path if not (Path(p).name == 'src' and Path(p).resolve() != src_path_abs)]
    
#     # Add our src directory at the beginning of the path
#     if str(src_path_abs) not in sys.path:
#         sys.path.insert(0, str(src_path_abs))

# # Also set it up at module import time as a fallback
# project_root = Path(__file__).parent.parent
# src_path_abs = (project_root / "src").resolve()
# sys.path = [p for p in sys.path if not (Path(p).name == 'src' and Path(p).resolve() != src_path_abs)]
# if str(src_path_abs) not in sys.path:
#     sys.path.insert(0, str(src_path_abs))
