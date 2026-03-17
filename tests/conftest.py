"""Pytest configuration: ensure src is on path and filter conflicting paths."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
our_src = str(project_root / "src")


def pytest_configure(config):
    """Run before test collection: put src first, filter other-project paths."""
    sys.path = [p for p in sys.path if "/arbor" not in p and "/solve-for-x" not in p]
    if our_src in sys.path:
        sys.path.remove(our_src)
    sys.path.insert(0, our_src)
