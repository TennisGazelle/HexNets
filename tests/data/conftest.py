"""Ensure src is first on sys.path so 'data' resolves to the app package."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
if _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)
