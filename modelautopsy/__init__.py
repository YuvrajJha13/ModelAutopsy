"""
ModelAutopsy: Enterprise ML Debugging Library.
"""
import sys

from .debugger import watch

# Engine Detection
ENGINE_STATUS = "Unknown"

try:
    # Try to import Rust module (Note: Name is usually from Cargo.toml)
    # We look for 'modelautopsy_rust' since we renamed project
    import modelautopsy_rust
    ENGINE_STATUS = "Rust Safety Core 🦀"
except ImportError:
    try:
        # Try C++ Module
        import _core_cpp
        ENGINE_STATUS = "C++ Speed Core ⚡"
    except ImportError:
        ENGINE_STATUS = "Pure Python (Slow) 🐢"

# Expose Analyze
from .debugger import analyze as _internal_analyze

def analyze(tensor):
    """Public API. Analyzes tensor using best available engine."""
    return _internal_analyze(tensor)

__all__ = ["watch", "analyze", "ENGINE_STATUS"]
