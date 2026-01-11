"""
MLGuardian: Enterprise ML Debugging Library.
"""
import sys

# Import Debugger
from .debugger import watch

# Engine Detection
ENGINE_STATUS = "Unknown"

# Try Rust
try:
    import mlguardian_rust
    ENGINE_STATUS = "Rust Safety Core 🦀"
except ImportError:
    # Try C++
    try:
        import _core_cpp
        ENGINE_STATUS = "C++ Speed Core ⚡"
    except ImportError:
        ENGINE_STATUS = "Pure Python (Slow) 🐢"

# Expose Analyze (Wrapper logic is inside debugger to avoid circular imports)
from .debugger import analyze as _internal_analyze

def analyze(tensor):
    """Public API. Analyzes tensor using best available engine."""
    return _internal_analyze(tensor)

__all__ = ["watch", "analyze", "ENGINE_STATUS"]
