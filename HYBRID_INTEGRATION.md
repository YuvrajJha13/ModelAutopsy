# Rust + C++ Hybrid Integration Guide

## Overview

ModelAutopsy now supports a **hybrid engine architecture** that seamlessly integrates both Rust and C++ for maximum performance. The system automatically selects the best available engine with graceful fallbacks.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Python Layer (debugger.py)             │
├─────────────────────────────────────────────────────┤
│  Engine Priority Manager                            │
│  ┌──────────────────────────────────────────────┐   │
│  │  1. Rust Engine (modelautopsy_rust)          │   │
│  │     - Parallel processing with Rayon         │   │
│  │     - Memory safe                            │   │
│  │     - Fastest for large datasets             │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  2. C++ Engine (_core_cpp)                   │   │
│  │     - SIMD optimized (-march=native)         │   │
│  │     - OpenMP multi-threading                 │   │
│  │     - pybind11 bindings                      │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  3. Pure Python Fallback                     │   │
│  │     - NumPy vectorized operations            │   │
│  │     - Always available                       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

**For C++ Engine:**
```bash
pip install pybind11
# Requires: g++ with C++17 support, OpenMP
```

**For Rust Engine:**
```bash
pip install maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

### Build Both Engines

```bash
# Using Makefile (recommended)
make build

# Or manually:
# 1. Build C++ engine
python setup.py build_ext --inplace

# 2. Build Rust engine
cd rust && maturin develop --release
```

## How Integration Works

### 1. Automatic Engine Selection

The `debugger.py` module automatically detects and prioritizes engines:

```python
from modelautopsy.debugger import analyze, _get_engine_priority

# System checks availability in order: Rust → C++ → Python
engine = _get_engine_priority()  # Returns: "rust", "cpp", or "python"

# Analysis uses the best available engine
result = analyze(numpy_array)
```

### 2. Graceful Degradation

If an engine fails at runtime, the system automatically falls back:

```python
def analyze(tensor):
    # Try Rust first
    if RUST_AVAILABLE:
        try:
            return modelautopsy_rust.rust_analyze(tensor)
        except Exception:
            pass  # Fall through to C++
    
    # Try C++
    if CPP_AVAILABLE:
        try:
            return _core_cpp.analyze(tensor)
        except Exception:
            pass  # Fall through to Python
    
    # Pure Python fallback (always works)
    return _analyze_python(tensor)
```

### 3. Shared Interface

Both engines expose identical functionality:

| Function | Rust | C++ | Python |
|----------|------|-----|--------|
| `analyze()` | ✓ | ✓ | ✓ |
| `has_failure()` | ✓ | ✓ | ✓ |
| Returns | Dict | Dict | Dict |

## Performance Comparison

Typical performance on 1M float32 elements:

```
Rust (Rayon parallel):   ~15-25ms
C++ (OpenMP + SIMD):     ~20-35ms
Pure Python (NumPy):     ~50-80ms
```

*Actual performance varies by hardware and data characteristics.*

## Testing Integration

Run the comprehensive integration test:

```bash
python test_hybrid_integration.py
```

This verifies:
- ✓ Engine detection
- ✓ Functional correctness
- ✓ Performance comparison (if both available)
- ✓ Decorator functionality

## Use Cases

### When to Use Rust
- Large datasets (>1M elements)
- Maximum parallelism needed
- Memory safety critical

### When to Use C++
- Existing C++ codebase integration
- Fine-grained SIMD control
- Lower dependency footprint

### When Python Fallback is Sufficient
- Small datasets (<100K elements)
- Prototyping/development
- Environments without compilers

## Configuration

Engine priority is cached after first check for performance:

```python
# Force re-evaluation (rarely needed)
from modelautopsy.debugger import _ENGINE_PRIORITY
_ENGINE_PRIORITY = None  # Clear cache
```

## Troubleshooting

### Rust Engine Not Loading
```bash
# Check Rust installation
rustc --version

# Rebuild with maturin
cd rust && maturin develop --release
```

### C++ Engine Not Loading
```bash
# Check compiler
g++ --version

# Rebuild extension
python setup.py build_ext --inplace
```

### Verify Both Engines
```python
try:
    import modelautopsy_rust
    print("✓ Rust available")
except ImportError:
    print("✗ Rust unavailable")

try:
    import _core_cpp
    print("✓ C++ available")
except ImportError:
    print("✗ C++ unavailable")
```

## Benefits of Hybrid Approach

1. **Maximum Performance**: Uses fastest available engine
2. **Reliability**: Multiple fallback layers ensure operation
3. **Flexibility**: Deploy in diverse environments
4. **Future-Proof**: Easy to add new engines
5. **Best of Both Worlds**: Rust safety + C++ maturity

## Example Usage

```python
import numpy as np
from modelautopsy.debugger import watch, analyze

# Automatic engine selection
data = np.random.randn(1_000_000).astype(np.float32)
stats = analyze(data)
print(f"Using optimal engine: {stats}")

# Monitoring with decorator
@watch(verbose=True, fail_on_nan=True)
def train_model(weights):
    # Automatically monitored for NaN/Inf
    return weights * 0.9

train_model(data)
```

---

**Conclusion**: ModelAutopsy's hybrid Rust+C++ architecture provides enterprise-grade reliability with cutting-edge performance, automatically adapting to your deployment environment.
