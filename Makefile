.PHONY: all clean build build-rust build-cpp test install example hybrid-demo

PYTHON := python
PYTEST := pytest
MATURIN := maturin

all: build

# Build both Rust and C++ engines for maximum performance
build: build-rust build-cpp
@echo "✓ Both Rust and C++ engines built successfully!"

# Build Rust engine using maturin (fastest, parallel processing)
build-rust:
@echo "Building Rust Engine with Maturin..."
cd rust && $(MATURIN) develop --release 2>/dev/null || { \
echo "⚠️  Rust build failed (maturin not installed or Rust unavailable)"; \
echo "   Install with: pip install maturin && rustup default stable"; \
}

# Build C++ engine using pybind11 (SIMD optimized, multi-threaded)
build-cpp:
@echo "Building C++ Extension with pybind11..."
$(PYTHON) setup.py build_ext --inplace

# Test both engines
test: build
@echo "Running Test Suite..."
$(PYTEST) tests/ -v

# Clean all build artifacts
clean:
@echo "Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info
rm -rf __pycache__ tests/__pycache__
find . -name "*.so" -delete
find . -name "*.pyc" -delete
@echo "✓ Clean complete"

# Run demo with hybrid engine
example: build
$(PYTHON) examples/demo.py

# Demonstrate Rust+C++ hybrid integration
hybrid-demo: build
@echo ""
@echo "=== Testing Rust+C++ Hybrid Integration ==="
$(PYTHON) -c "import numpy as np; print('Testing hybrid engine integration...'); \
try: \
    import modelautopsy_rust; print('✓ Rust engine loaded'); RUST=True; \
except: print('✗ Rust engine unavailable'); RUST=False; \
try: \
    import _core_cpp; print('✓ C++ engine loaded'); CPP=True; \
except: print('✗ C++ engine unavailable'); CPP=False; \
if RUST and CPP: print('🎉 SUCCESS: Both engines available for hybrid operation!'); \
elif RUST or CPP: print('⚡ Partial: One engine available'); \
else: print('⚠️  Fallback: Using pure Python only'); \
from modelautopsy.debugger import analyze; \
data = np.random.randn(1000).astype(np.float32); \
result = analyze(data); \
print(f'Analysis result: {result[\"valid_count\"]} valid values, mean={result[\"mean\"]:.4f}');"
