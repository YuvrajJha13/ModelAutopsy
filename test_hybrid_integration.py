"""
Hybrid Integration Test: Demonstrates Rust + C++ coexistence
"""
import numpy as np

print("=" * 60)
print("Rust + C++ Hybrid Integration Test")
print("=" * 60)

# Test 1: Check engine availability
print("\n1. Engine Availability:")
print("-" * 40)

rust_available = False
cpp_available = False

try:
    import modelautopsy_rust
    rust_available = True
    print("✓ Rust engine loaded successfully")
except ImportError as e:
    print(f"✗ Rust engine not available: {e}")

try:
    import _core_cpp
    cpp_available = True
    print("✓ C++ engine loaded successfully")
except ImportError as e:
    print(f"✗ C++ engine not available: {e}")

if rust_available and cpp_available:
    print("\n🎉 SUCCESS: Both engines available for hybrid operation!")
elif rust_available or cpp_available:
    print("\n⚡ Partial: One engine available (fallback chain active)")
else:
    print("\n⚠️  Fallback: Using pure Python only")

# Test 2: Functional test with debugger
print("\n2. Functional Test:")
print("-" * 40)

from modelautopsy.debugger import analyze, _get_engine_priority

# Create test data
data = np.random.randn(10000).astype(np.float32)
data[100] = np.nan  # Inject some failures
data[200] = np.inf

result = analyze(data)
engine = _get_engine_priority()

print(f"Active engine: {engine.upper()}")
print(f"Test data size: {len(data)} elements")
print(f"NaN count detected: {result['nan_count']}")
print(f"Inf count detected: {result['inf_count']}")
print(f"Valid count: {result['valid_count']}")
print(f"Mean: {result['mean']:.6f}")
print(f"L2 Norm: {result['l2_norm']:.6f}")

# Test 3: Performance comparison (if both engines available)
if rust_available and cpp_available:
    print("\n3. Performance Comparison:")
    print("-" * 40)
    import time
    
    large_data = np.random.randn(1_000_000).astype(np.float32)
    
    # Time Rust
    start = time.perf_counter()
    for _ in range(10):
        modelautopsy_rust.rust_analyze(large_data)
    rust_time = (time.perf_counter() - start) / 10
    
    # Time C++
    start = time.perf_counter()
    for _ in range(10):
        _core_cpp.analyze(large_data)
    cpp_time = (time.perf_counter() - start) / 10
    
    print(f"Rust engine:  {rust_time*1000:.2f}ms per iteration (1M elements)")
    print(f"C++ engine:   {cpp_time*1000:.2f}ms per iteration (1M elements)")
    
    if rust_time < cpp_time:
        print(f"→ Rust is {cpp_time/rust_time:.2f}x faster")
    else:
        print(f"→ C++ is {rust_time/cpp_time:.2f}x faster")

# Test 4: Decorator test
print("\n4. Decorator Test:")
print("-" * 40)

from modelautopsy.debugger import watch

@watch(verbose=True)
def process_tensor(x):
    """Sample function that might introduce NaN/Inf"""
    return x * 2.0 + np.sin(x)

test_input = np.random.randn(100).astype(np.float32)
output = process_tensor(test_input)
print(f"✓ Decorator executed successfully")

print("\n" + "=" * 60)
print("Integration Test Complete!")
print("=" * 60)
