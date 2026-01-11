import numpy as np
from debugger import watch, RUST_AVAILABLE, CPP_AVAILABLE

print(f"Rust Available: {RUST_AVAILABLE}")
print(f"C++ Available: {CPP_AVAILABLE}")

@watch(verbose=True)
def safe_layer(input_data):
    # Simulate a critical safety check
    input_data[0] = np.nan 
    return input_data * 2

# Run
data = np.random.randn(1000).astype(np.float32)
result = safe_layer(data)

if RUST_AVAILABLE:
    print("\n🛡️  Running in Safety Mode (Rust Protected)")
else:
    print("\n🚀 Running in Speed Mode (C++ Optimized)")
