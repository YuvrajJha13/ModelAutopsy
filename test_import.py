import numpy as np
import mlguardian

print(f"Engine Status: {mlguardian.ENGINE_STATUS}")

# Test Analyze
data = np.array([1.0, 2.0, np.nan], dtype=np.float32)
report = mlguardian.analyze(data)

print(f"Report: {report}")

# Test Decorator
@mlguardian.watch()
def step(x):
    return x * 2

print(step(np.array([1.0, 2.0])))
