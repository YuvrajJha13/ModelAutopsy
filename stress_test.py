from modelautopsy import watch
import numpy as np

@watch()
def pipeline(data):
    # Complex pipeline simulation
    data = data / 10.0
    data = data ** 2
    return data

# Stress test: Float64, Lists, Arrays
input_data = np.random.randn(10000).astype(np.float64) # Wrong type
result = pipeline(input_data)
