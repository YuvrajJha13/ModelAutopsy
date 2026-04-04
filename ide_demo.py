import numpy as np
from modelautopsy import watch

@watch()
def training_step(data):
    # Simulate a slight gradient update
    return data * 0.99 + 0.01

tensor = np.random.randn(100).astype(np.float32)
result = training_step(tensor)
