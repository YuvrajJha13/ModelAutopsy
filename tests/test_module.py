import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modelautopsy
from modelautopsy.debugger import analyze

def test_empty_tensor():
    """Empty tensor should return zeros without crashing."""
    result = modelautopsy.analyze(np.array([], dtype=np.float32))
    assert result["nan_count"] == 0
    assert result["inf_count"] == 0
    assert result["valid_count"] == 0
    assert result["mean"] == 0.0

def test_healthy_tensor():
    """Test analysis of healthy tensor data."""
    data = np.random.randn(1000).astype(np.float32)
    report = modelautopsy.analyze(data)
    assert report["nan_count"] == 0
    assert report["inf_count"] == 0
    assert report["valid_count"] == 1000

def test_nan_detection():
    """Test NaN detection in tensors."""
    data = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    report = modelautopsy.analyze(data)
    assert report["nan_count"] == 1
    assert report["valid_count"] == 3

def test_inf_detection():
    """Test Inf detection in tensors."""
    data = np.array([1.0, np.inf, -np.inf], dtype=np.float32)
    report = modelautopsy.analyze(data)
    assert report["inf_count"] == 2
    # Mean should be calculated only from valid values
    assert report["mean"] == 1.0 

def test_mixed_failures():
    """Test large dataset with mixed failures."""
    size = 1000000
    data = np.random.randn(size).astype(np.float32)
    data[0] = np.nan
    data[500] = np.inf
    data[-1] = np.nan
    
    report = modelautopsy.analyze(data)
    
    assert report["nan_count"] == 2
    assert report["inf_count"] == 1
    assert report["valid_count"] == size - 3

def test_wrong_dimensions():
    """2D tensors should be automatically flattened and processed."""
    data = np.random.randn(10, 10).astype(np.float32)
    # Should handle gracefully by flattening
    result = modelautopsy.analyze(data)
    assert result is not None
    assert result["valid_count"] == 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])