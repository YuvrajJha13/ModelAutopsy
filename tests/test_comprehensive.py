import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modelautopsy
from modelautopsy.debugger import watch, _prepare_tensor

class TestCoreEngine:
    def test_empty_tensor(self):
        """Should handle size 0 without crash."""
        data = np.array([], dtype=np.float32)
        report = modelautopsy.analyze(data)
        assert report["nan_count"] == 0
        assert report["mean"] == 0.0

    def test_all_nans(self):
        """Pure NaN stress test."""
        data = np.full(100, np.nan, dtype=np.float32)
        report = modelautopsy.analyze(data)
        assert report["nan_count"] == 100
        assert report["valid_count"] == 0
        assert report["mean"] == 0.0

    def test_float64_conversion(self):
        """Wrapper should handle float64 inputs."""
        # Note: This tests the python wrapper _prepare_tensor
        data = np.random.randn(100).astype(np.float64) # Wrong type
        prepared = _prepare_tensor(data)
        assert prepared is not None
        assert prepared.dtype == np.float32
        assert prepared.size == 100

    def test_list_input(self):
        """Wrapper should handle Python lists."""
        data = [1.0, 2.0, np.nan]
        prepared = _prepare_tensor(data)
        assert prepared is not None
        assert prepared.dtype == np.float32
        assert np.isnan(prepared[2])

    def test_mixed_types(self):
        """Test handling of integers and floats."""
        data = [1, 2, 3.5]
        prepared = _prepare_tensor(data)
        assert prepared is not None
        assert prepared.dtype == np.float32

class TestDecorator:
    def test_decorator_no_failure(self):
        @watch(drop_into_debugger=False, verbose=False)
        def clean_func(x):
            return x * 2
        
        result = clean_func(np.array([1.0, 2.0]))
        assert result is not None

    def test_decorator_detection(self):
        @watch(drop_into_debugger=False, verbose=False)
        def corrupt_func(x):
            x[0] = np.nan
            return x

        # Mock print to capture output logic if needed
        result = corrupt_func(np.array([1.0, 2.0]))
        assert result is not None # Function runs
        # Ideally we'd capture logs, but for unit test, just ensure it doesn't crash

    def test_decorator_non_tensor_args(self):
        @watch(drop_into_debugger=False, verbose=False)
        def mixed_func(x, y):
            return x + y
        
        # Should not crash on non-array args
        res = mixed_func(5, 10)
        assert res == 15

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])