# ModelAutopsy

ModelAutopsy is a high-performance, hybrid-engine (Rust/C++) Machine Learning debugging library. It utilizes advanced parallelism to perform multi-threaded analysis of gradient tensors, instantly detecting NaN, Infinity, Vanishing, and Exploding values in massive datasets.

## Features

*   **Hybrid Engine:** Rust (Safety) + C++ (Speed).
*   **Deep Analysis:** Instantly detects NaN, Inf, Vanishing, and Exploding values.
*   **Advanced Metrics:** Calculates L2 Norm, Variance, Mean, Min/Max for statistical health.
*   **IDE Integration:** Includes Rich Console UI and Python Decorators for real-time training loops.
*   **Zero-Copy:** Direct memory access for maximum performance on large tensors.

## Installation

### Pip Install (Recommended)

```bash
pip install modelautopsy
