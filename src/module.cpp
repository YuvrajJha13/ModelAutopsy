#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "../include/stats.h"

namespace py = pybind11;

py::object analyze_f32(py::array_t<float> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 1) throw std::runtime_error("Input must be 1-dimensional.");
    
    float *ptr = static_cast<float*>(buf.ptr);
    DebugReport report = compute_statistics<float>(ptr, buf.size);

    py::dict result;
    result["nan_count"] = report.nan_count;
    result["inf_count"] = report.inf_count;
    result["valid_count"] = report.valid_count;
    result["mean"] = report.mean;
    result["variance"] = report.variance;
    result["l2_norm"] = report.l2_norm;
    result["min_val"] = report.min_val;
    result["max_val"] = report.max_val;
    return result;
}

py::object analyze_f64(py::array_t<double> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 1) throw std::runtime_error("Input must be 1-dimensional.");
    
    double *ptr = static_cast<double*>(buf.ptr);
    DebugReport report = compute_statistics<double>(ptr, buf.size);

    py::dict result;
    result["nan_count"] = report.nan_count;
    result["inf_count"] = report.inf_count;
    result["valid_count"] = report.valid_count;
    result["mean"] = report.mean;
    result["variance"] = report.variance;
    result["l2_norm"] = report.l2_norm;
    result["min_val"] = report.min_val;
    result["max_val"] = report.max_val;
    return result;
}

py::object analyze(py::array input) {
    if (input.dtype().is(pybind11::dtype::of<float>())) {
        return analyze_f32(input.cast<py::array_t<float>>());
    } 
    else if (input.dtype().is(pybind11::dtype::of<double>())) {
        return analyze_f64(input.cast<py::array_t<double>>());
    }
    else {
        throw std::runtime_error("Unsupported dtype. Use float32 or float64.");
    }
}

// Renamed module to _core_cpp to avoid conflict with Python package
PYBIND11_MODULE(_core_cpp, m) {
    m.doc() = "MLGuardian C++ Engine";
    m.def("analyze", &analyze, "C++ Analysis (float32/float64)");
}
