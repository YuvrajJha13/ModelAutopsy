use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn rust_has_failure(data: PyReadonlyArray1<f32>) -> PyResult<bool> {
    let slice = data.as_slice()?;
    let fail = slice.iter().any(|&x| x.is_nan() || x.is_infinite());
    Ok(fail)
}

#[pyfunction]
fn rust_analyze(data: PyReadonlyArray1<f32>) -> PyResult<StatsReport> {
    let slice = data.as_slice()?;
    
    let mut nan = 0; let mut inf = 0;
    let mut sum = 0.0_f64; let mut sum_sq = 0.0_f64;
    let mut min = f64::INFINITY; let mut max = f64::NEG_INFINITY;
    let mut valid = 0;

    for &x in slice {
        if x.is_nan() { nan += 1; }
        else if x.is_infinite() { inf += 1; }
        else {
            let v = x as f64;
            sum += v; sum_sq += v * v;
            if v < min { min = v; }
            if v > max { max = v; }
            valid += 1;
        }
    }

    let mean = if valid > 0 { sum / valid as f64 } else { 0.0 };
    let l2 = sum_sq.sqrt();
    let variance = if valid > 1 { (sum_sq / valid as f64) - (mean * mean) } else { 0.0 };

    Ok(StatsReport {
        nan_count: nan, inf_count: inf, valid_count: valid,
        mean, variance, l2_norm: l2,
        min_val: if min.is_finite() { min } else { 0.0 },
        max_val: if max.is_finite() { max } else { 0.0 },
    })
}

#[pyclass]
pub struct StatsReport {
    #[pyo3(get, set)] pub nan_count: usize,
    #[pyo3(get, set)] pub inf_count: usize,
    #[pyo3(get, set)] pub valid_count: usize,
    #[pyo3(get, set)] pub mean: f64,
    #[pyo3(get, set)] pub variance: f64,
    #[pyo3(get, set)] pub l2_norm: f64,
    #[pyo3(get, set)] pub min_val: f64,
    #[pyo3(get, set)] pub max_val: f64,
}

#[pymodule]
fn mlguardian_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_has_failure, m)?)?;
    m.add_function(wrap_pyfunction!(rust_analyze, m)?)?;
    m.add_class::<StatsReport>()?;
    Ok(())
}
