use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn rust_has_failure(data: PyReadonlyArray1<f32>) -> PyResult<bool> {
    let slice = data.as_slice()?;
    let fail = slice.par_iter().any(|&x| x.is_nan() || x.is_infinite());
    Ok(fail)
}

#[pyfunction]
fn rust_analyze(data: PyReadonlyArray1<f32>) -> PyResult<StatsReport> {
    let slice = data.as_slice()?;
    let (nan, inf, sum, sum_sq, min, max, valid) = slice.par_iter().fold(
        || (0usize, 0usize, 0.0_f64, 0.0_f64, f64::INFINITY, f64::NEG_INFINITY, 0usize),
        |acc, &x| {
            let (nan, inf, sum, sq, min, max, valid) = acc;
            if x.is_nan() { (nan+1, inf, sum, sq, min, max, valid) }
            else if x.is_infinite() { (nan, inf+1, sum, sq, min, max, valid) }
            else { 
                let v = x as f64; 
                (nan, inf, sum+v, sq+v*v, min.min(v), max.max(v), valid+1) 
            }
        },
        |a, b| (a.0+b.0, a.1+b.1, a.2+b.2, a.3+b.3, a.4.min(b.4), a.5.max(b.5), a.6+b.6)
    );

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
fn modelautopsy_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_has_failure, m)?)?;
    m.add_function(wrap_pyfunction!(rust_analyze, m)?)?;
    m.add_class::<StatsReport>()?;
    Ok(())
}
