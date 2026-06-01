use cross_validation_for_time_series_data_core::rolling_window_cv_bounds;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn rolling_window_cv_bounds_py<'py>(
    py: Python<'py>,
    n_samples: usize,
    window_size: usize,
    step_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let bounds = rolling_window_cv_bounds(n_samples, window_size, step_size);
    let flat: Vec<f64> = bounds.into_iter().flat_map(|b| b.map(|x| x as f64)).collect();
    Ok(flat.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (n_samples, window_size, step_size, iterations=50_000))]
fn bench_kernel_py(
    n_samples: usize,
    window_size: usize,
    step_size: usize,
    iterations: usize,
) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = rolling_window_cv_bounds(n_samples, window_size, step_size);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn cross_validation_for_time_series_data_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_window_cv_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
