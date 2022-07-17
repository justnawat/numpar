use pyo3::prelude::*;
mod cwslice;
mod my_util;

mod vector_ops;
use vector_ops::*;

mod mat_trace;
use mat_trace::trace;

mod mat_transpose;
use mat_transpose::transpose;

mod mat_det;
use mat_det::det;

/// A Python module implemented in Rust.
#[pymodule]
fn numpar(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    Ok(())
}
