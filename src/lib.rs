use pyo3::prelude::*;
mod cwslice;
mod my_util;

mod dot_prod;
use dot_prod::dot;

mod outer_prod;
use outer_prod::outer;

mod vec_norm;
use vec_norm::norm;

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
