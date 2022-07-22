use pyo3::prelude::*;

mod cwslice;
mod linear_eqn_ops;
mod matrix_mult;
mod matrix_ops;
mod my_util;
mod vector_ops;

use linear_eqn_ops::*;
use matrix_mult::*;
use matrix_ops::*;
use vector_ops::*;

/// A Python module implemented in Rust.
#[pymodule]
fn numpar(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;

    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;

    // m.add_function(wrap_pyfunction!(matmul1, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;

    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_rank, m)?)?;

    Ok(())
}
