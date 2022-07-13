use pyo3::{exceptions::PyTypeError, pyfunction, types::PyList, PyResult};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[pyfunction]
pub fn norm(xs: &PyList) -> PyResult<f64> {
    match xs.extract::<Vec<f64>>() {
        Ok(xs_vec) => Ok(xs_vec.par_iter().map(|x| x * x).sum::<f64>().sqrt()),
        _ => Err(PyTypeError::new_err(
            "Parameter cannot be converted to list of floats.",
        )),
    }
}
