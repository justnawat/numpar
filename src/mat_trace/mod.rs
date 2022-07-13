use pyo3::exceptions::PyTypeError;
use pyo3::prelude::{pyfunction, PyResult};
use pyo3::types::PyList;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;

#[pyfunction]
pub fn trace(matrix: &PyList) -> PyResult<f64> {
    match matrix.extract::<Vec<Vec<f64>>>() {
        Err(_) => Err(PyTypeError::new_err("Parameter not a matrix.")),
        Ok(r_matrix) => {
            let n = r_matrix.len();
            let lens = r_matrix.par_iter().map(|row| row.len()).sum();

            if n == 0 {
                Ok(0.0)
            } else if n * n != lens {
                Err(PyTypeError::new_err("Matrix not square."))
            } else {
                Ok(r_matrix.par_iter().enumerate().map(|(i, row)| row[i]).sum())
            }
        }
    }
}
