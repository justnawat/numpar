use pyo3::prelude::{pyfunction, PyResult};
use pyo3::types::PyList;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;

#[pyfunction]
pub fn outer(xs: &PyList, ys: &PyList) -> PyResult<Vec<Vec<f64>>> {
    let xs: Vec<f64> = xs.extract()?;
    let ys: Vec<f64> = ys.extract()?;

    Ok(xs
        .par_iter()
        .map(|&x| ys.par_iter().map(|&y| y * x).collect())
        .collect())
}
