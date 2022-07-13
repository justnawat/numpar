use pyo3::{pyfunction, types::PyList, PyResult};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[pyfunction]
pub fn norm(xs: &PyList) -> PyResult<f64> {
    let xs_vec: Vec<f64> = xs.extract()?;
    Ok(xs_vec.par_iter().map(|x| x * x).sum::<f64>().sqrt())
}
