use pyo3::prelude::{pyfunction, PyResult};
use pyo3::types::PyList;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;

fn comp_wise_mult(xs: &Vec<f64>, ys: &Vec<f64>) -> Vec<f64> {
	xs.par_iter().zip(ys.par_iter()).map(|(&x, &y)| x*y).collect()
}

#[pyfunction]
pub fn outer(xs: &PyList, ys: &PyList) -> PyResult<Vec<Vec<f64>>> {
    let xs: Vec<f64> = xs.extract()?;
    let ys: Vec<f64> = ys.extract()?;

    let n = ys.len();
    let out: Vec<Vec<f64>> = xs
        .par_iter()
        .map(|&xi| {
            let mut v = Vec::new();
            v.resize_with(n, || xi);
			comp_wise_mult(&v, &ys)
        })
        .collect();

    Ok(out)
}
